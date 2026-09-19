"""
Ratatoskr — capability-dispatching scheduler.

Polls one Supabase Storage bucket, registers any new object as an `assets`
row, then for each object resolves a workflow and runs it.

Two dispatch paths, tried in order:

    contracts  — the `contracts` table holds versioned, ordered pipelines
                 ({"contract_id", "version", "trigger", "match",
                 "pipeline"}). For each asset, the most specific enabled
                 `trigger='ingest'` contract whose `match.mimetype`
                 matches the asset's mime type is selected (exact match
                 beats a wildcard). Its `pipeline` is an ordered list of
                 stages:
                     {"id": "text", "processor": "gunthercox",
                      "operation": "ocr", "config": {...}, "condition": {...}}
                 `(processor, operation)` resolves to exactly one
                 `processors` row via that row's `service` + `capability`
                 columns. `config` is a per-stage override merged over
                 that processor's `default_config` — this is what lets
                 the same processor row appear in a pipeline more than
                 once with different settings (e.g. gunthercox-ocr called
                 once with no override, once with
                 {"engine": "paddleocr", "lang": "ch"}). `condition`, if
                 present, overrides the processor row's own
                 `dispatch_condition` for this stage only; if absent, the
                 row's own condition (if any) still applies.
                 Match is mimetype-only for now — metadata-based
                 matching would need metadata that a pipeline stage
                 hasn't necessarily produced yet at resolution time, so
                 that case stays handled by per-stage `condition`
                 (checked against already-stored interpretations)
                 instead of being promoted to top-level `match`.

    run_by_default — the original mechanism: any enabled processor row
                 flagged `run_by_default` whose `input_mime_types`
                 matches. Used only when no contract matches the asset's
                 mime type, so mime types without a defined contract yet
                 keep working exactly as before.

Because a contract can invoke the same processor more than once per
asset (different stages, different config), idempotency and the
`processing_jobs` uniqueness are keyed on `(asset_id, processor_id,
stage_id)`, not just `(asset_id, processor_id)`. The legacy path uses a
fixed `stage_id` of `'_default'`, preserving its original one-job-ever
semantics.

Dispatch is keyed off `processors.type`, not `capability`:
    internal  — runs a Python function looked up by `capability`
                (currently just `metadata`: Pillow image EXIF / pypdf
                document info). No external service involved.
    http      — runs a *generic* handler driven entirely by data on the
                processor row: `endpoint_url` + `request_template`
                describe the outbound HTTP request, `output_mapping`
                describes which fields of the JSON response to lift into
                `asset_interpretations.raw_payload`/`confidence`. Adding a
                new HTTP-backed processor (OCR engine, BentoBox caption,
                CLIP embeddings, whatever comes next) is a `processors`
                INSERT, not a code change.

`request_template` shape (JSONB), interpreted by handle_http_json:
    {
      "method": "POST",
      "headers": {"accept": "application/json"},
      "multipart": {
        "<field name>": {"source": "asset_bytes", "content_type": "..."},
        "<field name>": {"source": "config_json"}
      }
    }
`source` is one of:
    asset_bytes  — the downloaded file body (uses asset's mime_type
                   unless the field overrides content_type)
    config_json  — the resolved config (default_config ⊕ job override),
                   JSON-encoded
    config_field — a single named key out of the resolved config (e.g.
                   "engine", "lang"), sent as its own multipart field

`output_mapping` shape (JSONB):
    {"text_path": "caption", "confidence_path": "confidence"}
Dotted paths into the JSON response; missing/absent paths just resolve to
None rather than raising. The full response is always stored in
raw_payload regardless of what output_mapping pulls out.

dispatch_condition / stage condition (same shape either way):
    {"source_capability": "metadata", "path": "exif.Make", "op": "in",
     "value": ["Canon", "NIKON CORPORATION", "SONY"]}
Lets a stage depend on another capability's *already-stored*
interpretation for the same asset. If the dependency hasn't produced an
interpretation yet, the stage is skipped for this poll cycle (not
treated as an error) and picked back up once it exists — the capability
index used to resolve `source_capability` is built from every enabled
processor, not just run_by_default ones, so a stage can depend on a
capability that only ever runs via a contract.

Per-processor settings live in `processors`, not env vars: `config_schema`
documents what a processor's API accepts (informal — nothing at dispatch
time enforces it, it's there for a human or a future Skuld config UI),
`default_config` holds the actual defaults, and either
`processing_jobs.configuration` (legacy path) or a stage's `config`
(contract path) supplies a per-job override. Dispatch resolves
default_config merged with that override — override values always win,
unvalidated. We don't own most of these downstream APIs, so a bad
key/value either gets ignored by the processor's own API or comes back as
an error response, which surfaces through the ordinary job-failure path
rather than being silently dropped before the request is even sent.

Required env vars:
    SUPABASE_URL            e.g. https://your-project.supabase.co
    SUPABASE_KEY            service_role key (Storage read + table access)
    SUPABASE_BUCKET         exact bucket name to watch
    POLL_INTERVAL_SECONDS   default: 10
"""
import io
import json
import os
import time
import traceback
from typing import Optional

import requests
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pypdf import PdfReader
from supabase import Client, create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SUPABASE_BUCKET = os.environ["SUPABASE_BUCKET"]

POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "10"))

DEFAULT_STAGE_ID = "_default"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# small shared helper — dotted-path lookup into JSON payloads
# ============================================================

def _dig(payload, path: Optional[str]):
    """Walk a dotted path ('exif.Make', 'result.caption') into a dict,
    returning None on any missing key or non-dict intermediate rather
    than raising. Used for both output_mapping and dispatch_condition."""
    if not path or not isinstance(payload, dict):
        return None
    cur = payload
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


# ============================================================
# processor / contract discovery
# ============================================================

def get_all_enabled_processors():
    """Every enabled processor, regardless of run_by_default. Used to
    build both the capability index (for dispatch_condition/stage
    condition sourcing) and the service+capability lookup contracts
    resolve against — a contract stage can invoke a processor that isn't
    anyone's run_by_default."""
    result = (
        supabase.table("processors")
        .select(
            "id, name, capability, service, type, endpoint_url, input_mime_types, "
            "config_schema, default_config, request_template, output_mapping, "
            "dispatch_condition, run_by_default"
        )
        .eq("enabled", True)
        .execute()
    )
    return result.data


def get_contracts():
    """Enabled, ingest-triggered contracts. Refetched every poll, same
    reasoning as processors — a newly-added or disabled contract takes
    effect on the next cycle without a restart."""
    result = (
        supabase.table("contracts")
        .select("id, contract_id, version, trigger, match, pipeline")
        .eq("enabled", True)
        .eq("trigger", "ingest")
        .execute()
    )
    return result.data


def mime_matches(pattern: str, mime_type: str) -> bool:
    if not mime_type:
        return False
    if pattern.endswith("/*"):
        return mime_type.startswith(pattern[:-1])
    return pattern == mime_type


def applicable_processors(processors: list, mime_type: str) -> list:
    return [
        p for p in processors
        if any(mime_matches(pat, mime_type) for pat in (p.get("input_mime_types") or []))
    ]


def build_capability_index(processors: list) -> dict:
    """capability -> [processor ids], built from ALL enabled processors
    (not just run_by_default ones) so a condition can depend on a
    capability that only runs inside a contract."""
    idx = {}
    for p in processors:
        idx.setdefault(p["capability"], []).append(p["id"])
    return idx


def build_processor_lookup(processors: list) -> dict:
    """(service, capability) -> processor row, for resolving a contract
    stage's (processor, operation) pair. If more than one enabled row
    shares a (service, capability) pair, prefer the run_by_default one
    and log — this shouldn't happen post-consolidation, but fail loud
    rather than silently picking an arbitrary row if it ever does."""
    lookup = {}
    for p in processors:
        key = (p.get("service"), p.get("capability"))
        existing = lookup.get(key)
        if existing is None:
            lookup[key] = p
        elif p.get("run_by_default") and not existing.get("run_by_default"):
            lookup[key] = p
        else:
            print(
                f"[scheduler] multiple processors for service='{key[0]}' "
                f"capability='{key[1]}' — using '{lookup[key]['name']}', ignoring '{p['name']}'"
            )
    return lookup


def mimetype_specificity(pattern: Optional[str], mime_type: str) -> Optional[int]:
    """None if pattern doesn't match; 2 for an exact mimetype match, 1
    for a wildcard match. Higher wins when more than one contract could
    apply to the same asset."""
    if not pattern or not mime_matches(pattern, mime_type):
        return None
    return 2 if pattern == mime_type else 1


def match_contract(contracts: list, mime_type: str) -> Optional[dict]:
    best = None
    best_score = -1
    for c in contracts:
        pattern = (c.get("match") or {}).get("mimetype")
        score = mimetype_specificity(pattern, mime_type)
        if score is None:
            continue
        if score > best_score:
            best, best_score = c, score
        elif score == best_score:
            print(
                f"[scheduler] ambiguous contract match for mime '{mime_type}': "
                f"keeping '{best['contract_id']}'@{best['version']}, ignoring "
                f"'{c['contract_id']}'@{c['version']}'"
            )
    return best


def resolve_pipeline_stages(contract: dict, processor_lookup: dict) -> list:
    """[(stage, processor), ...] in the contract's declared pipeline
    order. A stage referencing a (processor, operation) pair with no
    matching enabled processors row is logged and skipped, not fatal —
    the rest of the pipeline still runs."""
    stages = []
    for stage in contract.get("pipeline") or []:
        key = (stage.get("processor"), stage.get("operation"))
        processor = processor_lookup.get(key)
        if processor is None:
            print(
                f"[scheduler] contract '{contract['contract_id']}'@{contract['version']} "
                f"stage '{stage.get('id')}': no enabled processor for "
                f"service='{key[0]}' operation='{key[1]}' — skipping stage"
            )
            continue
        stages.append((stage, processor))
    return stages


# ============================================================
# per-processor configuration — plain override-merge, no validation
# ============================================================

def resolve_config(processor: dict, override: dict) -> dict:
    """Merge this processor's default_config with an override (job's
    stored `configuration`, or a contract stage's `config`). Override
    values always win. Deliberately unvalidated: config_schema is
    documentation, not an enforcement gate — we don't control most of
    these downstream APIs, so a bad key/value either gets ignored by the
    processor's own API or comes back as an error response, surfacing
    through mark_job("failed", ...) rather than disappearing silently."""
    return {**(processor.get("default_config") or {}), **(override or {})}


# ============================================================
# conditional dispatch — gate a stage on another's stored output
# ============================================================

def get_source_payload(asset_id: str, processor_ids: list) -> Optional[dict]:
    """Fetch the raw_payload of any existing interpretation on this asset
    from one of the given processor ids. Returns None if none has run
    yet — the caller treats that as "dependency not ready", not an
    error. Queried fresh each call, so a dependency produced earlier in
    the SAME poll cycle (an earlier stage in pipeline order) is already
    visible here — no need to wait for the next poll."""
    if not processor_ids:
        return None
    result = (
        supabase.table("asset_interpretations")
        .select("raw_payload")
        .eq("asset_id", asset_id)
        .in_("processor_id", processor_ids)
        .limit(1)
        .execute()
    )
    return result.data[0]["raw_payload"] if result.data else None


def condition_met(asset_id: str, condition: Optional[dict], capability_index: dict) -> bool:
    if not condition:
        return True  # no condition declared — always applicable

    source_ids = capability_index.get(condition.get("source_capability", "metadata"), [])
    payload = get_source_payload(asset_id, source_ids)
    if payload is None:
        return False  # dependency hasn't produced an interpretation yet — retry next poll

    value = _dig(payload, condition.get("path"))
    op = condition.get("op", "exists")
    target = condition.get("value")

    if op == "exists":
        return value is not None
    if op == "not_exists":
        return value is None
    if op == "eq":
        return value == target
    if op == "neq":
        return value != target
    if op == "in":
        return value in (target or [])
    if op == "not_in":
        return value not in (target or [])
    if op == "gt":
        return value is not None and value > target
    if op == "gte":
        return value is not None and value >= target
    if op == "lt":
        return value is not None and value < target
    if op == "lte":
        return value is not None and value <= target

    print(f"[scheduler] unknown dispatch_condition op '{op}' — treating as not met")
    return False


# ============================================================
# storage / assets
# ============================================================

def list_bucket_files():
    """Flat listing of the configured bucket. Assumes no subfolders."""
    return supabase.storage.from_(SUPABASE_BUCKET).list()


def get_or_create_asset(file_info: dict):
    """Look up the assets row for this Storage object, creating it if new.
    Supabase Storage's S3-compatible eTags come back wrapped in literal
    double quotes — strip them before storing as checksum."""
    name = file_info["name"]
    storage_path = f"{SUPABASE_BUCKET}/{name}"
    meta = file_info.get("metadata") or {}
    mime_type = meta.get("mimetype", "application/octet-stream")
    checksum = (meta.get("eTag") or "").strip('"')
    byte_size = meta.get("size")

    existing = (
        supabase.table("assets")
        .select("id, mime_type")
        .eq("storage_path", storage_path)
        .execute()
    )
    if existing.data:
        return existing.data[0]["id"], existing.data[0]["mime_type"]

    inserted = (
        supabase.table("assets")
        .insert({
            "storage_path": storage_path,
            "mime_type": mime_type,
            "checksum": checksum,
            "byte_size": byte_size,
            "original_filename": name,
            "display_name": name,
        })
        .execute()
    )
    row = inserted.data[0]
    return row["id"], row["mime_type"]


# ============================================================
# job bookkeeping — keyed on (asset_id, processor_id, stage_id)
# ============================================================

def already_processed(asset_id: str, processor_id: str, stage_id: str = DEFAULT_STAGE_ID) -> bool:
    """True if a COMPLETE job already exists for this exact
    (asset, processor, stage). Checking job status rather than
    asset_interpretations existence means a previously FAILED stage is
    retried on the next poll rather than treated as done."""
    existing = (
        supabase.table("processing_jobs")
        .select("id")
        .eq("asset_id", asset_id)
        .eq("processor_id", processor_id)
        .eq("stage_id", stage_id)
        .eq("status", "complete")
        .execute()
    )
    return bool(existing.data)


def create_job(
    asset_id: str,
    processor_id: str,
    stage_id: str = DEFAULT_STAGE_ID,
    contract_id: Optional[str] = None,
    contract_version: Optional[int] = None,
    configuration: Optional[dict] = None,
) -> dict:
    """Upsert on (asset_id, processor_id, stage_id) — a row for this
    exact stage may already exist from an earlier attempt. configuration
    is only included in the upsert payload when explicitly provided, so
    the legacy path (which never passes it) leaves any manually-set
    processing_jobs.configuration alone on retry instead of clobbering it."""
    fields = {
        "asset_id": asset_id,
        "processor_id": processor_id,
        "stage_id": stage_id,
        "status": "processing",
        "error_message": None,
        "contract_id": contract_id,
        "contract_version": contract_version,
    }
    if configuration is not None:
        fields["configuration"] = configuration
    row = (
        supabase.table("processing_jobs")
        .upsert(fields, on_conflict="asset_id,processor_id,stage_id")
        .execute()
    )
    return row.data[0]


def mark_job(job_id: str, status: str, error_message: Optional[str] = None):
    fields = {"status": status}
    if error_message is not None:
        fields["error_message"] = error_message
    supabase.table("processing_jobs").update(fields).eq("id", job_id).execute()


def store_interpretation(
    asset_id: str, job_id: str, processor_id: str, model_name: str,
    raw_payload: dict, confidence: Optional[float] = None,
):
    supabase.table("asset_interpretations").insert({
        "asset_id": asset_id,
        "job_id": job_id,
        "processor_id": processor_id,
        "model_name": model_name,
        "model_version": "unknown",
        "raw_payload": raw_payload,
        "confidence": confidence,
    }).execute()


# ============================================================
# internal capability — metadata (Pillow EXIF / pypdf), no HTTP involved
# ============================================================

def _sanitize(value):
    """Coerce PIL/pypdf value types (IFDRational, bytes, nested tuples)
    into plain JSON-serializable types for storage in raw_payload."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return value.hex()
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if hasattr(value, "__float__"):
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def extract_image_metadata(raw: bytes) -> dict:
    with Image.open(io.BytesIO(raw)) as img:
        result = {
            "format": img.format,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
        }
        exif = img.getexif()
        if exif:
            tags = {TAGS.get(tag_id, str(tag_id)): _sanitize(value) for tag_id, value in exif.items()}
            gps_ifd = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else {}
            if gps_ifd:
                tags["GPSInfo"] = {GPSTAGS.get(k, str(k)): _sanitize(v) for k, v in gps_ifd.items()}
            result["exif"] = tags
        return result


def extract_pdf_metadata(raw: bytes) -> dict:
    reader = PdfReader(io.BytesIO(raw))
    info = reader.metadata or {}
    return {
        "page_count": len(reader.pages),
        "document_info": {str(k).lstrip("/"): _sanitize(v) for k, v in info.items()},
    }


def extract_metadata(raw: bytes, mime_type: str) -> dict:
    if mime_type.startswith("image/"):
        return extract_image_metadata(raw)
    if mime_type == "application/pdf":
        return extract_pdf_metadata(raw)
    return {"note": f"no metadata extractor for mime type {mime_type}"}


def handle_metadata(asset_id, job_id, processor, file_bytes, name, mime_type, resolved_config):
    payload = extract_metadata(file_bytes, mime_type)
    store_interpretation(asset_id, job_id, processor["id"], processor["name"], payload)


INTERNAL_HANDLERS = {
    "metadata": handle_metadata,
}


# ============================================================
# http capability — generic, template-driven request/response handling
# ============================================================

def handle_http_json(asset_id, job_id, processor, file_bytes, name, mime_type, resolved_config):
    """Runs ANY http-type processor purely from its `request_template` +
    `output_mapping` — no per-service Python needed."""
    endpoint_url = processor.get("endpoint_url")
    if not endpoint_url:
        raise RuntimeError(f"processor '{processor['name']}' is type=http but has no endpoint_url")

    template = processor.get("request_template") or {}

    files = {}
    data = {}
    for field, spec in (template.get("multipart") or {}).items():
        source = spec.get("source")
        if source == "asset_bytes":
            content_type = spec.get("content_type") or mime_type
            files[field] = (name, file_bytes, content_type)
        elif source == "config_json":
            data[field] = json.dumps(resolved_config)
        elif source == "config_field":
            key = spec.get("key")
            if key is None or key not in resolved_config:
                print(
                    f"[scheduler] processor '{processor['name']}': config_field "
                    f"'{key}' not present in resolved config — skipping field '{field}'"
                )
                continue
            data[field] = resolved_config[key]
        else:
            print(
                f"[scheduler] processor '{processor['name']}': unknown multipart "
                f"source '{source}' for field '{field}' — skipping field"
            )

    resp = requests.request(
        template.get("method", "POST"),
        endpoint_url,
        headers=template.get("headers") or {},
        files=files or None,
        data=data or None,
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()

    output_mapping = processor.get("output_mapping") or {}
    confidence = _dig(payload, output_mapping.get("confidence_path"))
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None

    # asset_interpretations.search_vector only indexes a key literally
    # named 'raw_text' (see schema.sql). text_path tells us which field
    # in THIS processor's response is the searchable text ("markdown",
    # "text", "caption", ...) — copy it under that fixed name so full-text
    # search actually has something to index, without touching the
    # original response shape otherwise. Doesn't overwrite a payload that
    # already happens to have its own 'raw_text' key.
    text_value = _dig(payload, output_mapping.get("text_path"))
    if text_value is not None and isinstance(payload, dict) and "raw_text" not in payload:
        payload = {**payload, "raw_text": text_value}

    store_interpretation(
        asset_id, job_id, processor["id"], processor["name"],
        raw_payload=payload, confidence=confidence,
    )


# ============================================================
# dispatch — `type` decides HOW to run, `capability` is just a label
# ============================================================

def dispatch_processor(asset_id, job_id, processor, file_bytes, name, mime_type, resolved_config):
    ptype = processor.get("type", "internal")
    if ptype == "internal":
        handler = INTERNAL_HANDLERS.get(processor["capability"])
        if not handler:
            raise RuntimeError(f"no internal handler for capability '{processor['capability']}'")
        handler(asset_id, job_id, processor, file_bytes, name, mime_type, resolved_config)
    elif ptype == "http":
        handle_http_json(asset_id, job_id, processor, file_bytes, name, mime_type, resolved_config)
    else:
        raise RuntimeError(f"unknown processor type '{ptype}'")


# ============================================================
# per-asset dispatch
# ============================================================

def run_stage(asset_id, name, mime_type, stage_id, processor, condition, capability_index,
              file_bytes_holder, contract_id=None, contract_version=None, config=None, label=None):
    """Shared by both dispatch paths: idempotency + condition check +
    lazy download + job creation + dispatch + status marking.
    file_bytes_holder is a 1-element list used as a mutable box so a
    lazily-downloaded file is shared across stages for the same asset."""
    if already_processed(asset_id, processor["id"], stage_id):
        return

    if not condition_met(asset_id, condition, capability_index):
        return  # dependency not ready yet — retry next poll

    if file_bytes_holder[0] is None:
        file_bytes_holder[0] = supabase.storage.from_(SUPABASE_BUCKET).download(name)

    job = create_job(
        asset_id, processor["id"], stage_id=stage_id,
        contract_id=contract_id, contract_version=contract_version,
        configuration=config,
    )
    resolved_config = resolve_config(processor, job.get("configuration") or {})
    tag = label or f"{processor['name']} ({processor['capability']})"
    try:
        dispatch_processor(asset_id, job["id"], processor, file_bytes_holder[0], name, mime_type, resolved_config)
        mark_job(job["id"], "complete")
        print(f"[scheduler] {name}: {tag} complete")
    except Exception as exc:  # noqa: BLE001
        mark_job(job["id"], "failed", str(exc))
        print(f"[scheduler] {name}: {tag} FAILED — {exc}")
        traceback.print_exc()


def dispatch_via_contract(asset_id, name, mime_type, contract, processor_lookup, capability_index):
    stages = resolve_pipeline_stages(contract, processor_lookup)
    file_bytes_holder = [None]
    for stage, processor in stages:
        stage_id = stage.get("id") or DEFAULT_STAGE_ID
        # stage's own condition, if the key is present (even as null),
        # overrides the processor row's stored dispatch_condition for
        # this contract; if absent, the row's own condition still applies
        condition = stage.get("condition", processor.get("dispatch_condition"))
        label = f"{contract['contract_id']}@{contract['version']}/{stage_id} -> {processor['name']}"
        run_stage(
            asset_id, name, mime_type, stage_id, processor, condition, capability_index,
            file_bytes_holder,
            contract_id=contract["contract_id"], contract_version=contract["version"],
            config=stage.get("config"), label=label,
        )


def dispatch_legacy(asset_id, name, mime_type, default_processors, capability_index):
    matches = applicable_processors(default_processors, mime_type)
    if not matches:
        return
    # unconditioned first, so a same-cycle dependency usually resolves
    # without waiting for the next poll
    matches = sorted(matches, key=lambda p: p.get("dispatch_condition") is not None)
    file_bytes_holder = [None]
    for processor in matches:
        run_stage(
            asset_id, name, mime_type, DEFAULT_STAGE_ID, processor,
            processor.get("dispatch_condition"), capability_index, file_bytes_holder,
        )


def process_file(file_info: dict, contracts: list, processor_lookup: dict,
                  capability_index: dict, default_processors: list):
    name = file_info["name"]
    asset_id, mime_type = get_or_create_asset(file_info)
    if not mime_type:
        return

    contract = match_contract(contracts, mime_type)
    if contract is not None:
        dispatch_via_contract(asset_id, name, mime_type, contract, processor_lookup, capability_index)
    else:
        dispatch_legacy(asset_id, name, mime_type, default_processors, capability_index)


def main():
    print(f"[scheduler] starting — bucket={SUPABASE_BUCKET} poll={POLL_INTERVAL_SECONDS}s")
    while True:
        try:
            processors = get_all_enabled_processors()
            if not processors:
                print("[scheduler] no enabled processors found — nothing to dispatch")
            capability_index = build_capability_index(processors)
            processor_lookup = build_processor_lookup(processors)
            default_processors = [p for p in processors if p.get("run_by_default")]

            contracts = get_contracts()
            if not contracts:
                print("[scheduler] no enabled ingest contracts — falling back to run_by_default dispatch only")

            for file_info in list_bucket_files():
                process_file(file_info, contracts, processor_lookup, capability_index, default_processors)
        except Exception as exc:  # noqa: BLE001
            print(f"[scheduler] poll loop error: {exc}")
            traceback.print_exc()
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
