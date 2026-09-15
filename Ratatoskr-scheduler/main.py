"""
Ratatoskr — capability-dispatching scheduler.

Polls one Supabase Storage bucket, registers any new object as an `assets`
row, then for each object asks the `processors` table which processors are
both `enabled` and `run_by_default`, filters those down to the ones whose
`input_mime_types` match the asset's mime type, and runs each applicable
one exactly once (idempotency checked against asset_interpretations).

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
                INSERT, not a code change — `capability` is just a label
                for mime-matching, conditions, and logging on these.

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

`output_mapping` shape (JSONB):
    {"text_path": "caption", "confidence_path": "confidence"}
Dotted paths into the JSON response; missing/absent paths just resolve to
None rather than raising. The full response is always stored in
raw_payload regardless of what output_mapping pulls out.

Conditional dispatch (`dispatch_condition`, JSONB, nullable):
    {"source_capability": "metadata", "path": "exif.Make", "op": "in",
     "value": ["Canon", "NIKON CORPORATION", "SONY"]}
Lets one capability's dispatch depend on another's *already-stored*
interpretation for the same asset — e.g. only run paddleocr on images
whose EXIF Make looks like a DSLR, while still captioning everything.
If the dependency hasn't produced an interpretation yet, the processor is
skipped for this poll cycle (not treated as an error) and picked back up
once it exists — matches contains a query, e.g.
    {"path": "width", "op": "gt", "value": 1024}
Unconditioned processors are dispatched before conditioned ones within a
single poll so same-cycle dependencies (metadata -> paddleocr) usually
resolve without waiting for a second loop.

Which processors run automatically lives entirely in the `processors`
table (`run_by_default`) — no OCR_ENGINE/OCR_SERVICE_URL env vars. Switch
engines with:
    UPDATE processors SET run_by_default = false WHERE name = 'tesseract';
    UPDATE processors SET run_by_default = true  WHERE name = 'paddleocr';
(the partial unique index on processors enforces exactly one default per
capability, so those have to be two separate statements.)

Per-processor settings (e.g. OCR language) live in `processors`, not env
vars: `config_schema` declares what's configurable for that specific
processor row, `default_config` holds the actual defaults, and
`processing_jobs.configuration` is a per-job override. Dispatch resolves
default_config merged with job.configuration, validated field-by-field
against config_schema — unknown keys or wrong types are dropped with a
log line rather than sent through.

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
# processor discovery / dispatch matching
# ============================================================

def get_dispatchable_processors():
    """All enabled + run_by_default processors. Refetched every poll so a
    default-engine flip or a newly-added processor row takes effect on
    the next cycle without a restart."""
    result = (
        supabase.table("processors")
        .select(
            "id, name, capability, type, endpoint_url, input_mime_types, "
            "config_schema, default_config, request_template, "
            "output_mapping, dispatch_condition"
        )
        .eq("enabled", True)
        .eq("run_by_default", True)
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
    """capability -> [processor ids]. Used to resolve dispatch_condition's
    source_capability into the set of processor rows whose interpretation
    would satisfy it — a condition points at a capability, not a specific
    processor, so e.g. swapping tesseract for paddleocr as the default OCR
    engine doesn't break anything downstream that conditions on 'ocr'."""
    idx = {}
    for p in processors:
        idx.setdefault(p["capability"], []).append(p["id"])
    return idx


# ============================================================
# per-processor configuration contracts
# ============================================================

def _type_matches(value, expected_type: Optional[str]) -> bool:
    if expected_type is None:
        return True  # no declared type — nothing to check against
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True  # unrecognized type name in the schema — don't block on it


def resolve_config(processor: dict, job_configuration: dict) -> dict:
    """Merge this processor's default_config with a per-job override,
    validating each overridden field against config_schema. Unknown keys
    or wrong types are dropped (with a log line) rather than passed
    through — a bad override should never silently reach the processor."""
    schema = processor.get("config_schema") or {}
    resolved = dict(processor.get("default_config") or {})

    for key, value in (job_configuration or {}).items():
        field_schema = schema.get(key)
        if field_schema is None:
            print(f"[scheduler] ignoring unknown config key '{key}' for processor '{processor['name']}'")
            continue
        if not _type_matches(value, field_schema.get("type")):
            print(
                f"[scheduler] ignoring config key '{key}' for processor '{processor['name']}': "
                f"expected {field_schema.get('type')}, got {type(value).__name__}"
            )
            continue
        resolved[key] = value

    return resolved


# ============================================================
# conditional dispatch — gate a processor on another's stored output
# ============================================================

def get_source_payload(asset_id: str, processor_ids: list) -> Optional[dict]:
    """Fetch the raw_payload of any existing interpretation on this asset
    from one of the given processor ids (i.e. any processor that provides
    the condition's source_capability). Returns None if none has run yet
    — the caller treats that as "dependency not ready", not an error."""
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
# job bookkeeping (parametrized by processor_id, not one global engine)
# ============================================================

def already_processed(asset_id: str, processor_id: str) -> bool:
    existing = (
        supabase.table("asset_interpretations")
        .select("id")
        .eq("asset_id", asset_id)
        .eq("processor_id", processor_id)
        .execute()
    )
    return bool(existing.data)


def create_job(asset_id: str, processor_id: str) -> dict:
    """Upsert rather than insert: a row for this (asset_id, processor_id)
    may already exist from an earlier attempt. Reuse it instead of
    colliding with the UNIQUE constraint. Returns the full row (not just
    the id) so the caller can pick up any pre-existing `configuration`
    override and merge it against the processor's defaults."""
    row = (
        supabase.table("processing_jobs")
        .upsert(
            {
                "asset_id": asset_id,
                "processor_id": processor_id,
                "status": "processing",
                "error_message": None,
            },
            on_conflict="asset_id,processor_id",
        )
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
    `output_mapping` — no per-service Python needed. OCR, BentoBox
    captioning, future CLIP/text-embedding endpoints, etc. all go through
    this one function as long as they speak multipart-in/JSON-out."""
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

def process_file(file_info: dict, processors: list, capability_index: dict):
    name = file_info["name"]
    asset_id, mime_type = get_or_create_asset(file_info)
    if not mime_type:
        return

    matches = applicable_processors(processors, mime_type)
    if not matches:
        return

    # Unconditioned processors first, so a same-cycle dependency (e.g.
    # metadata -> paddleocr gated on EXIF) usually resolves without
    # waiting for the next poll.
    matches = sorted(matches, key=lambda p: p.get("dispatch_condition") is not None)

    file_bytes = None  # only fetched from Storage if something actually needs it
    for processor in matches:
        if already_processed(asset_id, processor["id"]):
            continue

        if not condition_met(asset_id, processor.get("dispatch_condition"), capability_index):
            continue  # dependency not ready yet — retry next poll

        if file_bytes is None:
            file_bytes = supabase.storage.from_(SUPABASE_BUCKET).download(name)

        job = create_job(asset_id, processor["id"])
        resolved_config = resolve_config(processor, job.get("configuration") or {})
        try:
            dispatch_processor(asset_id, job["id"], processor, file_bytes, name, mime_type, resolved_config)
            mark_job(job["id"], "complete")
            print(f"[scheduler] {name}: {processor['name']} ({processor['capability']}) complete")
        except Exception as exc:  # noqa: BLE001
            mark_job(job["id"], "failed", str(exc))
            print(f"[scheduler] {name}: {processor['name']} FAILED — {exc}")
            traceback.print_exc()


def main():
    print(f"[scheduler] starting — bucket={SUPABASE_BUCKET} poll={POLL_INTERVAL_SECONDS}s")
    while True:
        try:
            processors = get_dispatchable_processors()
            if not processors:
                print("[scheduler] no run_by_default+enabled processors found — nothing to dispatch")
            capability_index = build_capability_index(processors)
            for file_info in list_bucket_files():
                process_file(file_info, processors, capability_index)
        except Exception as exc:  # noqa: BLE001
            print(f"[scheduler] poll loop error: {exc}")
            traceback.print_exc()
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
