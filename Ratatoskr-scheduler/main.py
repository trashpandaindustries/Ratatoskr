"""
Ratatoskr — capability-dispatching scheduler.

Polls one Supabase Storage bucket, registers any new object as an `assets`
row, then for each object asks the `processors` table which processors are
both `enabled` and `run_by_default`, filters those down to the ones whose
`input_mime_types` match the asset's mime type, and runs each applicable
one exactly once (idempotency checked against asset_interpretations).

Two capabilities are implemented:
    metadata  — local extraction via Pillow (image EXIF) or pypdf (PDF
                document info + page count). No external service involved.
    ocr       — delegates to a self-hosted OCR-Service
                (https://github.com/gunthercox/ocr-service) over HTTP,
                using the dispatched processor's `name` as the engine
                param ("tesseract" or "paddleocr").

Which OCR engine runs automatically lives entirely in the `processors`
table now (`run_by_default`) — there's no OCR_ENGINE env var anymore.
Switch engines with:
    UPDATE processors SET run_by_default = false WHERE name = 'tesseract';
    UPDATE processors SET run_by_default = true  WHERE name = 'paddleocr';
(the partial unique index on processors enforces exactly one default per
capability, so those have to be two separate statements.)

Per-processor settings (e.g. OCR language) now live in `processors` too,
not env vars: `config_schema` declares what's configurable for that
specific processor row, `default_config` holds the actual defaults, and
`processing_jobs.configuration` is a per-job override. Dispatch resolves
default_config merged with job.configuration, validated field-by-field
against config_schema — unknown keys or wrong types are dropped with a
log line rather than sent through. There's no OCR_LANG env var anymore;
tesseract and paddleocr each carry their own correct default language
code, since they don't share a code vocabulary.

Required env vars:
    SUPABASE_URL            e.g. https://your-project.supabase.co
    SUPABASE_KEY            service_role key (Storage read + table access)
    SUPABASE_BUCKET         exact bucket name to watch
    OCR_SERVICE_URL         e.g. http://ocr-service:5000
    POLL_INTERVAL_SECONDS   default: 10
"""
import io
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

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr-service:5000")

POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "10"))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# processor discovery / dispatch matching
# ============================================================

def get_dispatchable_processors():
    """All enabled + run_by_default processors. Refetched every poll so a
    default-engine flip in the DB takes effect on the next cycle without a
    restart."""
    result = (
        supabase.table("processors")
        .select("id, name, capability, input_mime_types, config_schema, default_config")
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
# job bookkeeping (parametrized by processor_id now, not one global engine)
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
# metadata capability — local extraction, no external service
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


# ============================================================
# ocr capability — delegates to OCR-Service
# ============================================================

def run_ocr(file_bytes: bytes, filename: str, engine: str, lang: str) -> dict:
    resp = requests.post(
        OCR_SERVICE_URL.rstrip("/") + "/",
        files={"image": (filename, file_bytes)},
        data={"engine": engine, "lang": lang},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def average_confidence(regions: list) -> Optional[float]:
    scores = [r["confidence"] for r in regions if "confidence" in r]
    return sum(scores) / len(scores) if scores else None


def handle_ocr(asset_id, job_id, processor, file_bytes, name, mime_type, resolved_config):
    lang = resolved_config.get("lang", "eng")
    ocr_result = run_ocr(file_bytes, name, engine=processor["name"], lang=lang)
    regions = ocr_result.get("regions", [])
    payload = {"raw_text": ocr_result.get("text", ""), "regions": regions}
    store_interpretation(
        asset_id, job_id, processor["id"], processor["name"],
        payload, confidence=average_confidence(regions),
    )


CAPABILITY_HANDLERS = {
    "metadata": handle_metadata,
    "ocr": handle_ocr,
}


# ============================================================
# per-asset dispatch
# ============================================================

def process_file(file_info: dict, processors: list):
    name = file_info["name"]
    asset_id, mime_type = get_or_create_asset(file_info)
    if not mime_type:
        return

    matches = applicable_processors(processors, mime_type)
    if not matches:
        return

    file_bytes = None  # only fetched from Storage if something actually needs it
    for processor in matches:
        if already_processed(asset_id, processor["id"]):
            continue

        handler = CAPABILITY_HANDLERS.get(processor["capability"])
        if not handler:
            print(f"[scheduler] no handler for capability '{processor['capability']}' "
                  f"(processor '{processor['name']}') — skipping")
            continue

        if file_bytes is None:
            file_bytes = supabase.storage.from_(SUPABASE_BUCKET).download(name)

        job = create_job(asset_id, processor["id"])
        resolved_config = resolve_config(processor, job.get("configuration") or {})
        try:
            handler(asset_id, job["id"], processor, file_bytes, name, mime_type, resolved_config)
            mark_job(job["id"], "complete")
            print(f"[scheduler] {name}: {processor['name']} ({processor['capability']}) complete")
        except Exception as exc:  # noqa: BLE001
            mark_job(job["id"], "failed", str(exc))
            print(f"[scheduler] {name}: {processor['name']} FAILED — {exc}")
            traceback.print_exc()


def main():
    print(
        f"[scheduler] starting — bucket={SUPABASE_BUCKET} "
        f"ocr_url={OCR_SERVICE_URL} poll={POLL_INTERVAL_SECONDS}s"
    )
    while True:
        try:
            processors = get_dispatchable_processors()
            if not processors:
                print("[scheduler] no run_by_default+enabled processors found — nothing to dispatch")
            for file_info in list_bucket_files():
                process_file(file_info, processors)
        except Exception as exc:  # noqa: BLE001
            print(f"[scheduler] poll loop error: {exc}")
            traceback.print_exc()
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()