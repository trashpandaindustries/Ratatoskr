-- Ratatoskr working schema — needs cleaning in future.


DROP TABLE IF EXISTS asset_embeddings CASCADE;
DROP TABLE IF EXISTS asset_interpretations CASCADE;
DROP TABLE IF EXISTS processing_jobs CASCADE;
DROP TABLE IF EXISTS processors CASCADE;
DROP TABLE IF EXISTS assets CASCADE;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- assets — canonical, immutable-original record
-- ============================================================
CREATE TABLE assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    storage_path TEXT NOT NULL UNIQUE,
    mime_type TEXT,
    checksum TEXT,
    byte_size BIGINT,
    storage_version INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}'::jsonb,

    original_filename TEXT,
    display_name TEXT,
    logical_path TEXT DEFAULT '/',
    tags TEXT[] DEFAULT '{}',
    is_favorite BOOLEAN DEFAULT false,
    trashed_at TIMESTAMPTZ,
    owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,

    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now())
);

-- ============================================================
-- processors — registered processor capabilities, versioned
-- ============================================================
CREATE TABLE processors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0.0',
    endpoint_url TEXT,              -- NULL for internal, HTTP URL for external processors
    type TEXT NOT NULL DEFAULT 'internal',  -- 'internal' or 'http'
    capabilities JSONB DEFAULT '[]'::jsonb, -- e.g. ["image/jpeg", "ocr"]
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()),
    UNIQUE (name, version)
);

-- ============================================================
-- processing_jobs — async work queue
-- ============================================================
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    processor_id UUID NOT NULL REFERENCES processors(id),
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    configuration JSONB DEFAULT '{}'::jsonb,
    result JSONB,
    error_message TEXT,
    processing_identity TEXT,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()),
    UNIQUE (asset_id, processor_id)
);

-- ============================================================
-- asset_interpretations — versioned/disposable machine interpretations
-- ============================================================
CREATE TABLE asset_interpretations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    processor_id UUID NOT NULL REFERENCES processors(id),
    model_name TEXT,
    model_version TEXT,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb, -- expects raw_text key for full-text search
    confidence FLOAT8,
    is_human_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()),
    search_vector tsvector GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(raw_payload->>'raw_text', ''))) STORED
);

-- ============================================================
-- asset_embeddings — pgvector storage for CLIP / text embeddings
-- ============================================================
CREATE TABLE asset_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    interpretation_id UUID REFERENCES asset_interpretations(id) ON DELETE CASCADE,
    modality TEXT NOT NULL,  -- 'image_clip', 'text_chunk', 'audio_whisper'
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now())
);

-- ============================================================
-- Indexes
-- ============================================================
CREATE INDEX idx_assets_tags ON assets USING GIN (tags);
CREATE INDEX idx_processing_jobs_status_priority ON processing_jobs (status, priority DESC, created_at ASC);
CREATE INDEX idx_processing_jobs_identity ON processing_jobs (processing_identity);
CREATE INDEX idx_interpretations_asset_id ON asset_interpretations (asset_id);
CREATE INDEX idx_interpretations_processor_id ON asset_interpretations (processor_id);
CREATE INDEX idx_interpretations_search_vector ON asset_interpretations USING GIN (search_vector);
CREATE INDEX asset_embeddings_hnsw ON asset_embeddings USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- updated_at triggers
-- ============================================================
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_assets_modtime
    BEFORE UPDATE ON assets
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_processing_jobs_modtime
    BEFORE UPDATE ON processing_jobs
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_processors_modtime
    BEFORE UPDATE ON processors
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- ============================================================
-- Seed processors - NO LONGER IN THIS PATTERN
-- ============================================================
-- INSERT INTO processors (name, version, type, capabilities, enabled)
-- VALUES
--     ('metadata',  '0.1.0', 'internal', '["metadata"]'::jsonb, true),
--    ('tesseract', '5.0.0', 'internal', '["image/jpeg", "image/png", "ocr"]'::jsonb, true),
--    ('paddleocr', '2.7.0', 'internal', '["image/jpeg", "image/png", "ocr"]'::jsonb, true),
--    ('gemini',    '1.0.0', 'internal', '["image/jpeg", "image/png", "image/webp", "ocr"]'::jsonb, true)
--ON CONFLICT (name, version) DO UPDATE SET
--    capabilities = EXCLUDED.capabilities,
--    enabled = EXCLUDED.enabled;
--
-- ============================================================
-- Row-Level Security — owner-scoped on assets, cascading through the
-- FK on jobs/interpretations/embeddings; processors are shared config,
-- readable by any authenticated user, writable only via service_role.
-- ============================================================
ALTER TABLE assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE asset_interpretations ENABLE ROW LEVEL SECURITY;
ALTER TABLE asset_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE processors ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage their own assets"
ON assets FOR ALL
USING (auth.uid() = owner_id)
WITH CHECK (auth.uid() = owner_id);

CREATE POLICY "Users can view jobs for their assets"
ON processing_jobs FOR SELECT
USING (
    EXISTS (SELECT 1 FROM assets WHERE assets.id = processing_jobs.asset_id AND assets.owner_id = auth.uid())
);
CREATE POLICY "Users can create jobs for their assets"
ON processing_jobs FOR INSERT
WITH CHECK (
    EXISTS (SELECT 1 FROM assets WHERE assets.id = processing_jobs.asset_id AND assets.owner_id = auth.uid())
);
CREATE POLICY "Users can update jobs for their assets"
ON processing_jobs FOR UPDATE
USING (
    EXISTS (SELECT 1 FROM assets WHERE assets.id = processing_jobs.asset_id AND assets.owner_id = auth.uid())
);

CREATE POLICY "Users can view interpretations of their assets"
ON asset_interpretations FOR SELECT
USING (
    EXISTS (SELECT 1 FROM assets WHERE assets.id = asset_interpretations.asset_id AND assets.owner_id = auth.uid())
);
CREATE POLICY "Users can create interpretations for their assets"
ON asset_interpretations FOR INSERT
WITH CHECK (
    EXISTS (SELECT 1 FROM assets WHERE assets.id = asset_interpretations.asset_id AND assets.owner_id = auth.uid())
);

CREATE POLICY "Users can view embeddings of their assets"
ON asset_embeddings FOR SELECT
USING (
    EXISTS (SELECT 1 FROM assets WHERE assets.id = asset_embeddings.asset_id AND assets.owner_id = auth.uid())
);
CREATE POLICY "Users can create embeddings for their assets"
ON asset_embeddings FOR INSERT
WITH CHECK (
    EXISTS (SELECT 1 FROM assets WHERE assets.id = asset_embeddings.asset_id AND assets.owner_id = auth.uid())
);

CREATE POLICY "Authenticated users can view processors"
ON processors FOR SELECT
USING (auth.role() = 'authenticated');

-- Adds capability-based dispatch to processors, and lets exactly one
-- processor per capability be flagged as the one that auto-runs on ingest.
-- Run against the schema produced by schema.sql.

-- "capabilities" was a mixed bag of mime types and job-type tags
-- (["image/jpeg", "ocr"]) — split those into two clear columns.
ALTER TABLE processors RENAME COLUMN capabilities TO input_mime_types;
ALTER TABLE processors ADD COLUMN IF NOT EXISTS capability TEXT;
ALTER TABLE processors ADD COLUMN IF NOT EXISTS run_by_default BOOLEAN NOT NULL DEFAULT false;

UPDATE processors SET capability = 'metadata' WHERE name = 'metadata';
UPDATE processors SET capability = 'ocr' WHERE name IN ('tesseract', 'paddleocr', 'gemini');

ALTER TABLE processors ALTER COLUMN capability SET NOT NULL;

-- metadata now reads PDFs too (pypdf), not just image EXIF (Pillow)
UPDATE processors SET input_mime_types = '["image/*", "application/pdf"]'::jsonb
WHERE name = 'metadata';

-- the DB enforces "at most one default per capability" so flipping engines
-- is always a deliberate, race-free UPDATE, never two defaults active at once
CREATE UNIQUE INDEX IF NOT EXISTS processors_one_default_per_capability
ON processors (capability)
WHERE run_by_default AND enabled;

-- sane starting defaults: metadata always runs, tesseract is the OCR
-- engine used automatically until you decide otherwise
UPDATE processors SET run_by_default = true WHERE name = 'metadata';
UPDATE processors SET run_by_default = true WHERE name = 'tesseract';

-- to change the default OCR engine later:
--   UPDATE processors SET run_by_default = false WHERE name = 'tesseract';
--   UPDATE processors SET run_by_default = true  WHERE name = 'paddleocr';
-- (must be two statements — setting a second one true while the first is
-- still true+enabled would trip the unique index, which is the point)

-- Adds per-processor configuration contracts, replacing the last global
-- env var (OCR_LANG) with data that lives alongside each processor.
--
-- config_schema  — what's configurable for THIS processor row: field
--                  names, types, descriptions. Not a universal invocation
--                  protocol, just documentation + validation surface.
-- default_config — the actual default values, conforming to that schema.
--
-- processing_jobs.configuration (already present, unused until now) is
-- the per-job override. At dispatch time: default_config merged with
-- job.configuration, validated field-by-field against config_schema.

ALTER TABLE processors ADD COLUMN IF NOT EXISTS config_schema JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE processors ADD COLUMN IF NOT EXISTS default_config JSONB NOT NULL DEFAULT '{}'::jsonb;

-- tesseract and paddleocr both take a `lang` field via OCR-Service's
-- multipart API, but use different language-code vocabularies (tesseract:
-- codes like "eng", "deu"; paddleocr: its own short codes like "en",
-- "ch") — that mismatch is exactly why one shared OCR_LANG never worked.
UPDATE processors SET
    config_schema = '{"lang": {"type": "string", "description": "Tesseract language code, e.g. eng, deu, fra"}}'::jsonb,
    default_config = '{"lang": "eng"}'::jsonb
WHERE name = 'tesseract';

UPDATE processors SET
    config_schema = '{"lang": {"type": "string", "description": "PaddleOCR language code, e.g. en, ch, fr \u2014 different vocabulary than tesseract''s"}}'::jsonb,
    default_config = '{"lang": "en"}'::jsonb
WHERE name = 'paddleocr';

-- gemini is a placeholder for a future non-OCR-Service processor (its own
-- endpoint, its own API shape) — no handler dispatches to it yet, so its
-- contract is left empty rather than guessed at.
UPDATE processors SET
    config_schema = '{}'::jsonb,
    default_config = '{}'::jsonb
WHERE name = 'gemini';

-- metadata takes no options today; the columns just give it somewhere to
-- grow (e.g. a future extract_gps flag) without another env var.
UPDATE processors SET
    config_schema = '{}'::jsonb,
    default_config = '{}'::jsonb
WHERE name = 'metadata';

ALTER TABLE processors
  ADD COLUMN request_template JSONB,   -- http processors only: method/headers/multipart shape
  ADD COLUMN output_mapping JSONB,     -- http processors only: {"text_path": "...", "confidence_path": "..."}
  ADD COLUMN dispatch_condition JSONB; -- NULL = always applicable (subject to mime match)

-- always runs on images, feeds conditions
INSERT INTO processors (name, capability, type, input_mime_types, run_by_default, enabled)
VALUES ('exif-metadata', 'metadata', 'internal', '["image/*"]'::jsonb, true, true);

-- always captions, no condition
INSERT INTO processors (name, capability, type, endpoint_url, input_mime_types, run_by_default, enabled, request_template, output_mapping)
VALUES ('bentobox-caption', 'caption', 'http', 'http://192.168.1.69:3000/caption',
  '["image/*"]'::jsonb, true, true,
  '{"method":"POST","headers":{"accept":"application/json"},"multipart":{"image":{"source":"asset_bytes"}}}'::jsonb,
  '{"text_path":"caption"}'::jsonb);

-- only for DSLR-shot images
INSERT INTO processors (name, capability, type, endpoint_url, input_mime_types, run_by_default, enabled, request_template, output_mapping, dispatch_condition)
VALUES ('GuntherCox - Tesseract', 'ocr', 'http', 'http://192.168.1.69:5000',
  '["image/*"]'::jsonb, true, true,
  '{"method":"POST","headers":{"accept":"application/json"},"multipart":{"image":{"source":"asset_bytes"},"config":{"source":"config_json"}}}'::jsonb,
  '{"text_path":"text","confidence_path":"confidence"}'::jsonb,
  '{"source_capability":"metadata","path":"exif.Make","op":"in","value":["Canon","NIKON CORPORATION","SONY"]}'::jsonb);  

  INSERT INTO processors (name, capability, type, endpoint_url, input_mime_types,
                         run_by_default, enabled, request_template, output_mapping,
                         config_schema, default_config, dispatch_condition)
VALUES (
  'GuntherCox - PaddleOCR', 'Complex - OCR', 'http', 'http://192.168.1.69:5000',
  '["image/*"]'::jsonb,
  true, true,
  '{"method":"POST","headers":{"accept":"application/json"},
    "multipart":{
      "image":{"source":"asset_bytes"},
      "engine":{"source":"config_field","key":"engine"},
      "lang":{"source":"config_field","key":"lang"}
    }}'::jsonb,
  '{"text_path":"text","confidence_path":"confidence"}'::jsonb,
  '{"engine":{"type":"string"},"lang":{"type":"string"}}'::jsonb,
  '{"engine":"paddleocr","lang":"eng"}'::jsonb,
  '{"source_capability":"metadata","path":"exif.Make","op":"in",
    "value":["Canon","NIKON CORPORATION","SONY"]}'::jsonb
);
