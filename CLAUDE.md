# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**webQCV** is a web service that converts CVs from PDF, DOCX, or image formats into standardized Quantori Word document templates using Google Gemini 2.0 Flash for structured extraction.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (with auto-reload for development)
uvicorn app:app --reload

# Run on a specific port
uvicorn app:app --reload --port 8000
```

The app serves the frontend at `/` and the admin dashboard at `/admin/usage`.

## Environment / Configuration

- **API key**: Gemini API key must be set — loaded via `cv_engine.load_config()` from `~/.quantoricv_settings.json`
- **Master prompts**: Override prompts via `~/.master_prompts.json`
- **Templates**: `.docx` template files live in `templates/` and are discovered automatically
- **Cache**: Base JSON extraction results cached in `_cache/base_json/{sha256}.base.json` — safe to delete to force re-extraction
- **Usage log**: Appended to `usage_log.jsonl` in the project root

## Architecture

```
index.html              Vanilla JS single-page frontend; polls /jobs/{id} every 1.5s
app.py                  FastAPI endpoints; spawns background threads for jobs
converter_engine.py     Job orchestration: parse → check → autofix → anonymize → render
cv_engine.py            Core logic: LLM schema, prompts, sanitization, anonymization, DOCX generation
source_baseline_extractor.py  Raw text extraction from PDF/DOCX inputs
templates/              Quantori .docx template files (docxtpl-rendered)
_cache/                 File-based cache keyed by SHA256 of source content
```

### Processing Pipeline

1. `POST /jobs` — saves uploaded file, enqueues job, returns `job_id`
2. Background thread (`_run_job` in `converter_engine.py`) runs the pipeline:
   - Hash source file → check `_cache/` for previously extracted base JSON
   - If cache miss: call Gemini 2.0 Flash to extract CV into `CV_JSON_SCHEMA`
   - Optional: autofix pass (LLM QA → repair loop)
   - Optional: anonymization (PII removal via `smart_anonymize_data`)
   - Render DOCX via `docxtpl` using the selected template
3. Frontend polls `GET /jobs/{job_id}` for `{status, progress, error}`
4. `GET /jobs/{job_id}/download` returns the generated DOCX

### Key Data Structure

`CV_JSON_SCHEMA` (defined in `cv_engine.py`) is the canonical JSON structure that the LLM extracts into and that the DOCX templates consume. It includes: `basics`, `summary`, `skills`, `experience`, `education`, `certifications`, `languages`, `other_sections`.

### LLM Integration

- Model: `gemini-2.0-flash` with retry logic for 429 rate limit errors
- Images (PNG/JPG) are uploaded to Gemini Files API for vision-based extraction
- Prompt constants live in `cv_engine.py` as `DEFAULT_PROMPTS` dict; keys: `prompt_master_inst`, `prompt_qa`, `prompt_autofix`, `prompt_matcher`

### Caching

Source files are SHA256-hashed; extracted base JSON is reused across re-submissions of the same file with different options (template, anonymize, autofix).

### Limitations

- **Single-instance only**: job state is in-memory (`InMemoryJobStore`); no multi-server scaling
- **No authentication**: public API with IP logging only
- **No database**: job queue is in-memory; usage log is a flat JSONL file
