# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**webQCVT** is a web service that converts and **tailors** CVs from PDF, DOCX, or image formats into standardized Quantori Word document templates using Google Gemini AI. It is the "Tailor" variant of webQCV — same core engine but with JD-based tailoring, relevance checking, and keyword refinement.

**Related projects (same machine, separate repos):**
- `Q-CV` (desktop) — Flet desktop app, shares `cv_engine.py` prompts
- `webQCV` — simpler web converter without tailoring (DO NOT modify without explicit request)

## Running the Application

```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000

# With auto-reload for development
uvicorn app:app --reload --port 8000
```

The app serves the frontend at `/` and the admin dashboard at `/admin/usage`.

## Environment / Configuration

- **API key**: Gemini API key loaded from `~/.quantoricv_settings.json` under key `"api_key"`
- **Master prompts**: Override prompts via `~/.master_prompts.json`
- **Templates**: `.docx` template files in `templates/`, discovered automatically
- **Cache**: Base JSON cached in `_cache/base_json/{sha256}.base.json` — safe to delete to force re-extraction
- **Usage log**: Appended to `usage_log.jsonl` in the project root

## Architecture

```
index.html              Vanilla JS single-page frontend; polls /jobs/{id} every 1.5s
app.py                  FastAPI endpoints; spawns background threads for jobs
converter_engine.py     Job orchestration: parse → check → tailor → anonymize → render
cv_engine.py            Core logic: LLM schema, prompts, sanitization, anonymization, DOCX generation
source_baseline_extractor.py  Raw text extraction from PDF/DOCX inputs
templates/              Quantori .docx template files (docxtpl-rendered)
_cache/                 File-based cache keyed by SHA256 of source content
```

### Processing Pipeline

1. `POST /jobs` — saves uploaded file, enqueues job, returns `job_id`
2. Background thread (`_run_job`) runs the pipeline:
   - Hash source → check `_cache/` for previously extracted base JSON
   - If cache miss: Gemini extract CV into `CV_JSON_SCHEMA`
   - Optional: autofix pass
   - If tailor enabled: `_check_relevance()` (keyword overlap) → `_apply_tailor()` (LLM rewrite)
   - Optional: anonymization (`smart_anonymize_data`)
   - Build content details + JD keyword report
   - Render DOCX via `docxtpl`
3. Frontend polls `GET /jobs/{job_id}` for `{status, progress, ready, details}`
4. `GET /jobs/{job_id}/download` returns the generated DOCX

### Tailoring Features

- **Relevance check**: Deterministic keyword overlap (`_check_relevance`). Dual ratio `max(jd_ratio, cv_ratio)`. LOW (<5%) blocks tailoring unless force_tailor=true.
- **JD validation**: `validate_jd()` rejects empty/short/non-JD text (20 char, 5 word thresholds, `_JD_MARKERS` set).
- **Keyword report**: `_compute_jd_keyword_report()` compares JD vs tailored CV, returns matched/missing/added lists with match percentage. Shown in UI modal.
- **Refine (2nd pass)**: `POST /jobs/{id}/refine` — surgical LLM pass that weaves missing JD keywords into already-tailored CV. Limited to one refine per job. Uses `prompt_refine`.
- **Title cleanup**: Tech terms rescued from LinkedIn parenthesized titles into `skills["Title Specialties"]`.

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves index.html |
| POST | `/jobs` | Create conversion job (file + options) |
| GET | `/jobs/{id}` | Poll job status, progress, details |
| GET | `/jobs/{id}/download` | Download generated DOCX |
| POST | `/jobs/{id}/refine` | Trigger keyword refinement pass |
| GET | `/stats` | Server stats (active jobs, today count, uptime) |
| GET | `/templates` | List available templates |
| GET | `/admin/usage` | Usage dashboard |

### Key Data

- **`CV_JSON_SCHEMA`** (in `cv_engine.py`) — canonical schema for LLM extraction
- **`DEFAULT_PROMPTS`** (in `cv_engine.py`) — all LLM prompts including `prompt_tailor`, `prompt_refine`, `prompt_anonymize`
- **Job state** — `InMemoryJobStore` with `JobState` dataclass; after tailor, jobs store `_tailored_json` and `_jd_text` for refine reuse

### LLM Integration

- Model: `gemini-2.0-flash` (`choose_model_name()` in `converter_engine.py`)
- SDK: `from google import genai` — `genai.Client(api_key=...)` per call
- Images/PDFs uploaded via `client.files.upload()` with state polling until `ACTIVE`
- Retry logic for 429/quota errors in `_retry_generate()`

### Frontend (index.html)

- Single-file vanilla JS, no build step
- Stats bar in header (polls `/stats` every 10s)
- File upload with template/anonymize/tailor options
- JD textarea appears when "Tailor to JD" is checked
- After tailor: modal auto-opens showing JD keyword match report (matched/missing/added)
- Refine button in modal header (one-shot, closes modal and resumes polling)
- Low relevance → confirm dialog → force_tailor resubmit

### Caching

Source files SHA256-hashed; base JSON reused across re-submissions with different options. Tailor/anonymize applied on top of cached base JSON.

### Syncing with Desktop (Q-CV)

Prompts in `cv_engine.py` are shared between desktop and web. When improving prompts:
1. Edit in webQCVT first (easier to test)
2. Copy prompt changes to `Q-CV/cv_engine.py`
3. Do NOT sync web-specific code (endpoints, UI, converter_engine) to desktop

### Limitations

- **Single-instance only**: job state in-memory, no multi-server scaling
- **No authentication**: public API with IP logging only
- **No database**: job queue in-memory; usage log is flat JSONL
- **Ephemeral storage**: generated files lost on restart
