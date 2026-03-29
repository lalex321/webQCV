# webQCV

A web service that converts CVs from PDF, DOCX, or image formats into standardized Quantori Word document templates using Google Gemini AI.

## Features

- **Multi-format input**: PDF, DOCX, PNG, JPG, JPEG
- **Template selection**: Quantori Classic or Helvetica layouts
- **Anonymization**: Strips personal identifiers (name, email, phone, links, location)
- **AutoFix**: LLM-powered quality pass to repair extraction issues
- **Caching**: Identical source files are re-processed instantly from cache

## Setup (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running (local)

```bash
.venv/bin/uvicorn app:app --reload
```

Open http://localhost:8000 in your browser.

## Deployment (Render.com / Docker)

The repo includes a `Dockerfile`. Deploy to any Docker-compatible host:

```bash
docker build -t webcv .
docker run -e GEMINI_API_KEY=AIza... -p 8000:8000 webcv
```

For Render.com: connect the GitHub repo, set `GEMINI_API_KEY` as an environment variable in the Render dashboard — no other configuration needed.

## API Key Configuration

The app resolves the Gemini API key in this priority order:

| Priority | Source | Notes |
|----------|--------|-------|
| 1 | `GEMINI_API_KEY` env var | Recommended for production / Docker |
| 2 | `.api_key` file in app directory | Set via the `/setup` page — takes effect immediately |
| 3 | `~/.quantoricv_settings.json` | Shared with the Q-CV desktop app |

The easiest way to configure the key in a deployed instance is the **`/setup` page** in the browser — no server restart needed.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Frontend UI |
| `GET` | `/setup` | API key configuration page |
| `GET` | `/templates` | List available templates |
| `POST` | `/jobs` | Submit a conversion job |
| `GET` | `/jobs/{id}` | Poll job status |
| `GET` | `/jobs/{id}/download` | Download generated DOCX |
| `GET` | `/admin/usage` | Usage analytics dashboard |

## Stack

- **Backend**: Python, FastAPI, Uvicorn
- **AI**: Google Gemini 2.0 Flash (`google-genai`)
- **Document processing**: python-docx, docxtpl, pypdf, Pillow
- **Frontend**: Vanilla HTML/CSS/JavaScript
