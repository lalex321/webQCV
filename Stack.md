# webQCVT Stack

- **Backend**: Python 3, FastAPI, Uvicorn
- **Frontend**: Vanilla JS/HTML/CSS (single `index.html`, no build step)
- **AI**: Google Gemini 2.5 Flash (via `google-genai` SDK)
- **Document generation**: docxtpl (Jinja2-based DOCX templating)
- **Input parsing**: PyMuPDF (PDF), python-docx (DOCX), Pillow (images)
- **Storage**: File-based (JSON in `_store/`, JSONL usage log, no database)
- **Concurrency**: threading + semaphore (single-instance, in-memory job queue)
- **Deployment**: Render (with `DATA_DIR` for persistent disk)
