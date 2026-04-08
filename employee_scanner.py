#!/usr/bin/env python3
"""
Employee Scanner — bulk-import employee CVs from folder structure into webQCVT store.

Usage:
    python employee_scanner.py /path/to/Employees [--limit N] [--dry-run]

Folder structure expected:
    Employees/
        Firstname.Lastname/
            01.CV/ (or 1.CV/, 01. CV/, etc.)
                @Name_CV.docx          ← primary (Quantori template)
                @Name_CV_Python.docx   ← variant
                Name_CV_Source.pdf     ← original source
                Archive/               ← old versions
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path so we can import project modules
sys.path.insert(0, str(Path(__file__).parent))

from google import genai
from google.genai import types as genai_types
from cv_engine import CV_JSON_SCHEMA, extract_text_from_docx
from converter_engine import choose_model_name, extract_first_json_object, resolve_api_key
from source_baseline_extractor import extract_from_docx as _extract_from_docx
from cv_engine import _format_docx_sections_for_llm

# Load API key the same way the app does
_settings_path = Path.home() / ".quantoricv_settings.json"
_config = json.loads(_settings_path.read_text()) if _settings_path.exists() else {}
_gemini_api_key = resolve_api_key(Path(__file__).parent, _config)

# ── Filters ──────────────────────────────────────────────────────────────────

# Skip these filenames (templates, not real CVs)
TEMPLATE_PATTERNS = [
    "Kanda_CV_Template",
    "RBNC_CV_Template",
    "Bratislava_CV_Template",
    "Native_CV_Template",
    "Self_Intro",
    "Self_Presentation",
    "DO_NOT_USE",
]

CV_EXTENSIONS = {".docx", ".doc", ".pdf"}


def is_cv_file(path: Path) -> bool:
    """Check if file is a real CV (not a template or self-intro)."""
    if path.suffix.lower() not in CV_EXTENSIONS:
        return False
    name = path.name
    for pattern in TEMPLATE_PATTERNS:
        if pattern.lower() in name.lower():
            return False
    # Skip files in Self_Intro directories
    for part in path.parts:
        if "self_intro" in part.lower():
            return False
    return True


def priority_key(path: Path) -> tuple:
    """Sort key: @ files first, then non-archive, then by name."""
    name = path.name
    is_at = 0 if name.startswith("@") else 1
    is_archive = 1 if "archive" in str(path).lower() else 0
    is_copy = 1 if "copy of" in name.lower() else 0
    return (is_at, is_archive, is_copy, name)


# ── Scanner ──────────────────────────────────────────────────────────────────

def scan_employee_folder(folder: Path) -> list[Path]:
    """Find all CV files in an employee folder, sorted by priority."""
    cv_files = []
    for f in folder.rglob("*"):
        if f.is_file() and is_cv_file(f):
            cv_files.append(f)
    cv_files.sort(key=priority_key)
    return cv_files


def scan_all_employees(root: Path) -> dict[str, list[Path]]:
    """Scan all employee folders. Returns {folder_name: [cv_files]}."""
    employees = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and "." in d.name:  # Firstname.Lastname pattern
            files = scan_employee_folder(d)
            if files:
                employees[d.name] = files
    return employees


# ── Gemini extraction ────────────────────────────────────────────────────────

def _upload_and_wait(client: genai.Client, file_path: Path) -> object:
    """Upload file to Gemini and wait until ACTIVE."""
    suffix = file_path.suffix.lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
    }
    mime_type = mime_map.get(suffix, "application/octet-stream")
    uploaded = client.files.upload(
        file=str(file_path),
        config=genai_types.UploadFileConfig(mime_type=mime_type),
    )
    while getattr(uploaded, "state", None) and getattr(uploaded.state, "name", "") == "PROCESSING":
        time.sleep(1)
        uploaded = client.files.get(name=uploaded.name)
    state_name = getattr(getattr(uploaded, "state", None), "name", "")
    if state_name and state_name != "ACTIVE":
        raise RuntimeError(f"Upload failed: {state_name} for {file_path.name}")
    return uploaded


def _retry(fn, max_retries=5):
    """Retry on rate limit."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            if ("429" in err or "Resource" in err or "Quota" in err) and attempt < max_retries - 1:
                delay = [5, 5, 5, 10, 10][attempt]
                print(f"  ⚠️ Rate limit, sleeping {delay}s (attempt {attempt+1})")
                time.sleep(delay)
            else:
                raise


MERGE_PROMPT_TEMPLATE = """You are a STRICT Lossless CV Extractor and Merger. You will receive multiple CV documents for the SAME person: {person_name}.

These are different versions of their CV prepared for different projects/clients. Most content overlaps.

Your task:
1. Extract and MERGE all CV versions into ONE comprehensive JSON following the schema below.
2. COMBINE all unique experience entries, skills, certifications, education — no duplicates.
3. **EXPERIENCE DEDUPLICATION (CRITICAL):** The same role at the same company MUST appear ONLY ONCE in the output, even if it appears in multiple CV versions. When the same role appears in multiple versions with different highlights/descriptions, MERGE all highlights into ONE entry. Match roles by company name + overlapping or similar date ranges. Do NOT create separate entries for the same position just because different CV versions describe it differently.
4. The person's name is: {person_name} (use this, as CVs may be anonymized).
5. Translate everything to professional US English.
6. For skills: combine all skill categories and items from all versions, remove duplicates.

**CRITICAL ANTI-HALLUCINATION RULES (STRICTLY ENFORCED):**
- **NO INVENTED FACTS / NO DATA LOSS:** Extract ONLY facts explicitly present in the source CVs. NEVER generate, invent, or infer information that is not written in the documents.
- **HIGHLIGHTS INTEGRITY:** Each item in a role's `highlights` array MUST correspond to an actual bullet point, responsibility, or achievement explicitly written in the source CVs for that role. NEVER fabricate highlights.
- **SUMMARY INTEGRITY:** Every bullet in `summary.bullet_points` MUST be taken verbatim (or near-verbatim translated) from the source CVs. Do NOT add new summary points that don't exist in any of the source documents.
- **SKILLS INTEGRITY:** Only list skills, tools, and technologies that are explicitly mentioned in the source CVs. Do NOT infer or add skills based on context.
- **FIX TYPOS & GRAMMAR:** Silently fix obvious spelling mistakes and typos. Do NOT change factual content.
- Use only empty strings "" or arrays [] for missing values. Never output None or null.

OUTPUT SCHEMA:
{schema}

Return ONLY the JSON object, no markdown wrappers."""


def _extract_docx_text(path: Path) -> str:
    """Extract text from DOCX using the same method as the main app."""
    try:
        baseline = _extract_from_docx(str(path))
        return _format_docx_sections_for_llm(baseline)
    except Exception:
        return extract_text_from_docx(str(path))


def _convert_to_pdf(path: Path) -> Path | None:
    """Convert DOCX/DOC to PDF via LibreOffice. Returns PDF path or None."""
    tmp_dir = tempfile.mkdtemp(prefix="cv_scan_")
    try:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp_dir, str(path)],
            capture_output=True, timeout=30,
        )
        pdfs = list(Path(tmp_dir).glob("*.pdf"))
        return pdfs[0] if pdfs else None
    except Exception:
        return None


def extract_and_merge(person_name: str, cv_files: list[Path], max_files: int = 5) -> dict:
    """Send up to max_files to Gemini and get merged CV JSON."""
    client = genai.Client(api_key=_gemini_api_key)
    model = choose_model_name(_config)

    # Limit files to avoid context overflow
    files_to_send = cv_files[:max_files]

    print(f"  Processing {len(files_to_send)} file(s)...")
    contents = []
    for f in files_to_send:
        try:
            suffix = f.suffix.lower()
            if suffix in (".docx", ".doc"):
                # DOCX → extract text (Gemini doesn't accept DOCX uploads)
                text = _extract_docx_text(f)
                if text:
                    contents.append(f"--- CV FILE: {f.name} ---\n{text}")
                    print(f"    ✓ {f.name} (text)")
                else:
                    # Fallback: convert to PDF via LibreOffice
                    print(f"    ⚠ {f.name} (empty text, converting to PDF...)")
                    pdf_path = _convert_to_pdf(f)
                    if pdf_path:
                        uploaded = _upload_and_wait(client, pdf_path)
                        contents.append(uploaded)
                        print(f"    ✓ {f.name} (PDF fallback)")
                    else:
                        print(f"    ✗ {f.name} (conversion failed)")
            else:
                # PDF/images → upload to Gemini
                uploaded = _upload_and_wait(client, f)
                contents.append(uploaded)
                print(f"    ✓ {f.name} (uploaded)")
        except Exception as e:
            print(f"    ✗ {f.name}: {e}")

    if not contents:
        raise RuntimeError("No files processed successfully")

    prompt = MERGE_PROMPT_TEMPLATE.format(
        person_name=person_name.replace(".", " "),
        schema=CV_JSON_SCHEMA,
    )
    contents.append(prompt)

    print(f"  Calling Gemini ({model})...")
    response = _retry(
        lambda: client.models.generate_content(model=model, contents=contents)
    )
    txt = getattr(response, "text", None)
    if not txt:
        raise RuntimeError("Empty response from Gemini")

    cv_json = extract_first_json_object(txt)

    # Force correct name from folder
    if "basics" in cv_json:
        cv_json["basics"]["name"] = person_name.replace(".", " ")

    # Post-process: merge duplicate experience entries
    cv_json["experience"] = _dedup_experience(cv_json.get("experience", []))

    return cv_json


def _dedup_experience(experience: list) -> list:
    """Merge experience entries with same company + overlapping role."""
    if len(experience) < 2:
        return experience

    merged = []
    used = set()

    for i, entry in enumerate(experience):
        if i in used:
            continue
        company = (entry.get("company_name") or "").strip().lower()
        role = (entry.get("role") or "").strip().lower()

        # Find duplicates: same company + similar role, or same role when company is empty
        dupes = [i]
        for j in range(i + 1, len(experience)):
            if j in used:
                continue
            other_company = (experience[j].get("company_name") or "").strip().lower()
            other_role = (experience[j].get("role") or "").strip().lower()

            if company and company == other_company:
                # Same company — check if roles are similar enough
                if role == other_role or _roles_overlap(role, other_role):
                    dupes.append(j)
            elif not company and not other_company and role and role == other_role:
                # Both have empty company — match by exact role
                dupes.append(j)

        if len(dupes) == 1:
            merged.append(entry)
        else:
            # Merge all duplicates into one
            merged.append(_merge_entries([experience[idx] for idx in dupes]))
            used.update(dupes[1:])

    return merged


def _roles_overlap(role1: str, role2: str) -> bool:
    """Check if two role titles are similar enough to be the same position."""
    # Extract significant words (ignore common prefixes like senior/lead/jr)
    noise = {"senior", "lead", "junior", "jr", "sr", "principal", "staff", "intern"}
    def sig_words(r):
        return {w for w in r.split() if w not in noise and len(w) > 2}
    w1, w2 = sig_words(role1), sig_words(role2)
    if not w1 or not w2:
        return False
    overlap = len(w1 & w2) / min(len(w1), len(w2))
    return overlap >= 0.5


def _merge_entries(entries: list) -> dict:
    """Merge multiple experience entries into one."""
    base = dict(entries[0])

    # Collect all unique highlights
    all_highlights = []
    seen_hl = set()
    for e in entries:
        for h in e.get("highlights", []):
            h_key = h.strip().lower()
            if h_key not in seen_hl:
                seen_hl.add(h_key)
                all_highlights.append(h)
    base["highlights"] = all_highlights

    # Merge environment
    all_env = []
    seen_env = set()
    for e in entries:
        for env in e.get("environment", []):
            env_key = env.strip().lower()
            if env_key not in seen_env:
                seen_env.add(env_key)
                all_env.append(env)
    base["environment"] = all_env

    # Use widest date range
    starts = [e.get("dates", {}).get("start", "") for e in entries if e.get("dates", {}).get("start")]
    ends = [e.get("dates", {}).get("end", "") for e in entries if e.get("dates", {}).get("end")]
    if starts:
        base.setdefault("dates", {})["start"] = min(starts)
    if ends:
        # "Present" wins over any date
        if any(e.lower() == "present" for e in ends):
            base.setdefault("dates", {})["end"] = "Present"
        else:
            base.setdefault("dates", {})["end"] = max(ends)

    # Use longest project_description
    descs = [e.get("project_description", "") for e in entries if e.get("project_description")]
    if descs:
        base["project_description"] = max(descs, key=len)

    return base


# ── Store import ─────────────────────────────────────────────────────────────

def import_to_store(cv_json: dict, person_name: str, store_dir: Path) -> str:
    """Save CV JSON to store, returns store_id."""
    store_id = hashlib.sha256(person_name.encode()).hexdigest()

    basics = cv_json.get("basics", {})
    exp = cv_json.get("experience", [])
    skills_text = json.dumps(cv_json.get("skills", {}), ensure_ascii=False).lower()

    search_text = " ".join(filter(None, [
        basics.get("name", ""),
        basics.get("current_title", ""),
        exp[0].get("company_name", "") if exp else "",
        person_name,
        "employee",
        skills_text,
    ])).lower()

    meta = {
        "id": store_id,
        "name": basics.get("name", ""),
        "role": basics.get("current_title", ""),
        "company": exp[0].get("company_name", "") if exp else "",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_filename": f"Employee: {person_name}",
        "comments": "Source: Employee DB",
        "search_text": search_text,
    }

    data = {"_meta": meta, **{k: v for k, v in cv_json.items() if k != "_meta"}}
    out = store_dir / f"{store_id}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return store_id


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scan employee folders and import CVs into webQCVT store")
    parser.add_argument("employees_dir", help="Path to Employees folder")
    parser.add_argument("--limit", type=int, default=0, help="Max employees to process (0=all)")
    parser.add_argument("--max-files", type=int, default=5, help="Max CV files per person to send to Gemini")
    parser.add_argument("--dry-run", action="store_true", help="Only scan, don't call Gemini")
    parser.add_argument("--store-dir", type=str, default=None, help="Store directory (default: _store/ in project)")
    parser.add_argument("--workers", "-w", type=int, default=3, help="Parallel workers (default: 3)")
    args = parser.parse_args()

    root = Path(args.employees_dir)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    store_dir = Path(args.store_dir) if args.store_dir else Path(__file__).parent / "_store"
    store_dir.mkdir(exist_ok=True)

    if not _gemini_api_key:
        print("Error: No Gemini API key found. Set GEMINI_API_KEY or configure ~/.quantoricv_settings.json")
        sys.exit(1)

    # Scan
    print(f"Scanning {root}...")
    employees = scan_all_employees(root)
    print(f"Found {len(employees)} employees with CV files\n")

    if args.limit:
        names = list(employees.keys())[:args.limit]
        employees = {k: employees[k] for k in names}

    # Process
    total = len(employees)
    _lock = threading.Lock()
    counters = {"success": 0, "failed": 0, "skipped": 0, "done": 0}
    t_start = time.time()

    # Build todo list (skip existing)
    todo = []
    for name, files in employees.items():
        if args.dry_run:
            counters["skipped"] += 1
            counters["done"] += 1
            continue
        store_id = hashlib.sha256(name.encode()).hexdigest()
        if (store_dir / f"{store_id}.json").exists():
            counters["skipped"] += 1
            counters["done"] += 1
            continue
        todo.append((name, files))

    skipped_initial = counters["skipped"]
    print(f"Skipping {skipped_initial} already imported. Processing {len(todo)} new.\n")

    if args.dry_run or not todo:
        print(f"Done! Skipped: {skipped_initial}, Total: {total}")
        return

    def process_one(name, files):
        try:
            cv_json = extract_and_merge(name, files, max_files=args.max_files)
            store_id = import_to_store(cv_json, name, store_dir)
            role = cv_json.get("basics", {}).get("current_title", "?")
            with _lock:
                counters["success"] += 1
                counters["done"] += 1
                _print_progress(name, role, True, None, counters, total, t_start)
        except Exception as e:
            with _lock:
                counters["failed"] += 1
                counters["done"] += 1
                _print_progress(name, None, False, e, counters, total, t_start)

    workers = min(args.workers, len(todo))
    print(f"Starting {workers} parallel workers...\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, name, files): name for name, files in todo}
        for f in as_completed(futures):
            pass  # results handled in process_one

    elapsed = time.time() - t_start
    el_min, el_sec = divmod(int(elapsed), 60)
    print(f"\n{'='*60}")
    print(f"Done in {el_min}m{el_sec:02d}s! Success: {counters['success']}, Failed: {counters['failed']}, Skipped: {skipped_initial}, Total: {total}")


def _print_progress(name, role, ok, error, counters, total, t_start):
    done = counters["done"]
    success = counters["success"]
    failed = counters["failed"]
    skipped = counters["skipped"]
    elapsed = time.time() - t_start
    processed = success + failed
    if processed > 0:
        avg = elapsed / processed
        remaining = avg * (total - done)
        eta_min, eta_sec = divmod(int(remaining), 60)
        eta_str = f"ETA {eta_min}m{eta_sec:02d}s"
    else:
        eta_str = "ETA --:--"
    bar_len = 30
    filled = int(bar_len * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = int(100 * done / total) if total else 0
    if ok:
        print(f"[{bar}] {pct}% ({done}/{total}) | ✓{success} ✗{failed} ⊘{skipped} | {eta_str} | ✓ {name.replace('.', ' ')} — {role}")
    else:
        print(f"[{bar}] {pct}% ({done}/{total}) | ✓{success} ✗{failed} ⊘{skipped} | {eta_str} | ✗ {name.replace('.', ' ')}: {error}")


if __name__ == "__main__":
    main()
