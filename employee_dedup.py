#!/usr/bin/env python3
"""
Employee Dedup — remove duplicate experience entries from imported CVs.

Sends each CV JSON (not files) to Gemini asking to merge duplicate roles.
Much faster/cheaper than re-extraction since it's JSON-to-JSON.

Usage:
    python employee_dedup.py [--limit N] [--dry-run]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from google import genai
from converter_engine import choose_model_name, extract_first_json_object, resolve_api_key

_settings_path = Path.home() / ".quantoricv_settings.json"
_config = json.loads(_settings_path.read_text()) if _settings_path.exists() else {}
_gemini_api_key = resolve_api_key(Path(__file__).parent, _config)

DEDUP_PROMPT = """You are a CV data deduplicator. The JSON below contains a person's CV with DUPLICATE experience entries — the same role at the same company appears multiple times with overlapping dates because it was merged from multiple CV versions.

Your task:
1. Find experience entries that describe the SAME role at the SAME company (matching by company name and overlapping/similar date ranges).
2. MERGE duplicates into ONE entry: combine all unique highlights, pick the widest date range, merge environment lists.
3. Keep all NON-duplicate entries unchanged.
4. Do NOT modify basics, summary, skills, education, certifications, languages, or other_sections.
5. Do NOT invent any new information. Only reorganize what's already there.
6. Do NOT remove any highlights — merge them all into the combined entry, removing only exact duplicates.

INPUT JSON:
{cv_json}

Return the COMPLETE fixed JSON object (all sections, not just experience). No markdown wrappers."""


def _retry(fn, max_retries=5):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            if ("429" in err or "Resource" in err or "Quota" in err) and attempt < max_retries - 1:
                delay = [5, 5, 5, 10, 10][attempt]
                print(f"  ⚠️ Rate limit, sleeping {delay}s")
                time.sleep(delay)
            else:
                raise


def has_duplicate_experience(cv_json):
    """Check if CV has potential duplicate experience entries."""
    exp = cv_json.get("experience", [])
    if len(exp) < 2:
        return False
    # Check for same company appearing multiple times
    companies = {}
    for e in exp:
        key = (e.get("company_name", "").lower().strip(), e.get("role", "").lower().strip()[:30])
        companies[key] = companies.get(key, 0) + 1
    return any(v > 1 for v in companies.values())


def dedup_cv(cv_json, model, client):
    """Send CV JSON to Gemini for dedup. Returns fixed JSON."""
    # Strip meta/sessions before sending
    clean = {k: v for k, v in cv_json.items() if not k.startswith("_")}
    prompt = DEDUP_PROMPT.format(cv_json=json.dumps(clean, ensure_ascii=False, indent=2))

    response = _retry(
        lambda: client.models.generate_content(model=model, contents=prompt)
    )
    txt = getattr(response, "text", None)
    if not txt:
        raise RuntimeError("Empty response")
    return extract_first_json_object(txt)


def main():
    parser = argparse.ArgumentParser(description="Deduplicate experience entries in stored CVs")
    parser.add_argument("--limit", type=int, default=0, help="Max CVs to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Only detect duplicates, don't fix")
    parser.add_argument("--store-dir", type=str, default=None)
    args = parser.parse_args()

    store_dir = Path(args.store_dir) if args.store_dir else Path(__file__).parent / "_store"
    if not _gemini_api_key:
        print("Error: No Gemini API key")
        sys.exit(1)

    client = genai.Client(api_key=_gemini_api_key)
    model = choose_model_name(_config)

    # Find CVs with duplicates
    candidates = []
    for p in sorted(store_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "Employee" not in data.get("_meta", {}).get("source_filename", ""):
                continue
            if has_duplicate_experience(data):
                exp_count = len(data.get("experience", []))
                name = data.get("_meta", {}).get("name", "?")
                candidates.append((p, data, name, exp_count))
        except Exception:
            continue

    print(f"Found {len(candidates)} CVs with potential duplicate experience\n")

    if args.limit:
        candidates = candidates[:args.limit]

    success = 0
    failed = 0
    total_removed = 0
    t_start = time.time()
    total = len(candidates)

    for i, (path, data, name, exp_before) in enumerate(candidates, 1):
        # Progress bar
        done = i - 1
        elapsed = time.time() - t_start
        if done > 0 and (success + failed) > 0:
            avg = elapsed / (success + failed)
            remaining = avg * (total - done)
            eta_min, eta_sec = divmod(int(remaining), 60)
            eta_str = f"ETA {eta_min}m{eta_sec:02d}s"
        else:
            eta_str = "ETA --:--"

        bar_len = 30
        filled = int(bar_len * done / total) if total else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = int(100 * done / total) if total else 0
        print(f"\n[{bar}] {pct}% ({done}/{total}) | ✓{success} ✗{failed} | -{total_removed} dupes | {eta_str}")
        print(f"[{i}/{total}] {name} ({exp_before} exp)")

        if args.dry_run:
            exp = data.get("experience", [])
            seen = {}
            for e in exp:
                key = e.get("company_name", "").lower().strip()
                seen.setdefault(key, []).append(e.get("role", ""))
            for company, roles in seen.items():
                if len(roles) > 1:
                    print(f"  DUP: {company} x{len(roles)}")
            continue

        try:
            fixed = dedup_cv(data, model, client)
            exp_after = len(fixed.get("experience", []))
            removed = exp_before - exp_after

            for k in data:
                if k.startswith("_"):
                    fixed[k] = data[k]

            path.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
            total_removed += removed
            print(f"  ✓ {exp_before} → {exp_after} ({'-' + str(removed) if removed else 'no change'})")
            success += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    elapsed = time.time() - t_start
    m, s = divmod(int(elapsed), 60)
    print(f"\n{'='*60}")
    print(f"Done in {m}m{s:02d}s! Success: {success}, Failed: {failed}")
    print(f"Total duplicate entries removed: {total_removed}")


if __name__ == "__main__":
    main()
