#!/usr/bin/env python3
"""Pre-compute embeddings for all CVs in _store/. Saves to _cache/embeddings.npz"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from google import genai

STORE_DIR = Path(__file__).parent / "_store"
CACHE_DIR = Path(__file__).parent / "_cache"
EMBED_PATH = CACHE_DIR / "embeddings.npz"

EMBED_MODEL = "gemini-embedding-001"


def resolve_api_key() -> str:
    settings_path = Path.home() / ".quantoricv_settings.json"
    if settings_path.exists():
        return json.loads(settings_path.read_text()).get("api_key", "")
    return ""


def cv_text(cv: dict) -> str:
    basics = cv.get("basics", {})
    parts = [basics.get("current_title", "")]
    skills = cv.get("skills", {})
    for cat, items in skills.items():
        if isinstance(items, list):
            parts.append(", ".join(items))
        elif isinstance(items, str):
            parts.append(items)
    parts.append(cv.get("summary", ""))
    for exp in cv.get("experience", [])[:3]:
        parts.append(exp.get("role", ""))
        hl = exp.get("highlights", [])
        if isinstance(hl, list):
            for h in hl[:3]:
                if isinstance(h, str):
                    parts.append(h)
        env = exp.get("environment", "")
        if isinstance(env, str):
            parts.append(env)
    return " ".join(str(p) for p in parts if p)[:2000]


def compute_embedding(client: genai.Client, text: str) -> list[float] | None:
    try:
        result = client.models.embed_content(model=EMBED_MODEL, contents=text)
        return result.embeddings[0].values
    except Exception as e:
        print(f"  ⚠️ {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CV embeddings")
    parser.add_argument("-w", "--workers", type=int, default=5, help="Parallel workers")
    parser.add_argument("--force", action="store_true", help="Recompute all embeddings")
    args = parser.parse_args()

    api_key = resolve_api_key()
    if not api_key:
        print("ERROR: No API key found", file=sys.stderr)
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing
    existing_ids: list[str] = []
    existing_vecs: np.ndarray | None = None
    if EMBED_PATH.exists() and not args.force:
        data = np.load(EMBED_PATH, allow_pickle=True)
        existing_ids = data["ids"].tolist()
        existing_vecs = data["vecs"]
        print(f"Loaded {len(existing_ids)} existing embeddings")

    # Find missing
    store_files = sorted(STORE_DIR.glob("*.json"))
    existing_set = set(existing_ids)
    todo = [(p.stem, p) for p in store_files if p.stem not in existing_set]

    if not todo:
        print(f"All {len(existing_ids)} CVs already embedded. Use --force to recompute.")
        return

    print(f"Computing embeddings for {len(todo)} CVs ({args.workers} workers)...")

    new_ids: list[str] = []
    new_vecs: list[list[float]] = []
    done = 0
    errors = 0
    start_time = time.time()

    def process(sid: str, path: Path) -> tuple[str, list[float] | None]:
        data = json.loads(path.read_text(encoding="utf-8"))
        cv = {k: v for k, v in data.items() if not k.startswith("_")}
        text = cv_text(cv)
        if not text.strip():
            return sid, None
        client = genai.Client(api_key=api_key)
        vec = compute_embedding(client, text)
        return sid, vec

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, sid, path): sid for sid, path in todo}
        for future in as_completed(futures):
            sid, vec = future.result()
            done += 1
            if vec:
                new_ids.append(sid)
                new_vecs.append(vec)
            else:
                errors += 1

            if done % 10 == 0 or done == len(todo):
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(todo)} ({len(new_ids)} ok, {errors} err) "
                      f"| {rate:.1f}/s | ETA {int(eta)}s")

    # Merge with existing
    if new_vecs:
        new_matrix = np.array(new_vecs, dtype=np.float32)
        if existing_vecs is not None:
            all_ids = existing_ids + new_ids
            all_vecs = np.vstack([existing_vecs, new_matrix])
        else:
            all_ids = new_ids
            all_vecs = new_matrix

        np.savez_compressed(EMBED_PATH, ids=np.array(all_ids, dtype=object), vecs=all_vecs)
        print(f"\nSaved {len(all_ids)} embeddings to {EMBED_PATH}")
        print(f"  Dimensions: {all_vecs.shape[1]}")
        print(f"  File size: {EMBED_PATH.stat().st_size / 1024:.0f} KB")
    else:
        print("\nNo new embeddings computed")


if __name__ == "__main__":
    main()
