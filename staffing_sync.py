"""Sync employee + position data from the Quantori Staffing API.

The legacy schemas in `_employees.json` and `_positions.json` are preserved so the
rest of the app does not need to change. Sync is one-shot: fetch everything,
write atomically.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


class StaffingSyncError(Exception):
    pass


def _post_json(base_url: str, path: str, token: str, body: Any, timeout: int = 60) -> Any:
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise StaffingSyncError(f"HTTP {e.code} on POST {path}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise StaffingSyncError(f"Network error on POST {path}: {e.reason}") from e


def _get_json(base_url: str, path: str, token: str, timeout: int = 60) -> Any:
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise StaffingSyncError(f"HTTP {e.code} on GET {path}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise StaffingSyncError(f"Network error on GET {path}: {e.reason}") from e


def _iso_date(s: str | None) -> str:
    """Convert ISO datetime like '2022-05-16T00:00:00Z' into 'YYYY-MM-DD'."""
    if not s:
        return ""
    return s[:10]


def fetch_all_employees(base_url: str, token: str, page_size: int = 200) -> list[dict]:
    """Paginate through /api/Employee/getAll. Returns API-shape list."""
    out: list[dict] = []
    skip = 0
    while True:
        chunk = _post_json(base_url, "/api/Employee/getAll", token, {"take": page_size, "skip": skip})
        if not isinstance(chunk, list) or not chunk:
            break
        out.extend(chunk)
        if len(chunk) < page_size:
            break
        skip += page_size
    return out


def fetch_projects(base_url: str, token: str) -> list[dict]:
    return _get_json(base_url, "/api/Project", token) or []


def api_employee_to_local(e: dict) -> dict:
    """Map API employee into the legacy _employees.json schema."""
    first = (e.get("firstName") or "").strip()
    last = (e.get("lastName") or "").strip()
    name = f"{first} {last}".strip()
    return {
        "staffing_id": e.get("id"),
        "global_id": e.get("globalId", "") or "",
        "employee_name": name,
        "email": e.get("email", "") or "",
        "employment_status": (e.get("employmentStatus") or {}).get("name", "") or "",
        "job_title": (e.get("jobTitle") or {}).get("name", "") or "",
        "resource_pool": (e.get("resourcePool") or {}).get("name", "") or "",
        "join_date": _iso_date(e.get("joinDate")),
        "dismiss_date": _iso_date(e.get("dismissDate")),
        "bamboo_id": e.get("bambooHrId", "") or "",
    }


def _project_account_map(projects: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in projects:
        pid = p.get("id")
        if pid is None:
            continue
        acc = p.get("account") or {}
        out[pid] = {
            "project_name": p.get("name", "") or "",
            "account_name": acc.get("name", "") or "",
            "account_code": acc.get("code", "") or "",
        }
    return out


def _workload_role(w: dict) -> str:
    """Get billing type name from workload position. Fall back to billingTypeId mapping."""
    pos = w.get("position") or {}
    name = pos.get("positionBillingTypeName") or (pos.get("positionBillingType") or {}).get("name")
    if name:
        return name
    # Numeric fallback — covers all common billing types we've seen
    fallback = {1: "Billable", 2: "Non-billable", 3: "Internal Projects",
                4: "Training", 5: "Temporary", 6: "Governance", 7: "Non-production"}
    return fallback.get(w.get("billingTypeId"), "")


def api_employees_to_positions(api_employees: list[dict], projects: list[dict]) -> list[dict]:
    """Flatten allWorkloads[] across all employees into legacy _positions.json records."""
    proj_map = _project_account_map(projects)
    out: list[dict] = []
    for e in api_employees:
        first = (e.get("firstName") or "").strip()
        last = (e.get("lastName") or "").strip()
        name = f"{first} {last}".strip()
        email = e.get("email", "") or ""
        sid = e.get("id")
        for w in e.get("allWorkloads") or []:
            proj = w.get("project") or {}
            pid = proj.get("id")
            acc_info = proj_map.get(pid, {})
            out.append({
                "employee_name": name,
                "employee_email": email,
                "staffing_id": sid,
                "project_name": proj.get("name", "") or acc_info.get("project_name", ""),
                "account_name": acc_info.get("account_name", ""),
                "account_code": acc_info.get("account_code", ""),
                "role": _workload_role(w),
                "open_date": _iso_date(w.get("openDate")),
                "close_date": _iso_date(w.get("closeDate")),
                "load": w.get("load"),
                "actual_load": w.get("actualLoad"),
                "status": (w.get("workloadStatus") or {}).get("name", "") or "",
                "is_closed": bool(w.get("isClosed")),
            })
    return out


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def sync_all(base_url: str, token: str, employees_path: Path, positions_path: Path) -> dict:
    """Run a full sync. Returns a summary dict for logging / UI."""
    if not base_url or not token:
        raise StaffingSyncError("Staffing API URL and token are required")
    started = time.time()

    # 1. Employees
    api_emps = fetch_all_employees(base_url, token)
    # 2. Projects (for account lookup)
    projects = fetch_projects(base_url, token)

    local_emps = [api_employee_to_local(e) for e in api_emps]
    positions = api_employees_to_positions(api_emps, projects)

    # Diff vs previous
    prev_emps = []
    if employees_path.exists():
        try:
            prev_emps = json.loads(employees_path.read_text(encoding="utf-8"))
        except Exception:
            prev_emps = []
    prev_by_id = {e.get("staffing_id"): e for e in prev_emps if e.get("staffing_id") is not None}
    new_by_id = {e.get("staffing_id"): e for e in local_emps if e.get("staffing_id") is not None}
    added = [new_by_id[k] for k in new_by_id.keys() - prev_by_id.keys()]
    removed = [prev_by_id[k] for k in prev_by_id.keys() - new_by_id.keys()]
    status_changed = [
        k for k in new_by_id.keys() & prev_by_id.keys()
        if prev_by_id[k].get("employment_status") != new_by_id[k].get("employment_status")
    ]

    # Atomic write
    _atomic_write_json(employees_path, local_emps)
    _atomic_write_json(positions_path, positions)

    return {
        "ok": True,
        "duration_sec": round(time.time() - started, 1),
        "employees_total": len(local_emps),
        "employees_active": sum(1 for e in local_emps if e.get("employment_status") == "Active"),
        "employees_dismissed": sum(1 for e in local_emps if e.get("employment_status") == "Dismissed"),
        "employees_added": len(added),
        "employees_removed": len(removed),
        "employees_status_changed": len(status_changed),
        "positions_total": len(positions),
        "projects_fetched": len(projects),
    }
