"""Authentication module for webQCVT.

Designed to be swappable: replace login logic with MS OAuth later,
keep JWT session, roles, and pane visibility unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Request, HTTPException

# ── Roles ──
ADMIN = "admin"
USER = "user"
BASIC = "basic"
ALL_ROLES = (ADMIN, USER, BASIC)

# ── Pane visibility per role ──
ROLE_PANES = {
    ADMIN: ["paneConvert", "paneBatch", "paneJD", "paneXray", "paneGithub", "paneLogs", "panePrompt"],
    USER:  ["paneConvert", "paneBatch", "paneJD", "paneXray", "paneGithub", "paneLogs"],
    BASIC: ["paneConvert"],
}

# ── Configuration ──
_DATA_DIR: Path = Path(".")
_SECRET_KEY: str = ""
_USERS_LOCK = threading.Lock()
_users_cache: list[dict] | None = None

COOKIE_NAME = "session"
TOKEN_TTL = 86400  # 24 hours


def init(data_dir: Path):
    """Call once at app startup."""
    global _DATA_DIR, _SECRET_KEY
    _DATA_DIR = data_dir

    # Load or generate secret key
    secret_path = data_dir / ".session_secret"
    env_secret = os.environ.get("SESSION_SECRET", "")
    if env_secret:
        _SECRET_KEY = env_secret
    elif secret_path.exists():
        _SECRET_KEY = secret_path.read_text().strip()
    else:
        _SECRET_KEY = secrets.token_hex(32)
        secret_path.write_text(_SECRET_KEY)

    # Seed admin if no users file
    _seed_admin()


def _users_path() -> Path:
    return _DATA_DIR / "_users.json"


# ── User store ──

def _load_users() -> list[dict]:
    global _users_cache
    with _USERS_LOCK:
        if _users_cache is not None:
            return _users_cache
        p = _users_path()
        if p.exists():
            _users_cache = json.loads(p.read_text(encoding="utf-8"))
        else:
            _users_cache = []
        return _users_cache


def _save_users(users: list[dict]):
    global _users_cache
    with _USERS_LOCK:
        _users_cache = users
        _users_path().write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")


def _seed_admin():
    users = _load_users()
    if not users:
        users.append({
            "email": "admin@quantori.com",
            "name": "Admin",
            "role": ADMIN,
            "password_hash": _hash_password("admin"),
            "password_plain": "admin",
            "active": True,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        _save_users(users)


def get_user(email: str) -> dict | None:
    for u in _load_users():
        if u["email"].lower() == email.lower():
            return u
    return None


def list_users() -> list[dict]:
    """Return users without password hashes (but include plain password for admin view)."""
    return [{k: v for k, v in u.items() if k != "password_hash"} for u in _load_users()]


def upsert_user(email: str, name: str = "", role: str = USER, password: str = "", active: bool = True) -> dict:
    users = _load_users()
    existing = None
    for u in users:
        if u["email"].lower() == email.lower():
            existing = u
            break
    if existing:
        if name:
            existing["name"] = name
        if role in ALL_ROLES:
            existing["role"] = role
        if password:
            existing["password_hash"] = _hash_password(password)
            existing["password_plain"] = password
        existing["active"] = active
    else:
        generated_pw = password or secrets.token_hex(8)
        users.append({
            "email": email.lower(),
            "name": name or email.split("@")[0],
            "role": role if role in ALL_ROLES else USER,
            "password_hash": _hash_password(generated_pw),
            "password_plain": generated_pw,
            "active": active,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    _save_users(users)
    return get_user(email)


def delete_user(email: str) -> bool:
    users = _load_users()
    before = len(users)
    users = [u for u in users if u["email"].lower() != email.lower()]
    if len(users) < before:
        _save_users(users)
        return True
    return False


# ── Password hashing ──

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}:{h.hex()}"


def _verify_password(password: str, hashed: str) -> bool:
    try:
        salt, h = hashed.split(":")
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return expected.hex() == h
    except Exception:
        return False


# ── JWT tokens ──

def create_token(email: str, role: str) -> str:
    return jwt.encode(
        {"email": email, "role": role, "exp": time.time() + TOKEN_TTL},
        _SECRET_KEY,
        algorithm="HS256",
    )


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=["HS256"])
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ── FastAPI dependencies ──

def get_current_user(request: Request) -> dict | None:
    """Extract user from session cookie. Returns None if not authenticated."""
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    user = get_user(payload["email"])
    if not user or not user.get("active"):
        return None
    return {"email": user["email"], "name": user["name"], "role": user["role"]}


def require_auth(request: Request) -> dict:
    """Dependency: require any authenticated user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_role(*roles: str):
    """Dependency factory: require specific role(s)."""
    def checker(request: Request) -> dict:
        user = require_auth(request)
        if user["role"] not in roles:
            raise HTTPException(status_code=404, detail="Not found")
        return user
    return checker


# ── Auth endpoint handlers ──

def handle_login(email: str, password: str) -> tuple[str, dict] | None:
    """Validate credentials, return (token, user_info) or None."""
    user = get_user(email)
    if not user or not user.get("active"):
        return None
    if not _verify_password(password, user.get("password_hash", "")):
        return None
    token = create_token(user["email"], user["role"])
    info = {"email": user["email"], "name": user["name"], "role": user["role"]}
    return token, info


def user_info_response(user: dict) -> dict:
    """Build response for /api/auth/me."""
    role = user.get("role", BASIC)
    return {
        "email": user["email"],
        "name": user["name"],
        "role": role,
        "allowed_panes": ROLE_PANES.get(role, ROLE_PANES[BASIC]),
    }
