"""
PuffinZipAI — WebUI Credentials Manager
========================================
Auto-generates and persists a ``webui_credentials.json`` file in the project
root the first time the WebUI starts.  On subsequent runs the existing file is
loaded so the same credentials are reused.

Credential file format::

    {
        "username":   "<20-char alphanumeric>",
        "password":   "<64-char alphanumeric>",
        "secret_key": "<64-char hex>"
    }

Environment-variable overrides (``PUFFIN_USERNAME``, ``PUFFIN_PASSWORD``,
``PUFFIN_SECRET_KEY``) still take precedence when set.

The file is added to ``.gitignore`` and must **never** be committed.
"""

from __future__ import annotations

import json
import os
import secrets
import string
from pathlib import Path
from typing import TypedDict


class Credentials(TypedDict):
    username: str
    password: str
    secret_key: str


# ── Default paths ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_CREDENTIALS_FILE = _PROJECT_ROOT / "webui_credentials.json"

# Character pools
_ALPHA_POOL = string.ascii_letters + string.digits          # a-zA-Z0-9
_USERNAME_LENGTH = 20
_PASSWORD_LENGTH = 64
_SECRET_KEY_LENGTH = 64  # hex chars → 32 bytes of entropy


def _generate_random_string(length: int, pool: str = _ALPHA_POOL) -> str:
    """Generate a cryptographically random string from *pool*."""
    return ''.join(secrets.choice(pool) for _ in range(length))


def _generate_credentials() -> Credentials:
    """Create a fresh set of credentials with random values."""
    return Credentials(
        username=_generate_random_string(_USERNAME_LENGTH),
        password=_generate_random_string(_PASSWORD_LENGTH),
        secret_key=secrets.token_hex(_SECRET_KEY_LENGTH // 2),  # 32 bytes → 64 hex chars
    )


def _load_credentials_file(path: Path = _CREDENTIALS_FILE) -> Credentials | None:
    """Load credentials from disk, returning *None* on any failure."""
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        # Validate required keys
        if all(k in data for k in ('username', 'password', 'secret_key')):
            return Credentials(
                username=str(data['username']),
                password=str(data['password']),
                secret_key=str(data['secret_key']),
            )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def _save_credentials_file(creds: Credentials, path: Path = _CREDENTIALS_FILE) -> None:
    """Persist credentials to disk with restrictive permissions."""
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(dict(creds), fh, indent=2)
        fh.write('\n')
    # Best-effort: restrict read to owner only (no-op on Windows)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_or_create_credentials(path: Path = _CREDENTIALS_FILE) -> Credentials:
    """
    Return WebUI credentials, creating the file if it doesn't exist.

    Priority order:
    1. Environment variables (``PUFFIN_USERNAME``, ``PUFFIN_PASSWORD``,
       ``PUFFIN_SECRET_KEY``) — if **all three** are set they take full
       precedence and no file is read/written.
    2. Existing ``webui_credentials.json`` on disk.
    3. Auto-generated credentials (written to disk for reuse).
    """
    # 1. Full env-var override
    env_user = os.environ.get('PUFFIN_USERNAME', '').strip()
    env_pass = os.environ.get('PUFFIN_PASSWORD', '').strip()
    env_key  = os.environ.get('PUFFIN_SECRET_KEY', '').strip()
    if env_user and env_pass and env_key:
        return Credentials(username=env_user, password=env_pass, secret_key=env_key)

    # 2. Try loading from file
    creds = _load_credentials_file(path)
    if creds is not None:
        # Allow partial env-var overrides on top of the file
        if env_user:
            creds['username'] = env_user
        if env_pass:
            creds['password'] = env_pass
        if env_key:
            creds['secret_key'] = env_key
        return creds

    # 3. Generate fresh credentials and persist
    creds = _generate_credentials()
    _save_credentials_file(creds, path)
    return creds
