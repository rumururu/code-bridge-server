"""Secret store backed by ``~/.code-bridge/.env``.

The same file ``server.main._load_env_file_once`` reads at startup. This
module owns mutations to the file from now on; the boot loader stays
read-only and untouched.

Format is intentionally identical to what the boot loader understands:
``KEY=VALUE`` shell-style, one entry per line, ``#`` comment lines and
blank lines preserved when round-tripped (we keep the simpler "rewrite
KEY=VALUE entries" model — comments are not preserved on rewrite, since
they may have been added by a human editor and we don't want to silently
preserve stale annotations against fresh values).

All public functions are safe to call concurrently from FastAPI request
handlers thanks to a process-wide ``threading.Lock``.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path
from threading import Lock

_LOCK = Lock()

ENV_FILE: Path = Path.home() / ".code-bridge" / ".env"

# Keys allowed in the store. Reject anything not matching shell-style env
# var conventions (so injection of ``key\n=`` etc. fails at parse time).
_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SecretStoreError(ValueError):
    """Raised on validation / format failures."""


# Public API --------------------------------------------------------------

def list_keys() -> list[dict[str, bool]]:
    """Return ``[{"name": KEY, "has_value": bool}, ...]`` sorted by name.

    Values are **never** returned. Callers (and tests) rely on this — the
    GET /api/secrets route MUST NOT expose secret values to the client.
    """
    with _LOCK:
        return [
            {"name": name, "has_value": bool(value)}
            for name, value in sorted(_parse_lines().items())
        ]


def upsert(key: str, value: str) -> None:
    """Set or replace the value for ``key``.

    Side effects (all atomic under the lock):
      1. Write ``~/.code-bridge/.env`` with all entries sorted by key.
      2. Set file permission to ``0600`` (owner read/write only).
      3. Update ``os.environ[key]`` so in-process consumers see the new
         value without a server restart.

    ``value`` is never logged from this function — the caller (route /
    audit) must also log the key name only.
    """
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise SecretStoreError("key must match ^[A-Z][A-Z0-9_]*$")
    if not isinstance(value, str):
        raise SecretStoreError("value must be a string")
    if "\n" in value or "\r" in value:
        raise SecretStoreError("value must not contain newlines")
    with _LOCK:
        entries = _parse_lines()
        entries[key] = value
        _write_lines(entries)
        os.environ[key] = value


def delete(key: str) -> bool:
    """Delete ``key`` from the store *and* from ``os.environ``.

    Returns ``True`` if the key existed in the file before this call,
    ``False`` otherwise. A missing-key delete is **not** an error at this
    layer — the route turns it into a 404. We still validate the key
    syntax so a bogus path component fails fast.
    """
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise SecretStoreError("key must match ^[A-Z][A-Z0-9_]*$")
    with _LOCK:
        entries = _parse_lines()
        existed = key in entries
        if existed:
            del entries[key]
            _write_lines(entries)
            os.environ.pop(key, None)
        return existed


# Internal ---------------------------------------------------------------

def _parse_lines() -> dict[str, str]:
    """Read ``ENV_FILE`` into ``{key: value}``.

    Lines that don't match a valid ``KEY=VALUE`` shape, or whose key
    fails the regex check, are silently dropped — same lenient behaviour
    as the boot loader. ``KEY=`` with empty value is preserved as
    ``""`` so ``has_value`` correctly reports it as unset.
    """
    if not ENV_FILE.is_file():
        return {}
    try:
        raw_text = ENV_FILE.read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for raw in raw_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not _KEY_RE.match(key):
            continue
        out[key] = value
    return out


def _write_lines(entries: dict[str, str]) -> None:
    """Rewrite ``ENV_FILE`` with the given entries sorted by key."""
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{k}={v}\n" for k, v in sorted(entries.items()))
    ENV_FILE.write_text(content, encoding="utf-8")
    # 0600 — ``.env`` holds secrets, never world-readable. ``os.chmod``
    # is a no-op on Windows for the group/other bits, but we still call
    # it so the file is at least owner-only on the POSIX hosts we ship
    # to (macOS, Linux).
    if sys.platform != "win32":
        os.chmod(ENV_FILE, stat.S_IRUSR | stat.S_IWUSR)
