"""Registry of shell scripts workflow steps are allowed to run.

A ``shell`` step names a registered script by id. It never carries a command
line, because a workflow that could is remote code execution wearing a
workflow's clothes — especially once a standing policy rule lets it run
unattended. Registering is the deliberate, auditable act; running is a lookup.

Registration validates that the path exists and is a regular file, so a typo
fails at registration time instead of at 3am inside a scheduled run.

Registration is also where a script's **interface** is recorded. A script that
needs a directory and a URL before it can do anything used to say so only in
its own ``usage()`` text, which is prose: the registry kept a name, a path and
an interpreter, and the requirement was discarded. Nothing downstream could
then know that a step naming that script with no arguments was a step that
would stop and ask a human at 3am. ``parameters`` is that requirement in a
shape the authoring gate can read — see :func:`parse_declared_parameters` for
the declaration syntax and :data:`_PARAM_LINE`.

``None`` is not ``[]``. A script that declares nothing has an **unknown**
interface, which is what every row registered before this existed has, and what
a hand-written script with no declaration block has. ``[]`` is the stronger
statement that it was looked at and needs no arguments. Collapsing the two
would either invent a guarantee for old rows or fail every one of them.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from core.database import get_db_connection
from core.timestamps import to_utc_iso

_DEFAULT_TIMEOUT_SECONDS = 3600
_MAX_TIMEOUT_SECONDS = 6 * 3600
ALLOWED_INTERPRETERS = {"bash", "sh", "zsh", "python3", "node", "direct"}

# How a script declares what it needs. One line per parameter, in a comment, so
# the declaration lives in the file it describes and survives the user editing
# the draft before they approve it:
#
#     # @param CHECK_DIR required Directory whose free space is checked
#     # @param BEARER_TOKEN optional Sent as `Authorization: Bearer …`
#
# A comment marker is required. `@param` inside a here-doc or a string is not a
# declaration, and demanding the marker keeps the parse to lines that cannot be
# executable code. `required`/`optional` is spelled out rather than defaulted:
# which one a bare `@param NAME` meant would be a guess, and this whole change
# exists because something guessed.
_PARAM_MARKER = re.compile(r"^[ \t]*(?:#|//|--|;|\*)+[ \t]*@param\b")
_PARAM_LINE = re.compile(
    r"^[ \t]*(?:#|//|--|;|\*)+[ \t]*@param[ \t]+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]+"
    r"(?P<requirement>required|optional)"
    r"(?:[ \t]+(?P<description>.*))?[ \t]*$"
)

# Enough of a file to hold any plausible header block. A script is read here
# only to look for that block, and slurping an arbitrarily large file into
# memory on every registration would be a cost with no matching benefit.
_MAX_SCANNED_BYTES = 64 * 1024


class ScriptRegistrationError(ValueError):
    """Raised when a script cannot be registered as described."""


def _new_id() -> str:
    return f"script_{uuid.uuid4().hex}"


def _row_to_script(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "path": row["path"],
        "interpreter": row["interpreter"],
        "default_args": json.loads(row["default_args_json"] or "[]"),
        # NULL stays None all the way out to the API. A client that renders
        # "0 parameters" for a row whose interface was never captured is
        # telling the user something the registry does not know.
        "parameters": _stored_parameters(row["parameters_json"]),
        "timeout_seconds": row["timeout_seconds"],
        "created_by": row["created_by"],
        "created_at": to_utc_iso(row["created_at"]),
        "updated_at": to_utc_iso(row["updated_at"]),
    }


def _stored_parameters(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        # A row whose JSON will not parse is a row whose interface we cannot
        # read. That is the same state as never having captured one, and
        # saying "unknown" beats raising out of every list call.
        return None
    return loaded if isinstance(loaded, list) else None


def parse_declared_parameters(text: str) -> list[dict[str, Any]] | None:
    """Read a script's ``@param`` block, or ``None`` when it declares none.

    ``None`` means *this script did not say*, which is the honest answer for
    every script written before the convention existed. It is never conflated
    with ``[]`` — a file containing a declaration block that lists nothing is
    not a thing this can return, so ``[]`` only ever comes from a caller
    stating it outright.

    A line that opens like a declaration and then does not parse raises rather
    than being skipped. This module's whole premise is that a typo should fail
    at registration instead of at 3am, and a silently dropped ``@param`` is a
    required argument nobody will ever be asked for.
    """
    declared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not _PARAM_MARKER.match(line):
            continue
        match = _PARAM_LINE.match(line)
        if match is None:
            raise ScriptRegistrationError(
                f"line {line_number} looks like a parameter declaration but does "
                "not parse. Expected `# @param NAME required|optional "
                f"description` — got: {line.strip()}"
            )
        name = match.group("name")
        if name in seen:
            raise ScriptRegistrationError(
                f"line {line_number} declares parameter '{name}' twice"
            )
        seen.add(name)
        declared.append(
            {
                "name": name,
                "required": match.group("requirement") == "required",
                "description": (match.group("description") or "").strip(),
            }
        )
    return declared or None


def _parameters_from_file(path: str) -> list[dict[str, Any]] | None:
    """The declaration a registered file carries, if it carries one.

    Read from the file rather than taken from whoever asked for the script,
    so the declaration and the code it describes cannot drift apart: the
    reviewer who edits a drafted script before approving it edits both at once.
    An unreadable file (a compiled binary run through the ``direct``
    interpreter, say) simply declares nothing.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_MAX_SCANNED_BYTES)
    except OSError:
        return None
    return parse_declared_parameters(raw.decode("utf-8", "replace"))


def _normalize_parameters(raw: Any) -> list[dict[str, Any]] | None:
    """Validate a caller-supplied interface. ``None`` passes through as unknown."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ScriptRegistrationError("parameters must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw:
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        if not isinstance(entry, dict):
            raise ScriptRegistrationError(
                "each parameter must be an object with a name"
            )
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ScriptRegistrationError("each parameter needs a name")
        if name in seen:
            raise ScriptRegistrationError(f"parameter '{name}' is declared twice")
        seen.add(name)
        normalized.append(
            {
                "name": name,
                # Absent means required. A parameter whose necessity was not
                # stated is one the gate should ask about, not wave through.
                "required": bool(entry.get("required", True)),
                "description": str(entry.get("description") or "").strip(),
            }
        )
    return normalized


def _normalize_path(raw_path: str) -> str:
    text = str(raw_path or "").strip()
    if not text:
        raise ScriptRegistrationError("script path is required")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ScriptRegistrationError("script path must be absolute")
    if not path.exists():
        raise ScriptRegistrationError(f"script not found: {path}")
    if not path.is_file():
        raise ScriptRegistrationError(f"script path is not a file: {path}")
    return str(path)


def _normalize_interpreter(raw: str | None) -> str:
    value = (raw or "bash").strip().lower()
    if value not in ALLOWED_INTERPRETERS:
        raise ScriptRegistrationError(
            f"unsupported interpreter: {raw}. Use one of {sorted(ALLOWED_INTERPRETERS)}"
        )
    return value


def _normalize_args(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScriptRegistrationError("default_args must be a list")
    return [str(item) for item in raw]


def _normalize_timeout(raw: Any) -> int:
    if raw is None:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ScriptRegistrationError("timeout_seconds must be a number") from exc
    if value <= 0:
        raise ScriptRegistrationError("timeout_seconds must be positive")
    # An unbounded script would hold its schedule's task "active" forever,
    # which is the stall the scheduler grace period exists to avoid.
    return min(value, _MAX_TIMEOUT_SECONDS)


class ScriptStore:
    """CRUD over the registered-script table."""

    def register(
        self,
        *,
        name: str,
        path: str,
        description: str | None = None,
        interpreter: str | None = None,
        default_args: Any = None,
        parameters: Any = None,
        timeout_seconds: Any = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ScriptRegistrationError("script name is required")
        clean_path = _normalize_path(path)
        # Stated by the caller if they stated one; otherwise whatever the file
        # itself declares; otherwise unknown. The file is never allowed to
        # override an explicit answer — a caller who lists the interface has
        # looked at the script, and the script's own block may be the thing
        # they are correcting.
        clean_parameters = _normalize_parameters(parameters)
        if clean_parameters is None:
            clean_parameters = _parameters_from_file(clean_path)
        script_id = _new_id()
        with get_db_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM agent_scripts WHERE path = ?", (clean_path,)
            ).fetchone()
            if existing is not None:
                raise ScriptRegistrationError(f"script already registered: {clean_path}")
            conn.execute(
                """
                INSERT INTO agent_scripts (
                    id, name, description, path, interpreter,
                    default_args_json, parameters_json, timeout_seconds, created_by
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    script_id,
                    clean_name,
                    (description or "").strip() or None,
                    clean_path,
                    _normalize_interpreter(interpreter),
                    json.dumps(_normalize_args(default_args)),
                    None if clean_parameters is None else json.dumps(clean_parameters),
                    _normalize_timeout(timeout_seconds),
                    created_by,
                ),
            )
            conn.commit()
        return self.get(script_id) or {}

    def get(self, script_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_scripts WHERE id = ?", (script_id,)
            ).fetchone()
        return _row_to_script(row) if row else None

    def list_scripts(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                "SELECT * FROM agent_scripts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_script(row) for row in rows]

    def update(self, script_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        current = self.get(script_id)
        if current is None:
            return None
        fields: list[str] = []
        values: list[Any] = []
        if "name" in changes:
            clean = str(changes["name"] or "").strip()
            if not clean:
                raise ScriptRegistrationError("script name is required")
            fields.append("name = ?")
            values.append(clean)
        if "description" in changes:
            fields.append("description = ?")
            values.append((str(changes["description"] or "").strip()) or None)
        if "path" in changes:
            fields.append("path = ?")
            values.append(_normalize_path(changes["path"]))
        if "interpreter" in changes:
            fields.append("interpreter = ?")
            values.append(_normalize_interpreter(changes["interpreter"]))
        if "default_args" in changes:
            fields.append("default_args_json = ?")
            values.append(json.dumps(_normalize_args(changes["default_args"])))
        if "parameters" in changes:
            # Explicitly sending null is how a caller says "I no longer stand
            # behind this interface", which returns the row to unknown. It is
            # a different act from sending `[]`, and both are allowed.
            declared = _normalize_parameters(changes["parameters"])
            fields.append("parameters_json = ?")
            values.append(None if declared is None else json.dumps(declared))
        if "timeout_seconds" in changes:
            fields.append("timeout_seconds = ?")
            values.append(_normalize_timeout(changes["timeout_seconds"]))
        if not fields:
            return current
        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(script_id)
        with get_db_connection() as conn:
            conn.execute(
                f"UPDATE agent_scripts SET {', '.join(fields)} WHERE id = ?",
                tuple(values),
            )
            conn.commit()
        return self.get(script_id)

    def delete(self, script_id: str) -> bool:
        with get_db_connection() as conn:
            cursor = conn.execute("DELETE FROM agent_scripts WHERE id = ?", (script_id,))
            conn.commit()
        return cursor.rowcount > 0


_script_store: ScriptStore | None = None


def get_script_store() -> ScriptStore:
    global _script_store
    if _script_store is None:
        _script_store = ScriptStore()
    return _script_store


def required_parameters(script: Any) -> list[dict[str, Any]]:
    """The parameters a script says it cannot run without.

    ``[]`` for a script whose interface is unknown, which is the point: the
    caller asking this question is deciding whether to refuse something, and
    "we never captured what this needs" is not grounds to refuse.
    """
    if not isinstance(script, dict):
        return []
    declared = script.get("parameters")
    if not isinstance(declared, list):
        return []
    return [
        entry
        for entry in declared
        if isinstance(entry, dict) and entry.get("required", True)
    ]


__all__ = [
    "ALLOWED_INTERPRETERS",
    "ScriptRegistrationError",
    "ScriptStore",
    "get_script_store",
    "parse_declared_parameters",
    "required_parameters",
]
