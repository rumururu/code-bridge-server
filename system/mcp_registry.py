"""Code Bridge's own place to register MCP servers.

Until this module existed, `agent/capability_registry.py::_mcp_config_paths`
looked in exactly two places for MCP servers: `~/.claude.json` — the Claude
Code CLI's own settings file — and `<cwd>/.mcp.json`. So a Code Bridge user who
wanted their agent to reach an MCP server had to install a *different* product
and hand-edit its JSON. For a phone app whose whole premise is building agents
without a terminal, that is not a path anybody walks: the tool picker would
offer nothing, every `mcp_tool` step would park with "not configured on this
machine", and the only fix lived on a keyboard the user was not at.

This module is the missing store. It keeps the servers the *user of this
product* registered, in the same `app_settings` key-value table as the browser
preferences and the LLM selection (`core/database.py::SettingsDB`), so an
install that has registered nothing has no row and behaves exactly as before.

Two rules hold the whole thing honest:

*   **The entry shape is not new.** What is stored is the same `mcpServers`
    entry shape those two JSON files already use, and the same shape
    `capability_registry._sdk_mcp_server_config` already knows how to translate
    into a Claude SDK launch config. Inventing a Code Bridge-flavoured schema
    would mean a second translator, and two translators disagree eventually.
*   **Validation happens at write time.** An entry is only stored if
    `_sdk_mcp_server_config` can actually turn it into a launch config. The
    alternative is a server that lists fine in the app, is silently dropped at
    launch (`detected_mcp_server_configs` omits what it cannot translate), and
    parks the step days later with a message that names the server as missing —
    a registration the user watched succeed and that never existed.

Secrets: a stdio entry's `env` and an http entry's `headers` are exactly where
an API token goes, and the SDK needs them verbatim to start the server. They
are stored as given and **never** leave this machine in readable form:
:func:`public_server_view` emits key *names* only, the same discipline
`routes/secrets.py` already applies to `~/.code-bridge/.env`.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from audit.audit_redactor import SECRET_KEYS, redact_payload

logger = logging.getLogger(__name__)

#: One `app_settings` row holds the whole registry, as
#: ``{"servers": {name: entry}}``. The wrapper object exists so a later field
#: (an enabled flag, a last-verified timestamp) can be added without having to
#: tell an old row apart from a new one.
SETTING_KEY = "mcp_servers"

#: Label written on every entry that came from here, so a catalog row, a
#: verification message, or an API response can say *where* a server was
#: declared. Distinct from the `claude_cli_config` / `project_mcp_config`
#: labels `capability_registry._mcp_config_paths` uses for the two files.
REGISTRY_ORIGIN = "code_bridge_registry"

#: Human-readable location, for the same messages. Where the two file origins
#: print a path, this prints a place.
REGISTRY_LOCATION = "Code Bridge's own MCP server registry"

#: `browser` names the server's *own* Playwright runtime to every
#: `browser_action` step (`capability_registry.BROWSER_RUNTIME_CAPABILITY_NAME`),
#: and `_merged_mcp_servers` drops any configured server that claims it. Storing
#: one here would therefore be accepted and then silently ignored — the exact
#: outcome write-time validation exists to prevent — so it is refused up front.
RESERVED_SERVER_NAMES = frozenset({"browser"})

#: What a masked value reads as. Not the empty string: "this field is set and
#: withheld" and "this field is unset" are different facts, and a UI that
#: cannot tell them apart will offer to fill in a token that is already there.
MASKED = "[redacted]"


# --- secret-name matching ----------------------------------------------------
# `audit/audit_redactor.py` already owns the project's list of field names that
# carry secrets, so that list is reused rather than restated. Its own matcher
# (`_key_matches_secret`) is private and tuned for audit payload keys — flat
# names like `api_key` or anything ending `_token`. MCP env vars and URL query
# parameters are named differently (`GITHUB_TOKEN`, `N8N_API_KEY`,
# `?access_key=`), so the suffix set below widens the same list to catch them.
_SECRET_NAME_SUFFIXES = ("_token", "_key", "_secret", "_password", "_apikey", "_auth")


def _is_secret_name(name: Any) -> bool:
    """Whether a field name is one that conventionally carries a credential."""
    normalized = str(name).strip().lower().replace("-", "_")
    return normalized in SECRET_KEYS or normalized.endswith(_SECRET_NAME_SUFFIXES)


# --- validation --------------------------------------------------------------


class McpRegistryError(ValueError):
    """A registration this module refuses, with the reason a person can act on."""


def _translate(config: Any) -> dict[str, Any] | None:
    """Ask the one translator whether this entry can actually be launched.

    Imported here rather than at module scope because
    `agent/capability_registry.py` imports *this* module to merge the registry
    into detection; a top-level import in both directions is a cycle.
    """
    from agent.capability_registry import _sdk_mcp_server_config

    return _sdk_mcp_server_config(config)


def normalize_server_name(name: Any) -> str:
    """The stored key for a server name, or raise saying why it is not one."""
    text = str(name or "").strip()
    if not text:
        raise McpRegistryError("An MCP server needs a name.")
    if text in RESERVED_SERVER_NAMES:
        raise McpRegistryError(
            f'"{text}" is reserved for the built-in browser runtime this server '
            "provides itself. Pick a different name."
        )
    return text


def validate_server_config(config: Any) -> dict[str, Any]:
    """Return the entry to store, or raise saying what it is missing.

    The check is the translation itself, not a lookalike of it: whatever
    `_sdk_mcp_server_config` accepts is what gets stored, so a stored server can
    never be one the launch path drops.
    """
    if not isinstance(config, dict):
        raise McpRegistryError(
            "An MCP server entry must be an object with either a command "
            "(stdio) or a url (http/sse)."
        )
    if _translate(config) is None:
        raise McpRegistryError(
            "This MCP server entry cannot be launched from what it declares. "
            "A stdio server needs a non-empty \"command\"; an http or sse "
            "server needs a non-empty \"url\". Entries of type \"sdk\" describe "
            "an in-process object and cannot be registered here."
        )
    return dict(config)


# --- storage -----------------------------------------------------------------


def _read_raw() -> dict[str, Any]:
    """The stored registry. An unreadable store means empty, never a crash.

    Same posture as `system/browser_preferences.py::get_browser_preferences`:
    this is read on the detection path that every agent run and every catalog
    refresh goes through, so a corrupt row must cost the user their registered
    servers, not their server.
    """
    raw: Any = {}
    try:
        from core.database import get_settings_db

        raw = get_settings_db().get_json(SETTING_KEY, {})
    except Exception:  # noqa: BLE001 - an MCP setting must not take the server down
        logger.warning("MCP registry could not be read; treating it as empty", exc_info=True)
        return {}
    if not isinstance(raw, dict):
        return {}
    servers = raw.get("servers")
    if not isinstance(servers, dict):
        return {}
    return {
        str(name): config
        for name, config in servers.items()
        if isinstance(name, str) and name.strip() and isinstance(config, dict)
    }


def _write_raw(servers: dict[str, Any]) -> None:
    from core.database import get_settings_db

    get_settings_db().set_json(SETTING_KEY, {"servers": servers})


def list_registered_servers() -> dict[str, dict[str, Any]]:
    """Every server registered through Code Bridge, keyed by name.

    Values are the raw entries — they may hold credentials in ``env`` or
    ``headers``, because the SDK needs them to start the server. Anything that
    renders these for a person must go through :func:`public_server_view`.
    """
    return {name: dict(config) for name, config in _read_raw().items()}


def upsert_server(name: str, config: Any) -> dict[str, Any]:
    """Register (or replace) one MCP server. Returns the masked public view.

    Raises :class:`McpRegistryError` when the name or the entry is one this
    server would not be able to honour later.
    """
    key = normalize_server_name(name)
    entry = validate_server_config(config)
    servers = _read_raw()
    servers[key] = entry
    _write_raw(servers)
    return public_server_view(key, entry)


def remove_server(name: str) -> bool:
    """Drop one registration. ``False`` when there was nothing by that name."""
    key = str(name or "").strip()
    if not key:
        return False
    servers = _read_raw()
    if key not in servers:
        return False
    del servers[key]
    _write_raw(servers)
    return True


# --- masked representation ---------------------------------------------------


def _mask_args(args: Any) -> list[str]:
    """Argument list with credential-looking values replaced.

    Two shapes are covered because both appear in real `mcpServers` entries:
    ``--api-key=abc`` and ``--api-key abc``. The remaining arguments are worth
    showing — ``-y @playwright/mcp@latest`` is how a person recognises which
    server a row is — so they are kept and then run through the audit
    redactor, which catches a bare JWT or AWS key wherever it sits.
    """
    if not isinstance(args, list):
        return []
    masked: list[str] = []
    mask_next = False
    for raw in args:
        arg = str(raw)
        if mask_next:
            masked.append(MASKED)
            mask_next = False
            continue
        if "=" in arg:
            flag = arg.split("=", 1)[0]
            if _is_secret_name(flag.lstrip("-")):
                masked.append(f"{flag}={MASKED}")
                continue
        elif arg.startswith("-") and _is_secret_name(arg.lstrip("-")):
            masked.append(arg)
            mask_next = True
            continue
        masked.append(arg)
    return _redact_patterns(masked)


def _mask_url(url: Any) -> str | None:
    """URL with any embedded password or credential query value removed."""
    text = str(url or "").strip()
    if not text:
        return None
    try:
        parts = urlsplit(text)
    except ValueError:
        return MASKED
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, _, host = netloc.rpartition("@")
        user = userinfo.split(":", 1)[0]
        netloc = f"{user}:{MASKED}@{host}" if ":" in userinfo else f"{user}@{host}"
    query = parts.query
    if query:
        query = urlencode(
            [
                (key, MASKED if _is_secret_name(key) else value)
                for key, value in parse_qsl(query, keep_blank_values=True)
            ]
        )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _redact_patterns(value: Any) -> Any:
    """Second pass for secrets that no field name announced.

    Reuses `audit/audit_redactor.py`, which is the project's existing detector
    for JWTs, AWS keys, private key blocks, and emails sitting inside a string.
    """
    redacted, _categories = redact_payload(value)
    return redacted


def public_server_view(name: str, config: Any) -> dict[str, Any]:
    """One registered server as it may be shown to a person or an API client.

    `env` and `headers` are reduced to their key names. Not masked values —
    names. The value of `GITHUB_TOKEN` has no rendering that is both useful and
    safe, and a partially-masked token is still a leak of its length and shape;
    what a UI actually needs is "this variable is set", which the name plus its
    presence in the list already says.
    """
    settings = config if isinstance(config, dict) else {}
    launchable = _translate(settings)
    transport = (launchable or {}).get("type")
    command = settings.get("command")
    env = settings.get("env")
    headers = settings.get("headers")
    return {
        "name": name,
        "origin": REGISTRY_ORIGIN,
        "transport": transport,
        "command": _redact_patterns(command) if isinstance(command, str) else None,
        "args": _mask_args(settings.get("args")),
        "url": _mask_url(settings.get("url")),
        "env_keys": sorted(str(key) for key in env) if isinstance(env, dict) else [],
        "header_keys": sorted(str(key) for key in headers) if isinstance(headers, dict) else [],
        "launchable": launchable is not None,
    }


def public_registry() -> list[dict[str, Any]]:
    """Every registration, masked, in name order."""
    servers = _read_raw()
    return [public_server_view(name, servers[name]) for name in sorted(servers)]
