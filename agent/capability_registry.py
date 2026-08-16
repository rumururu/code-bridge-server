"""Discovery and persistence for Work Cockpit capabilities.

The catalog is a promise: every row says "the cockpit can route through
this". A row for something that does not exist on this machine is a
fabricated promise, and the only place it can be discovered is at run time,
by the user, when the step silently does nothing. So MCP servers are read
from the local Claude CLI config instead of being hardcoded, the built-in
browser runtime takes its status from an actual readiness probe, and
anything we cannot verify is reported as unverified rather than available.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.llm_settings import get_llm_options_snapshot

from .agent_store import get_agent_store
from .browser_action_adapter import (
    PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
    get_cached_browser_readiness_sync,
)

logger = logging.getLogger(__name__)


BUILTIN_CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "type": "builtin",
        "name": "file.read",
        "description": "Read project files through Code Bridge.",
        "permission_level": "read",
        "desktop_only": False,
        "local_only": True,
    },
    {
        "type": "builtin",
        "name": "file.write",
        "description": "Create or edit project files with approval policy.",
        "permission_level": "approval",
        "desktop_only": True,
        "local_only": True,
    },
    {
        "type": "builtin",
        "name": "process.terminal",
        "description": "Run project terminal commands.",
        "permission_level": "approval",
        "desktop_only": True,
        "local_only": True,
    },
    {
        "type": "builtin",
        "name": "git",
        "description": "Inspect and mutate Git repositories.",
        "permission_level": "approval",
        "desktop_only": True,
        "local_only": True,
    },
    {
        "type": "builtin",
        "name": "device.control",
        "description": "Open previews and run Flutter apps on devices.",
        "permission_level": "approval",
        "desktop_only": True,
        "local_only": True,
    },
    {
        "type": "builtin",
        "name": "app_builder",
        "description": "Create local Next.js, React/Vite, or Flutter app workspaces.",
        "permission_level": "approval",
        "desktop_only": True,
        "local_only": True,
    },
)


# These four have no runtime behind them. Nothing executes a "spreadsheets"
# skill: selecting one only tags the task for approval policy. They stay in
# the catalog because `_select_capabilities` keyword matching and existing DB
# rows reference them, but they are reported as `unverified` — the honest
# status for "declared, never verified to run".
SKILL_UNVERIFIED_NOTE = "No runtime backs this skill — selecting it only tags the task for approval policy."

SKILL_PLACEHOLDERS: tuple[dict[str, Any], ...] = (
    {
        "type": "skill",
        "name": "documents",
        "status": "unverified",
        "description": f"Prepare document artifacts from task context. {SKILL_UNVERIFIED_NOTE}",
        "permission_level": "approval",
    },
    {
        "type": "skill",
        "name": "spreadsheets",
        "status": "unverified",
        "description": f"Analyze tabular data and create workbook artifacts. {SKILL_UNVERIFIED_NOTE}",
        "permission_level": "approval",
    },
    {
        "type": "skill",
        "name": "presentations",
        "status": "unverified",
        "description": f"Create presentation decks as task artifacts. {SKILL_UNVERIFIED_NOTE}",
        "permission_level": "approval",
    },
    {
        "type": "skill",
        "name": "browser",
        "status": "unverified",
        "description": f"Inspect and verify local or remote web UI. {SKILL_UNVERIFIED_NOTE}",
        "permission_level": "approval",
    },
)


# Source label written on every MCP catalog row, and the key the stale-row
# cleanup pass uses to recognise rows it owns.
MCP_CATALOG_SOURCE = "mcp_registry"

# Reserved: the `browser` mcp_server row means the server's own Playwright
# runtime (what a `browser_action` step actually runs on), not a configured
# MCP server. `_select_workflow_step_capabilities` looks it up by this name.
BROWSER_RUNTIME_CAPABILITY_NAME = "browser"


#: `mcp_id` values an agent draft may carry that name a runtime **this server
#: implements itself**. They are not MCP servers, will never appear in
#: `~/.claude.json`, and cannot be handed to the Claude SDK — checking them
#: against the detected config would report the server's own executors as
#: missing. `browser_action` steps run on the built-in Playwright adapter
#: (`browser_action_adapter.py`) and `app_action` steps on the built-in Android
#: adapter, regardless of what any CLI has configured. Whether the browser
#: runtime is *installed* is a separate question, answered by
#: `_browser_runtime_entry()` and the readiness chip — not by this map.
BUILTIN_TOOL_RUNTIMES: dict[str, str] = {
    "playwright": "the server's built-in Playwright browser runtime",
    BROWSER_RUNTIME_CAPABILITY_NAME: "the server's built-in Playwright browser runtime",
    "app_action": "the server's built-in Android (adb) runtime",
}


@dataclass(frozen=True)
class ToolVerification:
    """Whether one declared ``mcp_id`` names something that exists here.

    ``source`` is ``"mcp_config"`` (a server declared in a local MCP config),
    ``"builtin_runtime"`` (one of :data:`BUILTIN_TOOL_RUNTIMES`), or ``""``
    when nothing on this machine answers to the name. ``detail`` is the
    sentence shown to the user — it must say what was looked for and where,
    never just "unavailable".
    """

    mcp_id: str
    verified: bool
    source: str
    detail: str
    config_path: str | None = None


def _mcp_config_paths() -> list[tuple[Path, str]]:
    """Config files that declare MCP servers, least specific first."""
    paths: list[tuple[Path, str]] = []
    try:
        paths.append((Path.home() / ".claude.json", "claude_cli_config"))
    except Exception:  # pragma: no cover - no resolvable home directory
        pass
    try:
        paths.append((Path.cwd() / ".mcp.json", "project_mcp_config"))
    except Exception:  # pragma: no cover - cwd deleted underneath us
        pass
    return paths


def _read_mcp_servers(path: Path) -> dict[str, Any]:
    """Return the `mcpServers` mapping in ``path``, or ``{}``.

    Every failure mode — missing file, unreadable file, malformed JSON,
    unexpected shape — yields an empty mapping. Nothing is invented to fill
    the gap: an unreadable config means we do not know of any server, not
    that some default set exists.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("Cannot read MCP config %s: %s", path, exc)
        return {}

    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        logger.warning("Cannot parse MCP config %s: %s", path, exc)
        return {}

    if not isinstance(parsed, dict):
        return {}
    servers = parsed.get("mcpServers")
    if not isinstance(servers, dict):
        return {}
    return {
        name: config
        for name, config in servers.items()
        if isinstance(name, str) and name.strip()
    }


def _mcp_catalog_entry(
    name: str,
    config: Any,
    *,
    path: Path,
    origin: str,
) -> dict[str, Any]:
    settings = config if isinstance(config, dict) else {}
    command = settings.get("command")
    url = settings.get("url")
    args = settings.get("args")
    transport = settings.get("type")
    if not isinstance(transport, str) or not transport:
        transport = "http" if isinstance(url, str) and url else "stdio" if command else None
    return {
        "type": "mcp_server",
        "name": name,
        "status": "available",
        "description": f'MCP server "{name}" configured on this machine ({path.name}).',
        "permission_level": "approval",
        "metadata": {
            "command": command if isinstance(command, str) else None,
            "args": [str(arg) for arg in args] if isinstance(args, list) else [],
            "url": url if isinstance(url, str) else None,
            "transport": transport,
            "config_path": str(path),
            "config_origin": origin,
            "detected": True,
        },
    }


def _merged_mcp_servers() -> dict[str, tuple[Any, Path, str]]:
    """Every MCP server declared in a local config, most specific last.

    One read of the config files, shared by the catalog entries, the
    declaration check, and the SDK launch configs, so the three can never
    disagree about which servers exist.
    """
    merged: dict[str, tuple[Any, Path, str]] = {}
    for path, origin in _mcp_config_paths():
        for name, config in _read_mcp_servers(path).items():
            if name == BROWSER_RUNTIME_CAPABILITY_NAME:
                # The name is taken by the built-in runtime entry below; a
                # configured server of the same name would silently change
                # what a browser_action step believes it is running on.
                logger.info(
                    "Ignoring MCP server named %r in %s: the name is reserved "
                    "for the built-in browser runtime capability.",
                    name,
                    path,
                )
                continue
            merged[name] = (config, path, origin)
    return merged


def _detect_mcp_servers() -> list[dict[str, Any]]:
    """Catalog entries for MCP servers actually configured on this machine."""
    merged = _merged_mcp_servers()
    return [
        _mcp_catalog_entry(name, merged[name][0], path=merged[name][1], origin=merged[name][2])
        for name in sorted(merged)
    ]


def _sdk_mcp_server_config(config: Any) -> dict[str, Any] | None:
    """Translate one `mcpServers` entry into a Claude SDK server config.

    Returns ``None`` when the entry cannot be launched from what it declares —
    a stdio entry with no command, an http/sse entry with no url, an ``sdk``
    entry (an in-process server object that no JSON file can describe), or a
    transport this build does not know. Guessing a command here would hand the
    SDK a server that fails to spawn, which is the same fabricated promise one
    layer down.
    """
    if not isinstance(config, dict):
        return None

    declared = config.get("type")
    transport = declared if isinstance(declared, str) and declared.strip() else None
    command = config.get("command")
    url = config.get("url")
    if transport is None:
        if isinstance(url, str) and url.strip():
            transport = "http"
        elif isinstance(command, str) and command.strip():
            transport = "stdio"

    if transport == "stdio":
        if not isinstance(command, str) or not command.strip():
            return None
        entry: dict[str, Any] = {"type": "stdio", "command": command}
        args = config.get("args")
        if isinstance(args, list):
            entry["args"] = [str(arg) for arg in args]
        env = config.get("env")
        if isinstance(env, dict):
            entry["env"] = {str(key): str(value) for key, value in env.items()}
        return entry

    if transport in {"http", "sse"}:
        if not isinstance(url, str) or not url.strip():
            return None
        entry = {"type": transport, "url": url}
        headers = config.get("headers")
        if isinstance(headers, dict):
            entry["headers"] = {str(key): str(value) for key, value in headers.items()}
        return entry

    return None


def detected_mcp_server_configs() -> dict[str, dict[str, Any]]:
    """Launchable SDK configs for the MCP servers configured on this machine.

    Keyed by server name. A configured server whose entry cannot be turned
    into a launch config is omitted rather than approximated; callers report
    the omission instead of passing something that would fail to spawn.

    The values carry ``env`` and ``headers`` verbatim because the SDK needs
    them to start the server — so they may hold credentials and must never be
    written to a run event, a log line, or an API response.
    """
    configs: dict[str, dict[str, Any]] = {}
    for name, (config, _path, _origin) in _merged_mcp_servers().items():
        entry = _sdk_mcp_server_config(config)
        if entry is not None:
            configs[name] = entry
    return configs


def verify_declared_mcp_ids(mcp_ids: Iterable[str]) -> list[ToolVerification]:
    """Check declared tool ids against what actually exists on this machine.

    Order and duplicates of the input are preserved as first-seen order with
    duplicates collapsed, so a caller can report on each declaration once.
    Nothing here raises: every config read failure already degrades to "no
    servers known", and reporting a declaration as unverified is the honest
    answer to "we could not read the config" as much as to "it is not there".
    """
    detected = {str(entry["name"]): entry for entry in _detect_mcp_servers()}
    results: list[ToolVerification] = []
    seen: set[str] = set()
    for raw in mcp_ids:
        mcp_id = str(raw or "").strip()
        if not mcp_id or mcp_id in seen:
            continue
        seen.add(mcp_id)

        # A real configured server wins over the built-in alias of the same
        # name. `playwright` is both here: the built-in adapter is what a
        # `browser_action` step runs on no matter what, but if the user has
        # also configured a `playwright` MCP server then that server is a real
        # thing they declared and an llm turn should be able to reach it.
        # Resolving to the built-in first would make a configured server
        # permanently unreachable — the same drift C6 exists to close.
        entry = detected.get(mcp_id)
        if entry is None:
            builtin = BUILTIN_TOOL_RUNTIMES.get(mcp_id)
            if builtin is not None:
                results.append(
                    ToolVerification(
                        mcp_id=mcp_id,
                        verified=True,
                        source="builtin_runtime",
                        detail=f'"{mcp_id}" runs on {builtin}.',
                    )
                )
                continue
        if entry is not None:
            metadata = entry.get("metadata") or {}
            config_path = metadata.get("config_path")
            results.append(
                ToolVerification(
                    mcp_id=mcp_id,
                    verified=True,
                    source="mcp_config",
                    detail=f'MCP server "{mcp_id}" is configured in {config_path}.',
                    config_path=str(config_path) if config_path else None,
                )
            )
            continue

        looked_in = ", ".join(str(path) for path, _origin in _mcp_config_paths()) or "no MCP config file"
        results.append(
            ToolVerification(
                mcp_id=mcp_id,
                verified=False,
                source="",
                detail=(
                    f'No MCP server named "{mcp_id}" is configured on this machine '
                    f"(looked in {looked_in}), and it is not one of the runtimes this "
                    "server provides itself."
                ),
            )
        )
    return results


def _browser_runtime_entry() -> dict[str, Any]:
    """Catalog entry for the server's built-in Playwright browser runtime.

    Status comes from the readiness cache. A cache miss is reported as
    ``unknown`` — the probe starts a driver process, so this synchronous
    path cannot run it, and "not probed" is not "ready".
    """
    readiness = get_cached_browser_readiness_sync()
    if readiness is None:
        status = "unknown"
        ready: bool | None = None
        message = "Browser runtime readiness has not been probed yet."
        install_command = PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND
    else:
        ready = bool(readiness.get("ready"))
        status = "available" if ready else "unavailable"
        message = str(readiness.get("message") or "")
        install_command = str(
            readiness.get("install_command") or PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND
        )
    return {
        "type": "mcp_server",
        "name": BROWSER_RUNTIME_CAPABILITY_NAME,
        "status": status,
        "description": "Browser automation on the server's built-in Playwright runtime.",
        "permission_level": "approval",
        "metadata": {
            "runtime": "server_playwright",
            "ready": ready,
            "readiness_probed": readiness is not None,
            "install_command": install_command,
            "message": message,
            "detected": True,
        },
    }


def _demote_stale_mcp_rows(store: Any, live_names: set[str]) -> list[dict[str, Any]]:
    """Downgrade catalog rows for MCP servers that are no longer detected.

    Rows written by an earlier build (github, gmail, …) survive in the DB
    with ``status='available'`` long after the code stopped claiming them.
    They are downgraded rather than deleted so existing task links keep
    resolving — but they stop advertising availability.
    """
    demoted: list[dict[str, Any]] = []
    try:
        rows = store.list_capabilities(capability_type="mcp_server", limit=500)
    except Exception as exc:  # pragma: no cover - storage failure
        logger.warning("Cannot scan MCP catalog rows for stale entries: %s", exc)
        return demoted

    for row in rows:
        if str(row.get("source") or "") != MCP_CATALOG_SOURCE:
            continue
        name = str(row.get("name") or "")
        if not name or name in live_names:
            continue
        metadata = dict(row.get("metadata") or {})
        metadata["detected"] = False
        metadata["unverified_reason"] = "not_present_in_local_mcp_config"
        demoted.append(
            store.upsert_capability(
                capability_type="mcp_server",
                name=name,
                provider_id=row.get("provider_id"),
                status="unverified",
                scope=str(row.get("scope") or "global"),
                source=MCP_CATALOG_SOURCE,
                description=row.get("description"),
                permission_level=str(row.get("permission_level") or "approval"),
                desktop_only=bool(row.get("desktop_only", True)),
                local_only=bool(row.get("local_only", True)),
                metadata=metadata,
            )
        )
    return demoted


def refresh_capability_registry() -> list[dict[str, Any]]:
    """Refresh durable read-only catalog entries from built-ins and LLM status."""
    store = get_agent_store()
    capabilities: list[dict[str, Any]] = []
    for item in BUILTIN_CAPABILITIES:
        capabilities.append(_upsert_catalog_item(store, item, source="codebridge"))
    for item in SKILL_PLACEHOLDERS:
        capabilities.append(_upsert_catalog_item(store, item, source="skill_registry"))

    mcp_items = [*_detect_mcp_servers(), _browser_runtime_entry()]
    for item in mcp_items:
        capabilities.append(_upsert_catalog_item(store, item, source=MCP_CATALOG_SOURCE))
    capabilities.extend(
        _demote_stale_mcp_rows(store, {str(item["name"]) for item in mcp_items})
    )

    snapshot = get_llm_options_snapshot()
    for company in snapshot.get("companies", []):
        if not isinstance(company, dict):
            continue
        provider_id = company.get("id")
        if not isinstance(provider_id, str) or not provider_id:
            continue
        provider_status = "available" if company.get("connected") else "unavailable"
        capabilities.append(
            store.upsert_capability(
                capability_type="llm_cli",
                name=company.get("name") or provider_id,
                provider_id=provider_id,
                status=provider_status,
                scope="global",
                source="llm_settings",
                description=f"{company.get('name') or provider_id} local CLI provider",
                permission_level="approval",
                desktop_only=True,
                local_only=True,
                metadata={
                    "command": company.get("command"),
                    "models": company.get("models") or [],
                    "capabilities": company.get("capabilities") or {},
                    "install_methods": company.get("install_methods") or [],
                },
            )
        )
        raw_capabilities = company.get("capabilities")
        if isinstance(raw_capabilities, dict):
            for key, enabled in sorted(raw_capabilities.items()):
                if not isinstance(enabled, bool):
                    continue
                capabilities.append(
                    store.upsert_capability(
                        capability_type="native_tool",
                        name=str(key),
                        provider_id=provider_id,
                        status="available" if enabled and company.get("connected") else "unavailable",
                        scope="project",
                        source="llm_settings",
                        description=f"{provider_id} capability: {key}",
                        permission_level="approval" if key not in {"chat", "chat_supported"} else "read",
                        desktop_only=key in {"permissions", "live_permission_prompts", "sandbox_mode"},
                        local_only=True,
                        metadata={"enabled": enabled, "provider": provider_id},
                    )
                )
    return capabilities


def _upsert_catalog_item(store: Any, item: dict[str, Any], *, source: str) -> dict[str, Any]:
    return store.upsert_capability(
        capability_type=str(item["type"]),
        name=str(item["name"]),
        status=str(item.get("status") or "available"),
        scope=str(item.get("scope") or "global"),
        source=source,
        description=item.get("description"),
        permission_level=str(item.get("permission_level") or "approval"),
        desktop_only=bool(item.get("desktop_only", True)),
        local_only=bool(item.get("local_only", True)),
        metadata=dict(item.get("metadata") or {}),
    )
