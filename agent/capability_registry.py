"""Discovery and persistence for Work Cockpit capabilities."""

from __future__ import annotations

from typing import Any

from llm.llm_settings import get_llm_options_snapshot

from .agent_store import get_agent_store


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


SKILL_PLACEHOLDERS: tuple[dict[str, Any], ...] = (
    {
        "type": "skill",
        "name": "documents",
        "description": "Prepare document artifacts from task context.",
        "permission_level": "approval",
    },
    {
        "type": "skill",
        "name": "spreadsheets",
        "description": "Analyze tabular data and create workbook artifacts.",
        "permission_level": "approval",
    },
    {
        "type": "skill",
        "name": "presentations",
        "description": "Create presentation decks as task artifacts.",
        "permission_level": "approval",
    },
    {
        "type": "skill",
        "name": "browser",
        "description": "Inspect and verify local or remote web UI.",
        "permission_level": "approval",
    },
)


MCP_PLACEHOLDERS: tuple[dict[str, Any], ...] = (
    {
        "type": "mcp_server",
        "name": "github",
        "description": "GitHub repository, issue, and pull request access.",
        "permission_level": "approval",
    },
    {
        "type": "mcp_server",
        "name": "gmail",
        "description": "Gmail-backed task inputs and email workflow support.",
        "permission_level": "approval",
    },
    {
        "type": "mcp_server",
        "name": "browser",
        "description": "Browser automation as a task capability.",
        "permission_level": "approval",
    },
)


def refresh_capability_registry() -> list[dict[str, Any]]:
    """Refresh durable read-only catalog entries from built-ins and LLM status."""
    store = get_agent_store()
    capabilities: list[dict[str, Any]] = []
    for item in BUILTIN_CAPABILITIES:
        capabilities.append(_upsert_catalog_item(store, item, source="codebridge"))
    for item in SKILL_PLACEHOLDERS:
        capabilities.append(_upsert_catalog_item(store, item, source="skill_registry"))
    for item in MCP_PLACEHOLDERS:
        capabilities.append(_upsert_catalog_item(store, item, source="mcp_registry"))

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
