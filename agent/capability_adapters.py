"""Capability adapter descriptors for task orchestration."""

from __future__ import annotations

from typing import Any


BUILTIN_POLICY_OPERATIONS = {
    "file.read": "file.read",
    "file.write": "file.write",
    "process.terminal": "process.terminal",
    "git": "git",
    "device.control": "device.control",
    "app_builder": "app_builder",
}


def describe_capability_adapter(capability: dict[str, Any]) -> dict[str, Any]:
    """Return the execution adapter descriptor for one catalog capability."""
    capability_type = str(capability.get("type") or "")
    name = str(capability.get("name") or "")
    if capability_type == "llm_cli":
        return {
            "adapter": "provider_session",
            "invocation": "chat_session",
            "policy_operation": "provider.turn",
            "provider_id": capability.get("provider_id"),
            "status": capability.get("status"),
        }
    if capability_type == "builtin":
        return {
            "adapter": "codebridge_builtin",
            "invocation": "server_route",
            "policy_operation": BUILTIN_POLICY_OPERATIONS.get(name, name),
            "status": capability.get("status"),
        }
    if capability_type == "skill":
        return {
            "adapter": "skill_registry",
            "invocation": "deferred_skill",
            "policy_operation": f"skill.{name}",
            "status": capability.get("status"),
        }
    if capability_type == "mcp_server":
        return {
            "adapter": "mcp_registry",
            "invocation": "deferred_mcp",
            "policy_operation": f"mcp.{name}",
            "status": capability.get("status"),
        }
    if capability_type == "native_tool":
        return {
            "adapter": "provider_native_tool",
            "invocation": "provider_permission",
            "policy_operation": f"provider.{name}",
            "provider_id": capability.get("provider_id"),
            "status": capability.get("status"),
        }
    return {
        "adapter": "unknown",
        "invocation": "manual",
        "policy_operation": name or capability_type or "unknown",
        "status": capability.get("status"),
    }
