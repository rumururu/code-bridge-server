"""Slash command discovery for configured LLM CLIs.

The chat UI needs two different command classes:
1. Code Bridge client actions, such as opening the project picker tutorial.
2. Provider CLI slash commands discovered from the installed CLI help output.

Provider commands are intentionally marked non-executable for the current
adapters because chat sessions are run through non-interactive subprocess/API
transports. This avoids pretending a UI click can drive a provider TTY command.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from llm.llm_settings import PROVIDERS, get_llm_options_snapshot

COMMAND_CACHE_TTL_SECONDS = 300


@dataclass(frozen=True)
class HelpProbe:
    args: tuple[str, ...]


PROVIDER_HELP_PROBES: dict[str, list[HelpProbe]] = {
    "anthropic": [HelpProbe(("claude", "--help"))],
    "openai": [HelpProbe(("codex", "--help")), HelpProbe(("codex", "exec", "--help"))],
    "google": [HelpProbe(("gemini", "--help"))],
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SLASH_LINE_RE = re.compile(r"^(?:[-*]\s*)?(/[A-Za-z][A-Za-z0-9_-]*)(?:\s+(.+))?$")
_COMMAND_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def get_llm_command_snapshot(
    provider_id: str | None = None,
    model: str | None = None,
    scope: str = "project",
    refresh: bool = False,
) -> dict[str, Any]:
    """Return slash commands for the active provider and current UI scope."""

    normalized_scope = "global" if scope == "global" else "project"
    selected = get_llm_options_snapshot().get("selected") or {}
    resolved_provider_id = (provider_id or selected.get("company_id") or "").strip().lower()
    resolved_model = model or selected.get("model")

    if not resolved_provider_id:
        return _snapshot(
            provider_id=None,
            model=resolved_model,
            scope=normalized_scope,
            commands=_code_bridge_commands(normalized_scope, provider_id=None),
            provider_capabilities=_base_capabilities(False, "No LLM provider is selected."),
            source="code_bridge",
        )

    cache_key = f"{resolved_provider_id}:{normalized_scope}"
    now = time.monotonic()
    cached = _COMMAND_CACHE.get(cache_key)
    if not refresh and cached and now - cached[0] < COMMAND_CACHE_TTL_SECONDS:
        payload = dict(cached[1])
        payload["model"] = resolved_model
        return payload

    provider_commands, discovery = _discover_provider_commands(resolved_provider_id)
    commands = _code_bridge_commands(normalized_scope, provider_id=resolved_provider_id)
    commands.extend(_provider_code_bridge_commands(resolved_provider_id, normalized_scope))
    if normalized_scope == "project":
        code_bridge_names = {command.get("name") for command in commands}
        commands.extend(
            command for command in provider_commands if command.get("name") not in code_bridge_names
        )

    source = "mixed" if normalized_scope == "project" and provider_commands else "code_bridge"
    payload = _snapshot(
        provider_id=resolved_provider_id,
        model=resolved_model,
        scope=normalized_scope,
        commands=commands,
        provider_capabilities=discovery,
        source=source,
    )
    _COMMAND_CACHE[cache_key] = (now, payload)
    return payload


def execute_llm_command(
    *,
    name: str,
    provider_id: str | None = None,
    model: str | None = None,
    scope: str = "project",
    project_name: str | None = None,
) -> dict[str, Any]:
    """Execute a Code Bridge slash command and return a chat-displayable result."""

    normalized_name = _normalize_command_name(name)
    normalized_scope = "global" if scope == "global" else "project"
    snapshot = get_llm_command_snapshot(
        provider_id=provider_id,
        model=model,
        scope=normalized_scope,
    )
    command = next(
        (item for item in snapshot["commands"] if item.get("name") == normalized_name),
        None,
    )
    if command is None:
        return {
            "success": False,
            "command": normalized_name,
            "execution": "disabled",
            "message": f"Unknown command: {normalized_name}",
        }
    if not command.get("enabled", True):
        return {
            "success": False,
            "command": normalized_name,
            "execution": "disabled",
            "message": command.get("disabled_reason") or f"{normalized_name} is not available.",
        }
    if command.get("requires_project") and not project_name:
        return {
            "success": False,
            "command": normalized_name,
            "execution": "disabled",
            "message": f"{normalized_name} requires a selected project.",
            "disabled_reason": "Select a project before running this command.",
            "payload": {"client_action": "project_picker_tutorial"},
        }

    execution = command.get("execution")
    if execution == "client_action":
        payload: dict[str, Any] = {}
        if command.get("client_action") == "session_picker":
            payload = {
                "project_name": project_name,
                "provider_id": snapshot.get("provider_id"),
                "model": snapshot.get("model"),
                "client_action": command.get("client_action"),
            }
        return {
            "success": True,
            "command": normalized_name,
            "execution": "client_action",
            "message": command.get("description") or command.get("label") or normalized_name,
            "client_action": command.get("client_action"),
            "payload": payload,
        }

    if execution == "prompt_action":
        return {
            "success": True,
            "command": normalized_name,
            "execution": "prompt_action",
            "message": command.get("template") or normalized_name,
            "prompt": command.get("template") or normalized_name,
            "prompt_action": command.get("prompt_action") or "insert_template",
        }

    if execution == "server_action":
        return _execute_server_action(
            normalized_name,
            snapshot=snapshot,
            project_name=project_name,
        )

    return {
        "success": False,
        "command": normalized_name,
        "execution": "disabled",
        "message": f"{normalized_name} cannot be executed by Code Bridge yet.",
    }


def clear_llm_command_cache(provider_id: str | None = None) -> None:
    """Clear command discovery cache, optionally for a single provider."""

    if provider_id is None:
        _COMMAND_CACHE.clear()
        return
    normalized = provider_id.strip().lower()
    for key in list(_COMMAND_CACHE):
        if key.startswith(f"{normalized}:"):
            _COMMAND_CACHE.pop(key, None)


def _snapshot(
    *,
    provider_id: str | None,
    model: str | None,
    scope: str,
    commands: list[dict[str, Any]],
    provider_capabilities: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    return {
        "provider_id": provider_id,
        "model": model,
        "scope": scope,
        "source": source,
        "refreshed_at": datetime.now(UTC).isoformat(),
        "capabilities": provider_capabilities,
        "commands": commands,
    }


def _normalize_command_name(name: str) -> str:
    normalized = name.strip().split(maxsplit=1)[0] if isinstance(name, str) else ""
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized


def _command(
    *,
    name: str,
    label: str,
    description: str,
    execution: str,
    scope: str,
    template: str = "",
    client_action: str | None = None,
    server_action: str | None = None,
    prompt_action: str | None = None,
    requires_project: bool = False,
    requires_session: bool = False,
    enabled: bool = True,
    disabled_reason: str | None = None,
    provider_id: str | None = None,
    provider_notes: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": name.lstrip("/").replace("-", "_"),
        "name": name,
        "label": label,
        "description": description,
        "template": template,
        "source": "code_bridge",
        "scope": scope,
        "execution": execution,
        "client_action": client_action,
        "server_action": server_action,
        "prompt_action": prompt_action or ("insert_template" if execution == "prompt_action" else None),
        "requires_project": requires_project,
        "requires_session": requires_session,
        "enabled": enabled,
    }
    if disabled_reason:
        payload["disabled_reason"] = disabled_reason
    if provider_id:
        payload["provider_id"] = provider_id
    if provider_notes:
        payload["provider_notes"] = provider_notes
    return payload


def _base_capabilities(discoverable: bool, reason: str | None = None) -> dict[str, Any]:
    return {
        "slash_commands_discoverable": discoverable,
        "slash_commands_executable": False,
        "slash_commands_passthrough": False,
        "requires_tty": True,
        "session_persistent": False,
        "disabled_reason": reason
        or "Current provider CLI help does not expose executable slash commands for Code Bridge's headless adapters.",
    }


def _code_bridge_commands(scope: str, *, provider_id: str | None = None) -> list[dict[str, Any]]:
    if scope == "global":
        return [
            _command(
                name="/project",
                label="Select or add a project",
                description="Show the project selector walkthrough",
                template="Show me how to select or add a project from this screen.",
                scope="global",
                execution="client_action",
                client_action="project_picker_tutorial",
            ),
            _command(
                name="/model",
                label="Current model",
                description="Show selected provider and model",
                scope="global",
                execution="server_action",
                server_action="model_status",
            ),
            _command(
                name="/status",
                label="Status",
                description="Show server, provider, and project state",
                scope="global",
                execution="server_action",
                server_action="status",
            ),
            _command(
                name="/doctor",
                label="Doctor",
                description="Run connection and CLI diagnostics",
                scope="global",
                execution="server_action",
                server_action="doctor",
            ),
            _command(
                name="/clear",
                label="Clear chat",
                description="Clear the visible chat log",
                scope="global",
                execution="server_action",
                server_action="clear",
            ),
            _command(
                name="/help",
                label="Help",
                description="List available slash commands",
                scope="global",
                execution="server_action",
                server_action="help",
            ),
        ]

    return [
        _command(
            name="/resume",
            label="Resume session",
            description="Open session history and resume",
            scope="project",
            execution="client_action",
            client_action="session_picker",
            requires_project=True,
            enabled=provider_id != "google",
            disabled_reason=(
                "Gemini CLI exposes --resume/--list-sessions, but Code Bridge does not yet have a "
                "project-path-safe Gemini session adapter."
                if provider_id == "google"
                else None
            ),
        ),
        _command(
            name="/clear",
            label="Clear chat",
            description="Clear the visible chat log and start fresh",
            scope="project",
            execution="server_action",
            server_action="clear",
            requires_project=True,
        ),
        _command(
            name="/model",
            label="Current model",
            description="Show selected provider and model",
            scope="project",
            execution="server_action",
            server_action="model_status",
        ),
        _command(
            name="/status",
            label="Status",
            description="Show server, provider, project, and session state",
            scope="project",
            execution="server_action",
            server_action="status",
        ),
        _command(
            name="/doctor",
            label="Doctor",
            description="Run connection and CLI diagnostics",
            scope="project",
            execution="server_action",
            server_action="doctor",
        ),
        _command(
            name="/permissions",
            label="Permissions",
            description="Show permission support and mode",
            scope="project",
            execution="server_action",
            server_action="permissions",
        ),
        _command(
            name="/compact",
            label="Compact",
            description="Summarize current work before continuing",
            template="Summarize this conversation into a compact handoff: current goal, decisions, changed files, verification, and next steps.",
            scope="project",
            execution="prompt_action",
            requires_project=True,
        ),
        _command(
            name="/fix",
            label="Diagnose and fix",
            description="Inspect, patch, and verify",
            template="Fix the current issue. First identify the root cause, then make the smallest scoped code change, and run the relevant verification.",
            scope="project",
            execution="prompt_action",
            requires_project=True,
        ),
        _command(
            name="/test",
            label="Run verification",
            description="Run relevant checks",
            template="Run the relevant tests or checks. If something fails, diagnose it and fix only the related issue.",
            scope="project",
            execution="prompt_action",
            requires_project=True,
        ),
        _command(
            name="/explain",
            label="Explain code",
            description="Explain files and flow",
            template="Explain how this part of the code works, with file references and the key control flow.",
            scope="project",
            execution="prompt_action",
            requires_project=True,
        ),
        _command(
            name="/commit",
            label="Prepare commit",
            description="Review diff and message",
            template="Review the current git diff, summarize the change, and prepare a focused commit message. Do not commit until I approve.",
            scope="project",
            execution="prompt_action",
            requires_project=True,
        ),
        _command(
            name="/help",
            label="Help",
            description="List available slash commands",
            scope="project",
            execution="server_action",
            server_action="help",
        ),
    ]


def _openai_code_bridge_commands(scope: str) -> list[dict[str, Any]]:
    global_commands = [
        _command(
            name="/features",
            label="Codex features",
            description="Show Codex feature flag support and how Code Bridge handles it",
            scope=scope,
            execution="server_action",
            server_action="codex_features",
            provider_id="openai",
            provider_notes="Maps to the `codex features` command family. Code Bridge reports support without editing config.",
        ),
        _command(
            name="/mcp",
            label="Codex MCP",
            description="Show Codex MCP integration status and safe next actions",
            scope=scope,
            execution="server_action",
            server_action="codex_mcp",
            provider_id="openai",
            provider_notes="Maps to `codex mcp`. Read-only guidance only; add/remove/login are not run from chat.",
        ),
        _command(
            name="/plugins",
            label="Codex plugins",
            description="Show Codex plugin marketplace support and safe next actions",
            scope=scope,
            execution="server_action",
            server_action="codex_plugins",
            provider_id="openai",
            provider_notes="Maps to `codex plugin marketplace`. Chat command does not install or remove plugins.",
        ),
    ]
    if scope == "global":
        return global_commands

    return [
        _command(
            name="/review",
            label="Codex review",
            description="Review current repository changes",
            template=(
                "Review the current repository like Codex review would. Focus on bugs, regressions, "
                "security issues, and missing tests. Lead with findings and include file references."
            ),
            scope="project",
            execution="prompt_action",
            requires_project=True,
            provider_id="openai",
            provider_notes="Maps to `codex review` intent through the existing chat adapter instead of spawning a second Codex run.",
        ),
        _command(
            name="/fork",
            label="Codex fork",
            description="Fork a previous Codex session",
            scope="project",
            execution="unsupported",
            requires_project=True,
            requires_session=True,
            enabled=False,
            disabled_reason=(
                "Codex exposes `codex fork`, but Code Bridge does not yet have a fork-session picker "
                "or safe adapter path for creating a visible fork from chat."
            ),
            provider_id="openai",
        ),
        _command(
            name="/search",
            label="Codex web search",
            description="Show Codex web-search launch support",
            scope="project",
            execution="server_action",
            server_action="codex_search",
            requires_project=True,
            provider_id="openai",
            provider_notes="Maps to Codex `--search`; currently reported as launch-time capability, not toggled mid-session.",
        ),
        _command(
            name="/sandbox",
            label="Codex sandbox",
            description="Show current Codex sandbox mode",
            scope="project",
            execution="server_action",
            server_action="permissions",
            requires_project=True,
            provider_id="openai",
            provider_notes="Alias for Code Bridge permission/sandbox status. Maps to Codex `-s/--sandbox`.",
        ),
        _command(
            name="/approval",
            label="Codex approval",
            description="Show Codex approval behavior",
            scope="project",
            execution="server_action",
            server_action="permissions",
            requires_project=True,
            provider_id="openai",
            provider_notes="Alias for Code Bridge permission/sandbox status. Maps to Codex `-a/--ask-for-approval` conceptually.",
        ),
        _command(
            name="/apply",
            label="Codex apply",
            description="Apply a Codex Cloud task diff",
            scope="project",
            execution="unsupported",
            requires_project=True,
            enabled=False,
            disabled_reason=(
                "`codex apply` and `codex cloud apply` modify the working tree from a task id. "
                "Code Bridge does not run that destructive action from chat without a dedicated review/confirmation UI."
            ),
            provider_id="openai",
        ),
        _command(
            name="/cloud",
            label="Codex Cloud",
            description="Browse or apply Codex Cloud tasks",
            scope="project",
            execution="unsupported",
            requires_project=True,
            enabled=False,
            disabled_reason=(
                "Codex Cloud commands require task browsing, auth state, and diff review UI that Code Bridge "
                "does not expose in the chat command layer yet."
            ),
            provider_id="openai",
        ),
        *global_commands,
    ]


def _provider_code_bridge_commands(provider_id: str, scope: str) -> list[dict[str, Any]]:
    if provider_id == "openai":
        return _openai_code_bridge_commands(scope)
    if provider_id == "anthropic":
        return _anthropic_code_bridge_commands(scope)
    if provider_id == "google":
        return _google_code_bridge_commands(scope)
    return []


def _google_code_bridge_commands(scope: str) -> list[dict[str, Any]]:
    read_only_commands = [
        _command(
            name="/mcp",
            label="Gemini MCP",
            description="List Gemini CLI MCP servers",
            scope=scope,
            execution="server_action",
            server_action="gemini_mcp_list",
            provider_id="google",
            provider_notes="Maps to `gemini mcp list`. Add/remove/enable/disable are not run from chat.",
        ),
        _command(
            name="/skills",
            label="Gemini skills",
            description="List Gemini CLI skills",
            scope=scope,
            execution="server_action",
            server_action="gemini_skills_list",
            provider_id="google",
            provider_notes="Maps to `gemini skills list`. Install/link/enable/disable are not run from chat.",
        ),
        _command(
            name="/extensions",
            label="Gemini extensions",
            description="List Gemini CLI extensions",
            scope=scope,
            execution="server_action",
            server_action="gemini_extensions_list",
            provider_id="google",
            provider_notes="Maps to `gemini extensions list`. Install/update/config changes require a dedicated flow.",
        ),
        _command(
            name="/gemma",
            label="Gemma status",
            description="Show Gemini local Gemma routing status",
            scope=scope,
            execution="server_action",
            server_action="gemini_gemma_status",
            provider_id="google",
            provider_notes="Maps to `gemini gemma status`. Setup/start/stop/logs are not run from chat.",
        ),
        _command(
            name="/approval",
            label="Gemini approval",
            description="Show Gemini approval mode support",
            scope=scope,
            execution="server_action",
            server_action="gemini_approval",
            provider_id="google",
            provider_notes="Maps to `--approval-mode`; Code Bridge reports support but does not mutate a live session.",
        ),
        _command(
            name="/policy",
            label="Gemini policy",
            description="Show Gemini policy flag support",
            scope=scope,
            execution="server_action",
            server_action="gemini_policy",
            provider_id="google",
            provider_notes="Maps to `--policy` and `--admin-policy`; chat command is read-only guidance.",
        ),
    ]

    unsupported = [
        _command(
            name="/acp",
            label="Gemini ACP",
            description="Start Gemini in ACP mode",
            scope=scope,
            execution="unsupported",
            enabled=False,
            disabled_reason="Gemini ACP is a launch protocol, not a live chat command.",
            provider_id="google",
        ),
        _command(
            name="/raw-output",
            label="Gemini raw output",
            description="Disable Gemini output sanitization",
            scope=scope,
            execution="unsupported",
            enabled=False,
            disabled_reason=(
                "Gemini --raw-output changes output sanitization and carries explicit security risk; "
                "Code Bridge does not toggle it from chat."
            ),
            provider_id="google",
        ),
        _command(
            name="/debug",
            label="Gemini debug",
            description="Open Gemini CLI debug console",
            scope=scope,
            execution="unsupported",
            enabled=False,
            disabled_reason="Gemini debug mode is an interactive CLI launch option.",
            provider_id="google",
        ),
    ]

    if scope == "global":
        return [*read_only_commands, *unsupported]

    return [
        _command(
            name="/sessions",
            label="Gemini sessions",
            description="List or resume Gemini native sessions",
            scope="project",
            execution="unsupported",
            requires_project=True,
            enabled=False,
            disabled_reason=(
                "Gemini exposes --list-sessions/--resume/--delete-session, but Code Bridge does not "
                "yet pass the selected project path into this command executor."
            ),
            provider_id="google",
        ),
        _command(
            name="/worktree",
            label="Gemini worktree",
            description="Start Gemini in a new git worktree",
            scope="project",
            execution="unsupported",
            requires_project=True,
            enabled=False,
            disabled_reason=(
                "Gemini --worktree changes session launch topology. Code Bridge needs a separate "
                "project/session creation flow before exposing it."
            ),
            provider_id="google",
        ),
        _command(
            name="/include-directories",
            label="Gemini include directories",
            description="Include additional directories in a Gemini workspace",
            scope="project",
            execution="unsupported",
            requires_project=True,
            enabled=False,
            disabled_reason=(
                "Gemini --include-directories must be applied when launching a provider session; live "
                "mutation is not supported by the current adapter."
            ),
            provider_id="google",
        ),
        _command(
            name="/hooks",
            label="Gemini hooks",
            description="Manage or migrate Gemini hooks",
            scope="project",
            execution="unsupported",
            enabled=False,
            disabled_reason=(
                "Gemini hooks currently expose a migration command that can modify local configuration, "
                "so Code Bridge does not run it from chat."
            ),
            provider_id="google",
        ),
        *read_only_commands,
        *unsupported,
    ]


def _anthropic_code_bridge_commands(scope: str) -> list[dict[str, Any]]:

    if scope == "global":
        return [
            _command(
                name="/auth",
                label="Claude auth",
                description="Show Claude authentication status",
                scope="global",
                execution="server_action",
                server_action="claude_auth",
                provider_id="anthropic",
            ),
            _command(
                name="/plugins",
                label="Claude plugins",
                description="List installed Claude Code plugins",
                scope="global",
                execution="server_action",
                server_action="claude_plugins",
                provider_id="anthropic",
            ),
        ]

    return [
        _command(
            name="/continue",
            label="Continue latest Claude session",
            description="Open session history to continue recent Claude work",
            scope="project",
            execution="client_action",
            client_action="session_picker",
            requires_project=True,
            provider_id="anthropic",
        ),
        _command(
            name="/agents",
            label="Claude agents",
            description="List configured Claude Code agents",
            scope="project",
            execution="server_action",
            server_action="claude_agents",
            provider_id="anthropic",
        ),
        _command(
            name="/auth",
            label="Claude auth",
            description="Show Claude authentication status",
            scope="project",
            execution="server_action",
            server_action="claude_auth",
            provider_id="anthropic",
        ),
        _command(
            name="/plugins",
            label="Claude plugins",
            description="List installed Claude Code plugins",
            scope="project",
            execution="server_action",
            server_action="claude_plugins",
            provider_id="anthropic",
        ),
        _command(
            name="/auto-mode",
            label="Claude auto mode",
            description="Show Claude auto mode classifier configuration",
            scope="project",
            execution="server_action",
            server_action="claude_auto_mode",
            provider_id="anthropic",
        ),
        _command(
            name="/mcp",
            label="Claude MCP",
            description="Claude MCP listing is not run from chat because it may spawn project MCP servers",
            scope="project",
            execution="unsupported",
            enabled=False,
            provider_id="anthropic",
            disabled_reason=(
                "Claude `mcp list/get` can spawn project MCP stdio servers. "
                "Run it from a trusted terminal, or use Code Bridge MCP setup screens when available."
            ),
        ),
        _command(
            name="/from-pr",
            label="Resume from PR",
            description="Claude can resume sessions linked to pull requests, but Code Bridge has no PR picker yet",
            scope="project",
            execution="unsupported",
            requires_project=True,
            enabled=False,
            provider_id="anthropic",
            disabled_reason="Claude `--from-pr` needs a PR selector/URL flow that is not implemented in Code Bridge yet.",
        ),
        _command(
            name="/fork",
            label="Fork session",
            description="Fork a resumed Claude session into a new session id",
            scope="project",
            execution="unsupported",
            requires_project=True,
            requires_session=True,
            enabled=False,
            provider_id="anthropic",
            disabled_reason="Claude `--fork-session` requires resume-flow support for forking before the next turn.",
        ),
        _command(
            name="/name",
            label="Name session",
            description="Set Claude session display name",
            scope="project",
            execution="unsupported",
            requires_project=True,
            enabled=False,
            provider_id="anthropic",
            disabled_reason="Claude `--name` is a launch-time session option; Code Bridge has no session rename action yet.",
        ),
        _command(
            name="/tools",
            label="Claude tools",
            description="Show tool allow/deny behavior through Code Bridge permissions",
            scope="project",
            execution="server_action",
            server_action="permissions",
            provider_id="anthropic",
        ),
        _command(
            name="/skills",
            label="Claude skills",
            description="Dynamic Claude skill slash commands are not enumerated by headless help",
            scope="project",
            execution="unsupported",
            enabled=False,
            provider_id="anthropic",
            disabled_reason=(
                "Claude supports dynamic `/skill-name` entries in the interactive CLI, "
                "but this installed CLI does not expose a headless skills catalog to Code Bridge."
            ),
        ),
    ]


def _execute_server_action(
    command_name: str,
    *,
    snapshot: dict[str, Any],
    project_name: str | None,
) -> dict[str, Any]:
    if command_name == "/help":
        commands: list[dict[str, Any]] = []
        lines = ["Available slash commands:"]
        for command in snapshot.get("commands", []):
            if command.get("source") != "code_bridge":
                continue
            disabled = "" if command.get("enabled", True) else " (disabled)"
            lines.append(f"- `{command.get('name')}`{disabled}: {command.get('description')}")
            commands.append(_command_help_payload(command))
        return _command_result(
            command_name,
            "\n".join(lines),
            server_action="help",
            payload={
                "commands": commands,
                "capabilities": snapshot.get("capabilities") or {},
            },
        )

    if command_name == "/model":
        return _command_result(
            command_name,
            _format_model_status(snapshot),
            server_action="model_status",
            payload=_model_status_payload(snapshot),
        )

    if command_name == "/status":
        return _command_result(
            command_name,
            _format_status(snapshot, project_name=project_name),
            server_action="status",
            payload=_status_payload(snapshot, project_name=project_name),
        )

    if command_name == "/doctor":
        return _command_result(
            command_name,
            _format_doctor(snapshot, project_name=project_name),
            server_action="doctor",
            payload={"checks": _doctor_checks(snapshot, project_name=project_name)},
        )

    if command_name == "/mcp" and snapshot.get("provider_id") == "google":
        return _gemini_readonly_result(
            command_name,
            ("gemini", "mcp", "list"),
            server_action="gemini_mcp_list",
            title="Gemini MCP servers",
            empty_message="No Gemini MCP servers were reported.",
        )

    if command_name == "/skills" and snapshot.get("provider_id") == "google":
        return _gemini_readonly_result(
            command_name,
            ("gemini", "skills", "list"),
            server_action="gemini_skills_list",
            title="Gemini skills",
            empty_message="No Gemini skills were reported.",
        )

    if command_name == "/extensions" and snapshot.get("provider_id") == "google":
        return _gemini_readonly_result(
            command_name,
            ("gemini", "extensions", "list"),
            server_action="gemini_extensions_list",
            title="Gemini extensions",
            empty_message="No Gemini extensions were reported.",
        )

    if command_name == "/gemma" and snapshot.get("provider_id") == "google":
        return _gemini_readonly_result(
            command_name,
            ("gemini", "gemma", "status"),
            server_action="gemini_gemma_status",
            title="Gemma local routing status",
            empty_message="Gemini did not return Gemma status output.",
            allow_nonzero=True,
        )

    if command_name == "/approval" and snapshot.get("provider_id") == "google":
        return _command_result(
            command_name,
            (
                "Gemini approval modes:\n"
                "- `default`: prompt for approval\n"
                "- `auto_edit`: auto-approve edit tools\n"
                "- `yolo`: auto-approve all tools\n"
                "- `plan`: read-only mode\n\n"
                "Code Bridge reports this launch-time support here. It does not mutate an active "
                "Gemini session from chat."
            ),
            server_action="gemini_approval",
            payload={
                "provider_id": "google",
                "modes": ["default", "auto_edit", "yolo", "plan"],
                "live_mutation_supported": False,
            },
        )

    if command_name == "/policy" and snapshot.get("provider_id") == "google":
        return _command_result(
            command_name,
            (
                "Gemini policy support:\n"
                "- `--policy`: additional policy files or directories\n"
                "- `--admin-policy`: additional admin policy files or directories\n\n"
                "These are launch-time inputs. Code Bridge does not add or remove policy files from chat."
            ),
            server_action="gemini_policy",
            payload={
                "provider_id": "google",
                "flags": ["--policy", "--admin-policy"],
                "live_mutation_supported": False,
            },
        )

    if command_name in {"/permissions", "/tools", "/sandbox"} or (
        command_name == "/approval" and snapshot.get("provider_id") != "google"
    ):
        return _command_result(
            command_name,
            _format_permissions(snapshot),
            server_action="permissions",
            payload=_permissions_payload(snapshot),
        )

    if command_name == "/search":
        return _command_result(
            command_name,
            _format_codex_search(snapshot),
            server_action="codex_search",
            payload={
                "provider_id": snapshot.get("provider_id"),
                "launch_flag": "--search",
                "toggle_supported": False,
            },
        )

    if command_name == "/features":
        return _command_result(
            command_name,
            _format_codex_features(snapshot),
            server_action="codex_features",
            payload={
                "provider_id": snapshot.get("provider_id"),
                "cli_command": "codex features",
                "mutates_config": False,
            },
        )

    if command_name == "/mcp" and snapshot.get("provider_id") == "openai":
        return _command_result(
            command_name,
            _format_codex_mcp(snapshot),
            server_action="codex_mcp",
            payload={
                "provider_id": snapshot.get("provider_id"),
                "cli_command": "codex mcp",
                "mutates_config": False,
            },
        )

    if command_name == "/agents":
        return _claude_readonly_result(
            command_name,
            ("claude", "agents"),
            server_action="claude_agents",
            title="Claude agents",
        )

    if command_name == "/auth":
        return _claude_readonly_result(
            command_name,
            ("claude", "auth", "status", "--text"),
            server_action="claude_auth",
            title="Claude authentication",
        )

    if command_name == "/plugins" and snapshot.get("provider_id") == "openai":
        return _command_result(
            command_name,
            _format_codex_plugins(snapshot),
            server_action="codex_plugins",
            payload={
                "provider_id": snapshot.get("provider_id"),
                "cli_command": "codex plugin marketplace",
                "mutates_config": False,
            },
        )

    if command_name == "/plugins":
        return _claude_readonly_result(
            command_name,
            ("claude", "plugin", "list"),
            server_action="claude_plugins",
            title="Claude plugins",
        )

    if command_name == "/auto-mode":
        return _claude_readonly_result(
            command_name,
            ("claude", "auto-mode", "config"),
            server_action="claude_auto_mode",
            title="Claude auto mode",
        )

    if command_name == "/clear":
        return _command_result(
            command_name,
            "Cleared the current chat view. A new provider turn will start fresh.",
            server_action="clear",
            client_action="clear_chat",
            payload={
                "client_actions": [{"type": "clear_chat"}],
                "project_name": project_name,
            },
        )

    return {
        "success": False,
        "command": command_name,
        "execution": "disabled",
        "message": f"No server action is implemented for {command_name}.",
    }


def _command_result(
    command_name: str,
    message: str,
    *,
    server_action: str,
    client_action: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "success": True,
        "command": command_name,
        "execution": "server_action",
        "server_action": server_action,
        "message": message,
        "display": "markdown",
        "payload": payload or {},
    }
    if client_action:
        result["client_action"] = client_action
    return result


def _command_help_payload(command: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": command.get("id"),
        "name": command.get("name"),
        "label": command.get("label"),
        "description": command.get("description"),
        "execution": command.get("execution"),
        "client_action": command.get("client_action"),
        "server_action": command.get("server_action"),
        "prompt_action": command.get("prompt_action"),
        "enabled": command.get("enabled", True),
        "disabled_reason": command.get("disabled_reason"),
    }


def _selected_company(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    options = get_llm_options_snapshot()
    provider_id = snapshot.get("provider_id")
    for company in options.get("companies", []):
        if company.get("id") == provider_id:
            return company
    return None


def _format_model_status(snapshot: dict[str, Any]) -> str:
    payload = _model_status_payload(snapshot)
    provider = payload["provider"].get("name") or payload["provider"].get("id")
    command = payload["provider"].get("command")
    model = payload.get("model") or "not selected"
    connected = "connected" if payload["provider"].get("connected") else "not connected"
    lines = [
        f"Provider: {provider or 'not selected'}",
        f"Model: {model}",
        f"CLI: {command or 'unknown'} ({connected})",
    ]
    return "\n".join(lines)


def _model_status_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    company = _selected_company(snapshot)
    return {
        "provider": {
            "id": snapshot.get("provider_id"),
            "name": company.get("name") if company else snapshot.get("provider_id"),
            "command": company.get("command") if company else None,
            "connected": bool(company and company.get("connected")),
            "selectable": bool(company and company.get("selectable")),
            "status_message": company.get("status_message") if company else None,
        },
        "model": snapshot.get("model"),
        "native_slash_passthrough": bool(
            (snapshot.get("capabilities") or {}).get("slash_commands_passthrough")
        ),
    }


def _format_status(snapshot: dict[str, Any], *, project_name: str | None) -> str:
    scope = snapshot.get("scope") or "project"
    project = project_name if project_name else ("not selected" if scope == "global" else "unknown")
    capabilities = snapshot.get("capabilities") or {}
    lines = [
        "Code Bridge status:",
        f"- Scope: {scope}",
        f"- Project: {project}",
        f"- Provider: {snapshot.get('provider_id') or 'not selected'}",
        f"- Model: {snapshot.get('model') or 'not selected'}",
        f"- Native CLI slash passthrough: {'yes' if capabilities.get('slash_commands_passthrough') else 'no'}",
        f"- Code Bridge slash commands: {len([c for c in snapshot.get('commands', []) if c.get('source') == 'code_bridge'])}",
    ]
    reason = capabilities.get("disabled_reason")
    if reason:
        lines.append(f"- Native slash note: {reason}")
    return "\n".join(lines)


def _status_payload(snapshot: dict[str, Any], *, project_name: str | None) -> dict[str, Any]:
    capabilities = snapshot.get("capabilities") or {}
    return {
        "scope": snapshot.get("scope") or "project",
        "project": {
            "name": project_name,
            "selected": bool(project_name),
        },
        "provider": _model_status_payload(snapshot)["provider"],
        "model": snapshot.get("model"),
        "capabilities": {
            "native_slash_passthrough": bool(capabilities.get("slash_commands_passthrough")),
            "native_slash_executable": bool(capabilities.get("slash_commands_executable")),
            "requires_tty": bool(capabilities.get("requires_tty")),
            "disabled_reason": capabilities.get("disabled_reason"),
        },
        "command_count": len(snapshot.get("commands", [])),
    }


def _format_doctor(snapshot: dict[str, Any], *, project_name: str | None) -> str:
    checks = _doctor_checks(snapshot, project_name=project_name)
    lines = ["Doctor checks:"]
    for check in checks:
        ok = check["ok"]
        label = check["label"]
        detail = check["detail"]
        mark = "OK" if ok else "WARN"
        lines.append(f"- {mark}: {label} - {detail}")
    return "\n".join(lines)


def _doctor_checks(snapshot: dict[str, Any], *, project_name: str | None) -> list[dict[str, Any]]:
    company = _selected_company(snapshot)
    provider_ok = bool(company and company.get("connected") and company.get("selectable"))
    project_ok = bool(project_name) or snapshot.get("scope") == "global"
    raw_checks = [
        ("server_command_registry", "Server command registry", True, "responding", "error"),
        ("project_selection", "Project selection", project_ok, project_name or "not selected", "warning"),
        (
            "provider_cli",
            "Provider CLI",
            provider_ok,
            (company or {}).get("status_message") or (company or {}).get("command") or "not selected",
            "warning",
        ),
        ("model_selection", "Model selection", bool(snapshot.get("model")), snapshot.get("model") or "not selected", "warning"),
        (
            "native_slash_passthrough",
            "Native slash passthrough",
            False,
            "not exposed by current headless adapters",
            "info",
        ),
    ]
    return [
        {
            "name": name,
            "label": label,
            "ok": ok,
            "detail": detail,
            "severity": "info" if ok else severity,
        }
        for name, label, ok, detail, severity in raw_checks
    ]


def _format_permissions(snapshot: dict[str, Any]) -> str:
    provider_id = snapshot.get("provider_id")
    if provider_id == "anthropic":
        return (
            "Permissions: Claude supports live permission prompts in Code Bridge. "
            "When a tool needs approval, the request appears above the input box."
        )
    if provider_id == "openai":
        payload = _permissions_payload(snapshot)
        sandbox = payload.get("sandbox_mode") or "unknown"
        approval = payload.get("approval_policy") or "not exposed"
        return (
            "Permissions: Codex runs through exec mode.\n"
            f"- Sandbox mode: {sandbox}\n"
            f"- Approval policy: {approval}\n"
            "- Live approve/deny prompts are not exposed by the current Codex exec adapter."
        )
    if provider_id == "google":
        return "Permissions: Gemini headless mode does not expose live approval prompts."
    return "Permissions: Select a provider to see permission behavior."


def _permissions_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    company = _selected_company(snapshot) or {}
    settings = company.get("settings") if isinstance(company.get("settings"), dict) else {}
    return {
        "provider_id": snapshot.get("provider_id"),
        "sandbox_mode": settings.get("sandbox_mode"),
        "sandbox_modes": settings.get("sandbox_modes") or [],
        "approval_policy": "not exposed by Code Bridge Codex settings",
        "live_permission_prompts": snapshot.get("provider_id") == "anthropic",
    }


def _format_codex_search(snapshot: dict[str, Any]) -> str:
    if snapshot.get("provider_id") != "openai":
        return "Codex web search is available only when Codex is the selected provider."
    return (
        "Codex web search:\n"
        "- CLI surface: `codex --search` / `codex resume --search`.\n"
        "- Code Bridge status: launch-time capability recognized, but this chat command does not toggle an active session.\n"
        "- Use model/provider settings once Code Bridge exposes a web-search toggle."
    )


def _format_codex_features(snapshot: dict[str, Any]) -> str:
    if snapshot.get("provider_id") != "openai":
        return "Codex feature flags are available only when Codex is the selected provider."
    return (
        "Codex feature flags:\n"
        "- CLI surface: `codex features list|enable|disable`.\n"
        "- Code Bridge status: read-only guidance from chat. Enable/disable is not run here because it edits Codex config."
    )


def _format_codex_mcp(snapshot: dict[str, Any]) -> str:
    if snapshot.get("provider_id") != "openai":
        return "Codex MCP settings are available only when Codex is the selected provider."
    return (
        "Codex MCP:\n"
        "- CLI surface: `codex mcp list|get|add|remove|login|logout`.\n"
        "- Code Bridge status: recognized as a Codex-specific capability. Mutating MCP config from chat is disabled until a review UI exists."
    )


def _format_codex_plugins(snapshot: dict[str, Any]) -> str:
    if snapshot.get("provider_id") != "openai":
        return "Codex plugin settings are available only when Codex is the selected provider."
    return (
        "Codex plugins:\n"
        "- CLI surface: `codex plugin marketplace`.\n"
        "- Code Bridge status: recognized as a Codex-specific capability. Installing/removing plugins from chat is disabled until a review UI exists."
    )


def _claude_readonly_result(
    command_name: str,
    args: tuple[str, ...],
    *,
    server_action: str,
    title: str,
) -> dict[str, Any]:
    if shutil.which("claude") is None:
        return {
            "success": False,
            "command": command_name,
            "execution": "disabled",
            "message": "Claude CLI is not installed or not on PATH.",
            "disabled_reason": "Install Claude Code CLI before running this command.",
        }

    output, error, returncode = _run_readonly_cli(args)
    if error:
        return {
            "success": False,
            "command": command_name,
            "execution": "server_action",
            "server_action": server_action,
            "message": f"{title} failed: {error}",
            "payload": {"returncode": returncode, "args": list(args)},
        }

    body = output.strip() or "No output."
    return _command_result(
        command_name,
        f"{title}:\n\n```text\n{body}\n```",
        server_action=server_action,
        payload={"output": body, "args": list(args), "returncode": returncode},
    )


def _gemini_readonly_result(
    command_name: str,
    args: tuple[str, ...],
    *,
    server_action: str,
    title: str,
    empty_message: str,
    allow_nonzero: bool = False,
) -> dict[str, Any]:
    if shutil.which("gemini") is None:
        return {
            "success": False,
            "command": command_name,
            "execution": "disabled",
            "message": "Gemini CLI is not installed or not on PATH.",
            "disabled_reason": "Install Gemini CLI before running this command.",
        }

    output, error, returncode = _run_readonly_cli(args)
    if error and not allow_nonzero:
        return {
            "success": False,
            "command": command_name,
            "execution": "server_action",
            "server_action": server_action,
            "message": f"{title} failed: {error}",
            "payload": {"returncode": returncode, "args": list(args)},
        }

    body = output.strip() or empty_message
    if error and allow_nonzero:
        body = output.strip() or error
    return _command_result(
        command_name,
        f"{title}:\n\n```text\n{body}\n```",
        server_action=server_action,
        payload={"output": body, "args": list(args), "returncode": returncode},
    )


def _run_readonly_cli(args: tuple[str, ...]) -> tuple[str, str | None, int | None]:
    try:
        completed = subprocess.run(
            list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=6,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", f"{' '.join(args)} timed out", None
    except OSError as exc:
        return "", str(exc), None

    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    if completed.returncode != 0:
        return output, output or f"{' '.join(args)} exited with {completed.returncode}", completed.returncode
    return output, None, completed.returncode


def _discover_provider_commands(provider_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = next((item for item in PROVIDERS if item.id == provider_id), None)
    if provider is None:
        return [], _base_capabilities(False, f"Unknown LLM provider: {provider_id}")

    if shutil.which(provider.command) is None:
        return [], _base_capabilities(False, f"{provider.command} CLI is not installed or not on PATH.")

    probes = PROVIDER_HELP_PROBES.get(provider_id, [HelpProbe((provider.command, "--help"))])
    parsed: dict[str, str] = {}
    probe_errors: list[str] = []

    for probe in probes:
        output, error = _run_help_probe(probe.args)
        if error:
            probe_errors.append(error)
            continue
        for name, description in _parse_slash_commands(output):
            parsed.setdefault(name, description)

    commands = [
        {
            "id": f"{provider_id}.{name.lstrip('/')}",
            "name": name,
            "label": name[1:].replace("-", " ").title(),
            "description": description or "Provider CLI slash command",
            "template": name,
            "source": "cli",
            "provider_id": provider_id,
            "scope": "project",
            "execution": "provider_slash",
            "client_action": None,
            "server_action": None,
            "prompt_action": None,
            "requires_project": True,
            "requires_session": True,
            "enabled": False,
            "disabled_reason": "Discovered from a CLI slash-command help section, but this provider does not expose headless slash execution through the current adapter.",
        }
        for name, description in sorted(parsed.items())
    ]

    if commands:
        return commands, _base_capabilities(True)

    reason = "No slash commands were found in CLI help output."
    if probe_errors:
        reason = probe_errors[-1]
    return [], _base_capabilities(False, reason)


def _run_help_probe(args: tuple[str, ...]) -> tuple[str, str | None]:
    try:
        completed = subprocess.run(
            list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=4,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", f"{' '.join(args)} timed out"
    except OSError as exc:
        return "", str(exc)

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if not output.strip():
        return "", f"{' '.join(args)} returned no help output"
    return output, None


def _parse_slash_commands(text: str) -> list[tuple[str, str]]:
    cleaned = _ANSI_RE.sub("", text)
    commands: dict[str, str] = {}
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped or "/" not in stripped:
            continue
        match = _SLASH_LINE_RE.match(stripped)
        if not match:
            continue
        name = match.group(1)
        if name in {"/api", "/path", "/tmp", "/usr", "/var"}:
            continue
        description = (match.group(2) or "").strip(" -:\t")
        commands.setdefault(name, description)
    return sorted(commands.items())
