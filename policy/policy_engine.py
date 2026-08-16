"""Default policy classification for Agent Cockpit operations.

The static operation table below is the *first* policy gate. After it
runs we consult three classifiers — :mod:`policy.command_classifier`,
:mod:`policy.path_guard`, and :mod:`policy.secret_classifier` — and let
them **escalate** (never downgrade) the effect based on the actual payload
contents. That means a static ``confirm_each`` shell call becomes
``forbidden`` if the command is ``rm -rf /``, a ``confirm_once`` file write
becomes ``desktop_only`` if the target is ``~/.ssh/id_rsa``, and any
operation whose details carry an AWS or OpenAI key gets bumped to
``desktop_only`` so the secret is not exposed through a remote-paired
mobile client.
"""

from typing import Any

from .command_classifier import classify_command, escalate
from .path_guard import classify_path, classify_paths
from .secret_classifier import classify_details

ALLOW_OPERATIONS = {
    "health.read",
    "connection.status",
    "project.read",
    "chat.send",
    "log.tail",
    "preview.token",
    # Reading a file changes nothing. What makes *which* file matter is
    # handled below, not here: `file.read` is a `_PATH_OPERATIONS` member, so
    # every request carrying a path runs through the path guard, and
    # `_apply_classifiers` escalates away from `allow` whenever that path is
    # sensitive. An allow here is "allow by default", never "allow always".
    "file.read",
}

CONFIRM_ONCE_OPERATIONS = {
    "file.copy",
    "file.upload",
    "file.write",
    "process.devserver",
    "provider.resume",
}

CONFIRM_EACH_OPERATIONS = {
    "file.delete",
    "file.move",
    "process.terminal",
    "package.install",
    "git.commit",
    "git.tag",
    "git.push",
    "device.control",
    "network.external",
    "settings.write",
}

DESKTOP_ONLY_OPERATIONS = {
    "identity.revoke",
    "identity.takeover",
    "settings.accessible_roots",
    "tunnel.start",
    "tunnel.stop",
    "provider.install",
    "provider.update",
    "danger.full_access",
}

FORBIDDEN_OPERATIONS = {
    "audit.disable",
    "credential.exfiltrate",
}


def decide_policy(
    operation: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the default policy decision for one operation."""
    normalized = (operation or "").strip()
    details = details or {}

    if not normalized:
        return _decision(
            operation=normalized,
            effect="forbidden",
            risk_level="high",
            reason="Operation is required",
        )

    if normalized in FORBIDDEN_OPERATIONS:
        return _decision(
            operation=normalized,
            effect="forbidden",
            risk_level="critical",
            reason="Operation is forbidden by default policy",
        )

    base_effect: str
    base_reason: str
    base_risk: str

    if normalized in ALLOW_OPERATIONS:
        base_effect, base_risk, base_reason = (
            "allow",
            "low",
            "Operation is read-only or low risk",
        )
    elif normalized in CONFIRM_ONCE_OPERATIONS:
        base_effect, base_risk, base_reason = (
            "confirm_once",
            "medium",
            "Operation changes local state and requires confirmation",
        )
    elif normalized in CONFIRM_EACH_OPERATIONS:
        base_effect, base_risk, base_reason = (
            "confirm_each",
            "high",
            "Operation can modify the system, network, repository, or device",
        )
    elif normalized in DESKTOP_ONLY_OPERATIONS:
        base_effect, base_risk, base_reason = (
            "desktop_only",
            "high",
            "Operation changes host-level trust or exposure and requires desktop approval",
        )
    elif _details_indicate_secret_access(details):
        base_effect, base_risk, base_reason = (
            "desktop_only",
            "critical",
            "Operation may access secrets and requires desktop approval",
        )
    else:
        base_effect, base_risk, base_reason = (
            "confirm_each",
            "medium",
            "Unknown operations require explicit confirmation",
        )

    return _apply_classifiers(
        operation=normalized,
        details=details,
        base_effect=base_effect,
        base_risk=base_risk,
        base_reason=base_reason,
    )


def default_policy_snapshot() -> dict[str, Any]:
    """Return the built-in policy sets for settings and diagnostics UI."""
    return {
        "allow": sorted(ALLOW_OPERATIONS),
        "confirm_once": sorted(CONFIRM_ONCE_OPERATIONS),
        "confirm_each": sorted(CONFIRM_EACH_OPERATIONS),
        "desktop_only": sorted(DESKTOP_ONLY_OPERATIONS),
        "forbidden": sorted(FORBIDDEN_OPERATIONS),
    }


def _decision(
    *,
    operation: str,
    effect: str,
    risk_level: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "effect": effect,
        "risk_level": risk_level,
        "reason": reason,
        "approval_required": effect in {"confirm_once", "confirm_each", "desktop_only"},
        "desktop_only": effect == "desktop_only",
        "forbidden": effect == "forbidden",
    }


# ---------------------------------------------------------------------------
# Classifier orchestration
# ---------------------------------------------------------------------------

# Which operations care about which classifier types. Operations not listed here
# still get a secret-classifier sweep over their details, because any payload
# can accidentally carry a token in a free-text field.
_COMMAND_OPERATIONS = {
    "process.terminal",
    "process.devserver",
    "process.shell",
}

_PATH_OPERATIONS = {
    "file.copy",
    "file.delete",
    "file.move",
    "file.read",
    "file.upload",
    "file.write",
    "process.devserver",
    "process.terminal",
}


_RISK_BY_EFFECT = {
    "allow": "low",
    "confirm_once": "medium",
    "confirm_each": "high",
    "desktop_only": "high",
    "forbidden": "critical",
}


def _apply_classifiers(
    *,
    operation: str,
    details: dict[str, Any],
    base_effect: str,
    base_risk: str,
    base_reason: str,
) -> dict[str, Any]:
    """Escalate the base effect using command/path/secret classifiers."""
    classifications: dict[str, Any] = {}
    candidate_effects: list[str] = [base_effect]
    extra_reasons: list[str] = []

    if operation in _COMMAND_OPERATIONS or details.get("command"):
        command = details.get("command")
        if isinstance(command, list):
            from .command_classifier import classify_argv
            command_result = classify_argv(command)
        else:
            command_result = classify_command(command if isinstance(command, str) else None)
        if command_result.matched or command is not None:
            classifications["command"] = command_result.to_dict()
            candidate_effects.append(command_result.effect)
            if command_result.matched:
                extra_reasons.extend(command_result.reasons)

    if operation in _PATH_OPERATIONS or any(
        details.get(key) for key in ("path", "paths", "target_path")
    ):
        workspace_root = _extract_workspace_root(details)
        path_targets = _extract_paths(details)
        if path_targets:
            path_result = classify_paths(path_targets, workspace_root=workspace_root)
            classifications["path"] = path_result.to_dict()
            candidate_effects.append(path_result.effect)
            if path_result.category != "workspace_internal":
                extra_reasons.extend(path_result.reasons)

    secret_result = classify_details(details)
    if secret_result.matched:
        classifications["secret"] = secret_result.to_dict()
        candidate_effects.append(secret_result.effect)
        extra_reasons.append(
            f"detected {len(secret_result.findings)} likely secret(s) in details"
        )

    final_effect = escalate(*candidate_effects)
    final_reason = base_reason
    if final_effect != base_effect and extra_reasons:
        final_reason = f"{base_reason}; escalated: {'; '.join(extra_reasons)}"

    decision = _decision(
        operation=operation,
        effect=final_effect,
        risk_level=_RISK_BY_EFFECT.get(final_effect, base_risk),
        reason=final_reason,
    )
    if classifications:
        decision["classifications"] = classifications
    return decision


def _extract_workspace_root(details: dict[str, Any]) -> str | None:
    for key in ("workspace_root", "workspace_path", "project_path", "cwd"):
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_paths(details: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("path", "target_path", "destination", "source", "resource"):
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value)
    raw_paths = details.get("paths")
    if isinstance(raw_paths, list):
        for value in raw_paths:
            if isinstance(value, str) and value.strip():
                paths.append(value)
    return paths


def _details_indicate_secret_access(details: dict[str, Any]) -> bool:
    path_values: list[str] = []
    for key in ("path", "resource", "target", "cwd"):
        value = details.get(key)
        if isinstance(value, str):
            path_values.append(value.lower())
    for value in details.get("paths") or []:
        if isinstance(value, str):
            path_values.append(value.lower())

    secret_markers = (
        ".env",
        "id_rsa",
        "id_ed25519",
        "keystore",
        "google-services.json",
        "firebase_config.json",
        "secret",
        "token",
    )
    return any(marker in path for path in path_values for marker in secret_markers)
