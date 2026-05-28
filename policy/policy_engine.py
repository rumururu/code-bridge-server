"""Default policy classification for Agent Cockpit operations."""

from typing import Any

ALLOW_OPERATIONS = {
    "health.read",
    "connection.status",
    "project.read",
    "chat.send",
    "log.tail",
    "preview.token",
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

    if normalized in ALLOW_OPERATIONS:
        return _decision(
            operation=normalized,
            effect="allow",
            risk_level="low",
            reason="Operation is read-only or low risk",
        )

    if normalized in CONFIRM_ONCE_OPERATIONS:
        return _decision(
            operation=normalized,
            effect="confirm_once",
            risk_level="medium",
            reason="Operation changes local state and requires confirmation",
        )

    if normalized in CONFIRM_EACH_OPERATIONS:
        return _decision(
            operation=normalized,
            effect="confirm_each",
            risk_level="high",
            reason="Operation can modify the system, network, repository, or device",
        )

    if normalized in DESKTOP_ONLY_OPERATIONS:
        return _decision(
            operation=normalized,
            effect="desktop_only",
            risk_level="high",
            reason="Operation changes host-level trust or exposure and requires desktop approval",
        )

    if _details_indicate_secret_access(details):
        return _decision(
            operation=normalized,
            effect="desktop_only",
            risk_level="critical",
            reason="Operation may access secrets and requires desktop approval",
        )

    return _decision(
        operation=normalized,
        effect="confirm_each",
        risk_level="medium",
        reason="Unknown operations require explicit confirmation",
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
