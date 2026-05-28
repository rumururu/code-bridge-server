"""Detect well-known secret formats in operation payloads.

This classifier is consulted by :func:`policy.policy_engine.decide_policy`
*before* an operation runs. When it finds a credential pattern in any text
field of the operation details, it escalates the policy effect — typically
to ``desktop_only`` for "this request body contains a token, route it
through the desktop dashboard" — instead of letting an unattended approval
on a mobile client expose the value over a third-party LLM provider.

The classifier never logs the secret itself. Findings carry a redacted
preview (``"ghp_abcd…wxyz"``) so the Cockpit can show *what kind* of secret
was detected without leaking the value into audit storage.

Patterns favor false-positives over false-negatives. A few extra approval
prompts are acceptable; silently exfiltrating an API key is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .command_classifier import (
    EFFECT_CONFIRM_EACH,
    EFFECT_DESKTOP_ONLY,
    EFFECT_FORBIDDEN,
    escalate,
)


# ---------------------------------------------------------------------------
# Pattern catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SecretPattern:
    name: str
    description: str
    effect: str
    pattern: re.Pattern[str]


def _compile(rules: Iterable[tuple[str, str, str, str]]) -> list[_SecretPattern]:
    return [
        _SecretPattern(name=name, description=description, effect=effect, pattern=re.compile(regex))
        for name, description, effect, regex in rules
    ]


_PATTERNS: list[_SecretPattern] = _compile(
    [
        # ---- Cloud / SaaS API keys (desktop_only — sensitive, but request still
        # may need to happen via dashboard) ------------------------------------
        (
            "aws_access_key_id",
            "AWS Access Key ID",
            EFFECT_DESKTOP_ONLY,
            r"\b(?:AKIA|ASIA|AGPA|AROA|AIDA|ANPA|ANVA)[A-Z0-9]{16}\b",
        ),
        (
            "aws_secret_access_key",
            "AWS Secret Access Key",
            EFFECT_DESKTOP_ONLY,
            r"(?i)aws[_-]?(?:secret|secret[_-]?access)[_-]?key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?",
        ),
        (
            "github_pat",
            "GitHub personal access / app / refresh token",
            EFFECT_DESKTOP_ONLY,
            r"\bgh[pousr]_[A-Za-z0-9]{30,255}\b",
        ),
        (
            "github_oauth_classic",
            "Legacy GitHub OAuth token (40 hex)",
            EFFECT_DESKTOP_ONLY,
            r"(?i)\bgithub[_-]?(?:token|api[_-]?token)\s*[=:]\s*['\"]?[0-9a-f]{40}['\"]?",
        ),
        (
            "openai_api_key",
            "OpenAI API key",
            EFFECT_DESKTOP_ONLY,
            r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b",
        ),
        (
            "anthropic_api_key",
            "Anthropic API key",
            EFFECT_DESKTOP_ONLY,
            r"\bsk-ant-(?:api|admin)[A-Za-z0-9_\-]{20,}\b",
        ),
        (
            "google_api_key",
            "Google / Firebase API key",
            EFFECT_DESKTOP_ONLY,
            r"\bAIza[0-9A-Za-z_\-]{35}\b",
        ),
        (
            "google_oauth_refresh",
            "Google OAuth refresh token",
            EFFECT_DESKTOP_ONLY,
            r"\b1//0[A-Za-z0-9_\-]{40,}\b",
        ),
        (
            "stripe_live_key",
            "Stripe live API key",
            EFFECT_DESKTOP_ONLY,
            r"\bsk_live_[A-Za-z0-9]{20,}\b",
        ),
        (
            "stripe_restricted_key",
            "Stripe restricted key",
            EFFECT_DESKTOP_ONLY,
            r"\brk_live_[A-Za-z0-9]{20,}\b",
        ),
        (
            "stripe_test_key",
            "Stripe test API key",
            EFFECT_CONFIRM_EACH,
            r"\bsk_test_[A-Za-z0-9]{20,}\b",
        ),
        (
            "slack_token",
            "Slack token",
            EFFECT_DESKTOP_ONLY,
            r"\bxox[abprso]-[A-Za-z0-9\-]{10,}\b",
        ),
        (
            "discord_bot_token",
            "Discord bot token",
            EFFECT_DESKTOP_ONLY,
            r"\b[MN][A-Za-z\d]{23}\.[\w\-]{6}\.[\w\-]{27}\b",
        ),
        (
            "telegram_bot_token",
            "Telegram bot token",
            EFFECT_DESKTOP_ONLY,
            r"\b\d{6,12}:[A-Za-z0-9_\-]{35}\b",
        ),
        (
            "twilio_auth_token",
            "Twilio auth token",
            EFFECT_DESKTOP_ONLY,
            r"\bSK[0-9a-fA-F]{32}\b",
        ),
        (
            "supabase_service_role",
            "Supabase service role JWT",
            EFFECT_DESKTOP_ONLY,
            r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b",
        ),
        (
            "google_service_account",
            "Google service account JSON marker",
            EFFECT_DESKTOP_ONLY,
            r'"type"\s*:\s*"service_account"',
        ),
        (
            "private_key_pem",
            "PEM-encoded private key",
            EFFECT_FORBIDDEN,
            r"-----BEGIN(?:\s+OPENSSH| ENCRYPTED| RSA| EC| DSA| PGP)?\s+PRIVATE KEY-----",
        ),
        (
            "ssh_authorized_key",
            "Inline SSH private key block",
            EFFECT_FORBIDDEN,
            r"-----BEGIN OPENSSH PRIVATE KEY-----",
        ),
        # ---- Generic keyish patterns ----------------------------------------
        (
            "generic_password_assignment",
            "Generic password-like assignment",
            EFFECT_CONFIRM_EACH,
            r"(?i)(?:password|passwd|pwd|api[_-]?secret)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
        ),
        (
            "generic_bearer_token",
            "Bearer token in Authorization header",
            EFFECT_CONFIRM_EACH,
            r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9_\-\.=+/]{20,}",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SecretFinding:
    type: str
    description: str
    effect: str
    preview: str
    field: str

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.type,
            "description": self.description,
            "effect": self.effect,
            "preview": self.preview,
            "field": self.field,
        }


@dataclass
class SecretClassification:
    effect: str
    findings: list[SecretFinding] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return bool(self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "effect": self.effect,
            "matched": self.matched,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _redact(value: str) -> str:
    """Return a short safe preview of ``value`` for audit storage.

    Never returns more than the first 4 + last 4 characters; the rest is
    replaced with an ellipsis. Length is preserved as a separate field so the
    Cockpit can show "AWS access key, 20 chars" without ever holding the
    cleartext value.
    """
    cleaned = value.strip()
    if len(cleaned) <= 8:
        return "[redacted]"
    return f"{cleaned[:4]}…{cleaned[-4:]}"


def classify_text(text: str | None, *, field_name: str = "value") -> SecretClassification:
    """Scan a string and return any secret findings."""
    if not text:
        return SecretClassification(effect=EFFECT_CONFIRM_EACH if text == "" else EFFECT_CONFIRM_EACH)

    findings: list[SecretFinding] = []
    seen: set[tuple[str, str]] = set()  # de-dupe identical (type, preview) hits.

    for pat in _PATTERNS:
        for match in pat.pattern.finditer(text):
            value = match.group(0)
            preview = _redact(value)
            key = (pat.name, preview)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                SecretFinding(
                    type=pat.name,
                    description=pat.description,
                    effect=pat.effect,
                    preview=preview,
                    field=field_name,
                )
            )

    if not findings:
        # Caller decides the floor; the classifier itself does not push toward
        # restriction when no pattern matched.
        return SecretClassification(effect=EFFECT_CONFIRM_EACH)

    return SecretClassification(
        effect=escalate(*(f.effect for f in findings)),
        findings=findings,
    )


def classify_details(details: dict[str, Any] | None) -> SecretClassification:
    """Recursively scan all string leaves of an operation details payload.

    Each leaf is scanned independently; the final effect is the most
    restrictive across all findings.
    """
    if not details:
        return SecretClassification(effect=EFFECT_CONFIRM_EACH)

    findings: list[SecretFinding] = []
    seen: set[tuple[str, str, str]] = set()

    def _walk(node: Any, prefix: str) -> None:
        if isinstance(node, str):
            sub = classify_text(node, field_name=prefix or "value")
            for finding in sub.findings:
                key = (finding.type, finding.preview, finding.field)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(finding)
        elif isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                _walk(value, f"{prefix}[{idx}]" if prefix else f"[{idx}]")

    _walk(details, "")

    if not findings:
        return SecretClassification(effect=EFFECT_CONFIRM_EACH)

    return SecretClassification(
        effect=escalate(*(f.effect for f in findings)),
        findings=findings,
    )


def list_patterns() -> list[dict[str, str]]:
    """Expose patterns for the Cockpit policy view (without compiled regex)."""
    return [
        {
            "name": pat.name,
            "description": pat.description,
            "effect": pat.effect,
            "pattern": pat.pattern.pattern,
        }
        for pat in _PATTERNS
    ]
