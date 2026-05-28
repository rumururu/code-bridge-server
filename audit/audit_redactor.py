"""Redaction helpers for local audit records."""

from typing import Any

SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "refresh_token",
    "access_token",
    "firebase_id_token",
    "firebase_refresh_token",
    "client_secret",
    "secret",
    "token",
}


def redact_payload(value: Any) -> Any:
    """Return a JSON-safe copy with obvious secret fields redacted."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in SECRET_KEYS or normalized_key.endswith("_token"):
                redacted[str(key)] = "[redacted]"
            else:
                redacted[str(key)] = redact_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return [redact_payload(item) for item in value]
    return value

