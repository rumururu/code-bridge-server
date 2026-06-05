"""Redaction helpers for local audit records.

Returns redacted payloads alongside the set of detected categories so the
caller can persist a forensics-friendly record of *what kind* of secret
was stripped (e.g. ``aws_key``, ``jwt``, ``email``, ``private_key``).
"""

import re
from typing import Any

# Field-name match (case-insensitive, dashes/underscores normalised).
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

# Category constants — exported for tests and downstream consumers.
CATEGORY_GENERIC_SECRET = "generic_secret"
CATEGORY_AWS_KEY = "aws_key"
CATEGORY_JWT = "jwt"
CATEGORY_EMAIL = "email"
CATEGORY_PRIVATE_KEY = "private_key"

# Patterns evaluated against string values (not field names).
_AWS_ACCESS_KEY_RE = re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
)

_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_PRIVATE_KEY_RE, CATEGORY_PRIVATE_KEY),
    (_AWS_ACCESS_KEY_RE, CATEGORY_AWS_KEY),
    (_JWT_RE, CATEGORY_JWT),
    (_EMAIL_RE, CATEGORY_EMAIL),
)


def _key_matches_secret(key: str) -> bool:
    normalized_key = str(key).lower().replace("-", "_")
    return normalized_key in SECRET_KEYS or normalized_key.endswith("_token")


def _redact_string_value(value: str, categories: set[str]) -> str:
    """Detect secret patterns inside a string value.

    Returns the value with secret-bearing substrings replaced by
    ``[redacted:<category>]``. ``categories`` is mutated with every
    matched category so the caller can record what was caught.
    """
    redacted = value
    for pattern, category in _VALUE_PATTERNS:
        if pattern.search(redacted):
            categories.add(category)
            redacted = pattern.sub(f"[redacted:{category}]", redacted)
    return redacted


def _walk(value: Any, categories: set[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _key_matches_secret(key):
                categories.add(CATEGORY_GENERIC_SECRET)
                out[str(key)] = "[redacted]"
            else:
                out[str(key)] = _walk(item, categories)
        return out
    if isinstance(value, list):
        return [_walk(item, categories) for item in value]
    if isinstance(value, tuple):
        return [_walk(item, categories) for item in value]
    if isinstance(value, str):
        return _redact_string_value(value, categories)
    return value


def redact_payload(value: Any) -> tuple[Any, set[str]]:
    """Return a JSON-safe redacted copy plus the detected category set.

    The category set is a flat collection of strings such as
    ``{"aws_key", "email"}`` — empty when nothing matched. Callers that
    just want the redacted payload can drop the second tuple element.
    """
    categories: set[str] = set()
    redacted = _walk(value, categories)
    return redacted, categories
