"""One place that decides what a stored timestamp means.

SQLite's ``CURRENT_TIMESTAMP`` writes naive UTC — "2026-08-01 02:57:48" with
nothing to say which zone that is. Handed to a client as-is, a phone in UTC+9
reads it as local time and places the row nine hours in the past. That is how
an agent that had run half an hour earlier came to report "9 hours ago", while
the field beside it (already emitted with an offset) said something else
entirely.

Every store sends its rows through :func:`to_utc_iso`, so no timestamp can
leave the server without an instant attached.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

# Naive only: anything already carrying an offset (or a "Z") is left alone.
_NAIVE_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?$")


def to_utc_iso(value: Any) -> Any:
    """Return ``value`` as an ISO-8601 UTC instant, or unchanged.

    Non-strings, blanks, already-offset timestamps and anything unparseable
    pass through untouched — this normalizes, it never guesses.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or not _NAIVE_TIMESTAMP.match(text):
        return value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return value
    return parsed.replace(tzinfo=UTC).isoformat()


__all__ = ["to_utc_iso"]
