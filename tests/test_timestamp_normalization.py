"""No timestamp leaves the server without an instant attached.

SQLite writes `CURRENT_TIMESTAMP` as naive UTC — "2026-08-01 02:57:48", with
nothing to say which zone that is. Sent as-is, a client in UTC+9 reads it as
local time and puts the row nine hours in the past: an agent that had run
thirty minutes earlier reported "9 hours ago" on the phone, while the field
next to it (already emitted with an offset) disagreed.

The failure is silent and locale-dependent — correct on a UTC machine, wrong
by exactly the offset everywhere else — so it is pinned at the conversion
layer rather than per endpoint.
"""

import re
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from core.timestamps import to_utc_iso

_HAS_OFFSET = re.compile(r"([+-]\d{2}:\d{2}|Z)$")


class ToUtcIsoTest(unittest.TestCase):
    def test_naive_sqlite_timestamp_gains_utc(self):
        self.assertEqual(
            to_utc_iso("2026-08-01 02:57:48"), "2026-08-01T02:57:48+00:00"
        )

    def test_naive_iso_form_too(self):
        self.assertEqual(
            to_utc_iso("2026-08-01T02:57:48"), "2026-08-01T02:57:48+00:00"
        )

    def test_fractional_seconds_survive(self):
        self.assertTrue(_HAS_OFFSET.search(to_utc_iso("2026-08-01 02:57:48.123456")))

    def test_an_offset_already_present_is_left_alone(self):
        # Re-stamping would be a second chance to get the zone wrong.
        for value in ("2026-08-01T02:57:48+00:00", "2026-08-01T11:57:48+09:00"):
            self.assertEqual(to_utc_iso(value), value)

    def test_non_timestamps_pass_through(self):
        for value in (None, 42, "", "  ", "never", "agent_1"):
            self.assertEqual(to_utc_iso(value), value)

    def test_unparseable_lookalike_is_not_guessed_at(self):
        self.assertEqual(to_utc_iso("2026-13-45 99:99:99"), "2026-13-45 99:99:99")


class ConverterCoverageTest(unittest.TestCase):
    """Every stored timestamp a converter emits must go through the helper.

    Adding a column and forgetting the wrapper is exactly how this got shipped
    the first time, and nothing else would catch it.
    """

    TIMESTAMP_FIELDS = (
        "created_at",
        "updated_at",
        "started_at",
        "ended_at",
        "last_used",
        "paired_at",
        "expires_at",
        "closed_at",
        "archived_at",
        "last_run_at",
        "next_run_at",
        "decided_at",
    )

    SOURCES = (
        "agent/_row_converters.py",
        "agent/schedule_store.py",
        "agent/script_store.py",
        "approvals/approval_store.py",
        "policy/policy_store.py",
    )

    def test_no_raw_timestamp_column_is_emitted(self):
        offenders: list[str] = []
        for relative in self.SOURCES:
            text = (SERVER_DIR / relative).read_text(encoding="utf-8")
            for field in self.TIMESTAMP_FIELDS:
                # `"created_at": row["created_at"]` with no wrapper around it.
                if re.search(rf'"{field}":\s*row\["{field}"\]', text):
                    offenders.append(f"{relative}:{field}")
        self.assertEqual(
            offenders,
            [],
            "these emit a naive timestamp straight from the row: "
            + ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
