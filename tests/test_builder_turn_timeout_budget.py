"""Every client must outlast the server's Configurator ceiling.

A builder turn runs as a background job: the server keeps working after the
client stops asking, so the client is the only party that can throw a finished
answer away. If its give-up point sits at or below the server's own ceiling,
a turn that completes just under the limit is reported to the user as "no
answer" while the reply it produced is discarded.

That is not hypothetical. Both numbers were 120 — the app polled 120 times at
one second against a 120-second server timeout — so the two gave up in the
same instant, and every slow-but-successful turn was a coin flip.

The three numbers live in three languages and cannot import each other, which
is exactly why they drift. This file reads them where they are written and
checks the one relationship that has to hold.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.agents import (  # noqa: E402
    BUILDER_CONVERSE_FAST_TIMEOUT_SECONDS,
    BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS,
)

REPO_ROOT = SERVER_DIR.parent
DASHBOARD_TEMPLATE = SERVER_DIR / "dashboard" / "templates" / "agents.html"
BUILDER_PROVIDER = REPO_ROOT / "lib" / "providers" / "builder_provider.dart"

#: How much slack a client must keep beyond the server's ceiling. Not zero:
#: equal budgets are the bug this file exists for, and a client that gives up
#: the same second the server does still races it.
MINIMUM_CLIENT_SLACK_SECONDS = 30.0


def _number(path: Path, pattern: str) -> float:
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(
            f"{path.name} no longer declares {pattern!r}. The constant was "
            "renamed or removed; this check cannot be silently skipped, "
            "because a missing budget is how the two ends drifted apart."
        )
    return float(match.group(1))


class ClientPatienceExceedsServerCeilingTest(unittest.TestCase):
    def test_the_dashboard_waits_longer_than_the_server_works(self):
        interval_ms = _number(DASHBOARD_TEMPLATE, r"BUILDER_POLL_MS\s*=\s*(\d+)")
        limit = _number(DASHBOARD_TEMPLATE, r"BUILDER_POLL_LIMIT\s*=\s*(\d+)")
        budget = interval_ms / 1000.0 * limit
        self.assertGreaterEqual(
            budget,
            BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS + MINIMUM_CLIENT_SLACK_SECONDS,
            f"the dashboard gives up after {budget:.0f}s but the server keeps "
            f"working for {BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS:.0f}s — raise "
            "BUILDER_POLL_LIMIT in dashboard/templates/agents.html",
        )

    def test_the_app_waits_longer_than_the_server_works(self):
        # One-second delay per attempt in `_converseViaJob`, so the attempt
        # count is the budget in seconds. Asserted rather than assumed, so a
        # change to the delay cannot quietly halve the budget.
        interval = _number(
            BUILDER_PROVIDER,
            r"_converseJobPollAttempts\b[\s\S]{0,400}?"
            r"await Future<void>\.delayed\(const Duration\(seconds: (\d+)\)\)",
        )
        attempts = _number(
            BUILDER_PROVIDER, r"_converseJobPollAttempts\s*=\s*(\d+)"
        )
        budget = interval * attempts
        self.assertGreaterEqual(
            budget,
            BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS + MINIMUM_CLIENT_SLACK_SECONDS,
            f"the app gives up after {budget:.0f}s but the server keeps "
            f"working for {BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS:.0f}s — raise "
            "_converseJobPollAttempts in lib/providers/builder_provider.dart",
        )

    def test_the_synchronous_ceiling_stays_below_the_job_ceiling(self):
        # The fast path answers inline and hands off to a job when it runs
        # long; a fast timeout at or above the job's would make the handoff
        # unreachable and put every slow turn back on the synchronous path
        # that reports it as a failed turn.
        self.assertLess(
            BUILDER_CONVERSE_FAST_TIMEOUT_SECONDS,
            BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
