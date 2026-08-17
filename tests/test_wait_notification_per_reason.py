"""One quiet day per problem, not one quiet day per agent.

The once-a-day throttle was keyed on the agent alone. That made a *different*
problem look like a repeat: an agent that reported an expired login in the
morning stayed silent about a captcha in the afternoon, even though the captcha
needs a different action from the person and blocks the agent just as hard.

Keying on the agent *and* the wait reason keeps the flood protection — each
reason still notifies at most once a day, and the set of reasons is small and
closed — while letting a new kind of trouble through.

The reason has to be stored to be queried, so it is a column rather than
something inferred from the notification title: the title contains the agent's
name and a human label, both of which change without the problem changing.
"""

from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import agent.task_orchestrator as orchestrator  # noqa: E402


class _Store:
    def __init__(self, runs=None) -> None:
        self._runs = runs or []

    def list_runs(self, agent_id=None, limit=10):
        return self._runs


class _Notifications:
    """Records what was written and answers the throttle's query honestly."""

    def __init__(self, rows=None) -> None:
        self.rows = list(rows or [])
        self.queries: list[dict] = []

    def list_notifications(self, *, agent_id=None, level=None, reason=None, limit=50):
        self.queries.append({"agent_id": agent_id, "level": level, "reason": reason})
        found = [
            row
            for row in self.rows
            if (agent_id is None or row.get("agent_id") == agent_id)
            and (level is None or row.get("level") == level)
            and (reason is None or row.get("reason") == reason)
        ]
        found.sort(key=lambda row: row["created_at"], reverse=True)
        return found[:limit]


def _row(reason, *, hours_ago=1.0, agent_id="agent_1", level="warning"):
    return {
        "agent_id": agent_id,
        "level": level,
        "reason": reason,
        "created_at": (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat(),
    }


class PerReasonThrottleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._real = orchestrator.get_notification_store if hasattr(
            orchestrator, "get_notification_store"
        ) else None

    def _gate(self, rows, reason, *, runs=None):
        notifications = _Notifications(rows)
        import agent.notification_store as notification_store

        original = notification_store.get_notification_store
        notification_store.get_notification_store = lambda: notifications
        try:
            allowed, why = orchestrator._wait_notification_gate(
                store=_Store(runs), agent_id="agent_1", run_id="run_2", reason=reason
            )
        finally:
            notification_store.get_notification_store = original
        return allowed, why, notifications

    def test_a_repeat_of_the_same_problem_stays_quiet(self) -> None:
        allowed, why, _ = self._gate([_row("login_required")], "login_required")
        self.assertFalse(allowed)
        self.assertIn("login_required", why)

    def test_a_different_problem_gets_through(self) -> None:
        """The case that was broken: a captcha silenced by a morning login."""
        allowed, _, _ = self._gate([_row("login_required")], "captcha_or_bot_challenge")
        self.assertTrue(allowed)

    def test_the_query_is_scoped_to_the_reason(self) -> None:
        _, _, notifications = self._gate([], "login_required")
        self.assertEqual(notifications.queries[-1]["reason"], "login_required")
        self.assertEqual(notifications.queries[-1]["level"], "warning")

    def test_the_window_still_expires(self) -> None:
        allowed, why, _ = self._gate(
            [_row("login_required", hours_ago=25)], "login_required"
        )
        self.assertTrue(allowed)
        self.assertIn("24h", why)

    def test_a_failure_notification_does_not_silence_a_wait(self) -> None:
        """Different level, different channel; it never did, and must not
        start now that both rows carry a reason."""
        allowed, _, _ = self._gate(
            [_row("run_failed", level="error")], "login_required"
        )
        self.assertTrue(allowed)

    def test_rows_written_before_the_column_existed_do_not_suppress(self) -> None:
        """An upgrade must not start out silent: a null reason matches nothing,
        so the first park of each kind notifies once."""
        allowed, _, _ = self._gate([_row(None)], "login_required")
        self.assertTrue(allowed)

    def test_no_agent_still_notifies(self) -> None:
        allowed, why = orchestrator._wait_notification_gate(
            store=_Store(), agent_id=None, run_id="run_2", reason="login_required"
        )
        self.assertTrue(allowed)
        self.assertIn("no agent", why)


class TheReasonIsPersistedTest(unittest.TestCase):
    """A throttle can only key on what was written down."""

    def test_the_store_accepts_and_filters_on_reason(self) -> None:
        import inspect

        from agent.notification_store import NotificationStore

        create = inspect.signature(NotificationStore.create).parameters
        listing = inspect.signature(NotificationStore.list_notifications).parameters
        self.assertIn("reason", create)
        self.assertIn("reason", listing)

    def test_the_column_exists_on_a_fresh_database(self) -> None:
        import sqlite3

        from core.database import _migrate_agent_notifications

        conn = sqlite3.connect(":memory:")
        _migrate_agent_notifications(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_notifications)")}
        self.assertIn("reason", columns)

    def test_the_schema_version_was_bumped_so_the_migration_runs(self) -> None:
        """Migrations are version-gated. Editing the function without moving
        the version leaves every existing database untouched — measured: the
        deployed database still had no `reason` column after a restart."""
        from core.database import AGENT_NOTIFICATIONS_SCHEMA_VERSION

        self.assertGreater(AGENT_NOTIFICATIONS_SCHEMA_VERSION, 2026080200)

    def test_an_existing_database_gains_the_column(self) -> None:
        """The migration runs against databases that predate it."""
        import sqlite3

        from core.database import _migrate_agent_notifications

        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE agent_notifications ("
            "id TEXT PRIMARY KEY, run_id TEXT, task_id TEXT, agent_id TEXT, "
            "title TEXT NOT NULL, body TEXT, level TEXT NOT NULL DEFAULT 'info', "
            "read_at TIMESTAMP, created_at TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO agent_notifications (id, title) VALUES ('n1', 'older row')"
        )
        _migrate_agent_notifications(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_notifications)")}
        self.assertIn("reason", columns)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM agent_notifications").fetchone()[0], 1
        )


if __name__ == "__main__":
    unittest.main()
