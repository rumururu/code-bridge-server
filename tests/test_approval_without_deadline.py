"""An approval with no deadline of its own still expires, on the default one.

Rows written before approvals had deadlines (or while expiry was switched
off) carry ``expires_at = NULL``, and the expiry sweep only ever looked at
rows whose deadline had passed — so one such row, an agent asking to read
``~/.ssh/id_rsa`` on 2026-08-16 for a run that no longer existed, sat in the
pending list for eighteen days telling the dashboard a schedule was waiting
on it. The sweep now reads a missing deadline as ``created_at`` plus the
default expiry.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from approvals import approval_store  # noqa: E402
from approvals.approval_service import expire_stale_approvals  # noqa: E402
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402


class ApprovalWithoutDeadlineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "approvals.db"
        approval_store._approval_store = None
        audit_store._audit_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = approval_store.get_approval_store()

    def _restore(self) -> None:
        approval_store._approval_store = None
        audit_store._audit_store = None
        database.DB_PATH = self._original

    def _pending_without_deadline(self, *, age_days: float) -> str:
        row = self.store.create_request(
            operation="file.read", run_id="run_gone", risk_level="high", expires_at=None
        )
        created = (datetime.now(tz=timezone.utc) - timedelta(days=age_days)).isoformat()
        with database.get_db_connection() as conn:
            conn.execute(
                "UPDATE approval_requests SET created_at = ? WHERE id = ?",
                (created, row["id"]),
            )
            conn.commit()
        return row["id"]

    def _server_format_row(self, *, age_hours: float) -> str:
        """A row whose ``created_at`` is what the server itself writes.

        ``CURRENT_TIMESTAMP`` yields ``YYYY-MM-DD HH:MM:SS`` — no ``T``, no
        offset. The helper above writes ISO strings, which is exactly why the
        text comparison in the sweep went unnoticed: ISO against ISO sorts
        correctly.
        """
        row = self.store.create_request(
            operation="feedback.reply.send", run_id=None, risk_level="high", expires_at=None
        )
        created = (datetime.now(tz=timezone.utc) - timedelta(hours=age_hours)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        with database.get_db_connection() as conn:
            conn.execute(
                "UPDATE approval_requests SET created_at = ? WHERE id = ?", (created, row["id"])
            )
            conn.commit()
        return row["id"]

    def test_a_fresh_server_written_row_survives_the_sweep(self):
        # Regression: compared as text, "2026-09-15 23:19:32" < "2026-09-15T00:00:05+00:00",
        # so anything created on the cutoff's calendar day expired at UTC midnight.
        # Every age below lands on "yesterday" for some wall-clock time, so at
        # least one of them reproduces the old behaviour whenever this runs.
        fresh = [self._server_format_row(age_hours=h) for h in (0.5, 6, 12, 18, 23)]
        stale = self._server_format_row(age_hours=25)

        with patch("approvals.approval_service.approval_expiry_seconds", return_value=24 * 3600):
            expired = expire_stale_approvals()

        self.assertEqual([r["id"] for r in expired], [stale])
        self.assertEqual(
            sorted(r["id"] for r in self.store.list_pending()), sorted(fresh)
        )

    def test_an_old_row_with_no_deadline_is_expired_by_the_sweep(self):
        approval_id = self._pending_without_deadline(age_days=18)
        self.assertEqual([r["id"] for r in self.store.list_pending()], [approval_id])

        with patch("approvals.approval_service.approval_expiry_seconds", return_value=24 * 3600):
            expired = expire_stale_approvals()

        self.assertEqual([r["id"] for r in expired], [approval_id])
        self.assertEqual(self.store.list_pending(), [])
        self.assertEqual(self.store.get_request(approval_id)["status"], "expired")

    def test_a_fresh_row_with_no_deadline_is_left_alone(self):
        approval_id = self._pending_without_deadline(age_days=0.1)
        with patch("approvals.approval_service.approval_expiry_seconds", return_value=24 * 3600):
            self.assertEqual(expire_stale_approvals(), [])
        self.assertEqual([r["id"] for r in self.store.list_pending()], [approval_id])

    def test_with_expiry_switched_off_nothing_changes(self):
        approval_id = self._pending_without_deadline(age_days=400)
        with patch("approvals.approval_service.approval_expiry_seconds", return_value=0):
            self.assertEqual(expire_stale_approvals(), [])
        self.assertEqual([r["id"] for r in self.store.list_pending()], [approval_id])
