import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import browser_session_store  # noqa: E402
from core import database  # noqa: E402


class BrowserSessionStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "browser_sessions.db"
        browser_session_store._browser_session_store = None
        database.init_db()
        self.store = browser_session_store.get_browser_session_store()

    def tearDown(self):
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_create_get_update_and_close_session(self):
        session = self.store.create(
            run_id="run_1",
            task_id="task_1",
            step_id="step_1",
            workflow_step_id="login",
            current_url="https://example.test/login",
            title="Login",
            handoff_reason="captcha_or_bot_challenge",
            metadata={"source": "test"},
        )

        self.assertTrue(session["id"].startswith("bs_"))
        self.assertEqual(session["status"], "created")
        self.assertEqual(session["run_id"], "run_1")
        self.assertEqual(session["workflow_step_id"], "login")
        self.assertEqual(session["metadata"], {"source": "test"})
        self.assertTrue(Path(session["context_dir"]).exists())

        waiting = self.store.mark_waiting(
            session["id"],
            reason="captcha_or_bot_challenge",
            current_url="https://example.test/challenge",
            title="Challenge",
        )
        assert waiting is not None
        self.assertEqual(waiting["status"], "waiting_for_user")
        self.assertEqual(waiting["current_url"], "https://example.test/challenge")

        active = self.store.find_active_for_step(run_id="run_1", step_id="step_1")
        assert active is not None
        self.assertEqual(active["id"], session["id"])

        resumed = self.store.mark_resumed(session["id"])
        assert resumed is not None
        self.assertEqual(resumed["status"], "resumed")

        closed = self.store.close(session["id"])
        assert closed is not None
        self.assertEqual(closed["status"], "closed")
        self.assertIsNotNone(closed["closed_at"])
        self.assertIsNone(self.store.find_active_for_step(run_id="run_1", step_id="step_1"))

    def test_expire_due_marks_active_sessions_expired(self):
        expired = self.store.create(
            run_id="run_expired",
            step_id="step_1",
            ttl_minutes=1,
        )
        future = self.store.create(
            run_id="run_future",
            step_id="step_2",
            ttl_minutes=60,
        )
        now = datetime.now(UTC)
        self.store.update(expired["id"], {"expires_at": now - timedelta(minutes=1)})
        self.store.update(future["id"], {"expires_at": now + timedelta(minutes=30)})

        count = self.store.expire_due(now=now)

        self.assertEqual(count, 1)
        self.assertEqual(self.store.get(expired["id"])["status"], "expired")
        self.assertEqual(self.store.get(future["id"])["status"], "created")

    def test_latest_resumable_for_run_returns_storage_backed_session(self):
        no_storage = self.store.create(run_id="run_1", step_id="step_1")
        with_storage = self.store.create(run_id="run_1", step_id="step_2")
        storage_path = Path(with_storage["context_dir"]) / "storage_state.json"
        storage_path.write_text("{}", encoding="utf-8")
        self.store.update(
            with_storage["id"],
            {
                "status": "resumed",
                "storage_state_path": str(storage_path),
            },
        )

        latest = self.store.latest_resumable_for_run("run_1")

        assert latest is not None
        self.assertEqual(latest["id"], with_storage["id"])
        self.assertNotEqual(latest["id"], no_storage["id"])


if __name__ == "__main__":
    unittest.main()
