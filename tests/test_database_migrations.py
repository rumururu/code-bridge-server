import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from core import database  # noqa: E402


class DatabaseMigrationTest(unittest.TestCase):
    def test_agent_cockpit_schema_migration_is_idempotent(self):
        original_db_path = database.DB_PATH

        with tempfile.TemporaryDirectory() as tmp_dir:
            database.DB_PATH = Path(tmp_dir) / "code_bridge_test.db"
            try:
                database.init_db()
                database.init_db()

                with sqlite3.connect(database.DB_PATH) as conn:
                    version_row = conn.execute(
                        """
                        SELECT name FROM schema_migrations
                        WHERE version = ?
                        """,
                        (database.AGENT_COCKPIT_SCHEMA_VERSION,),
                    ).fetchone()
                    self.assertEqual(version_row[0], "agent_cockpit_foundation")
                    work_version_row = conn.execute(
                        """
                        SELECT name FROM schema_migrations
                        WHERE version = ?
                        """,
                        (database.WORK_COCKPIT_SCHEMA_VERSION,),
                    ).fetchone()
                    self.assertEqual(work_version_row[0], "work_cockpit_foundation")
                    agents_version_row = conn.execute(
                        """
                        SELECT name FROM schema_migrations
                        WHERE version = 2026053100
                        """
                    ).fetchone()
                    self.assertEqual(agents_version_row[0], "agents_and_memories")
                    browser_sessions_version_row = conn.execute(
                        """
                        SELECT name FROM schema_migrations
                        WHERE version = ?
                        """,
                        (database.BROWSER_SESSIONS_SCHEMA_VERSION,),
                    ).fetchone()
                    self.assertEqual(browser_sessions_version_row[0], "browser_sessions")

                    tables = {
                        row[0]
                        for row in conn.execute(
                            """
                            SELECT name FROM sqlite_master
                            WHERE type = 'table'
                            """
                        ).fetchall()
                    }

                self.assertTrue(
                    {
                        "workspaces",
                        "agent_runs",
                        "agent_tasks",
                        "agent_events",
                        "agent_messages",
                        "agent_permissions",
                        "agent_artifacts",
                        "approval_requests",
                        "approval_decisions",
                        "policy_rules",
                        "audit_events",
                        "preview_sessions",
                        "agent_task_runs",
                        "agent_task_steps",
                        "agent_capabilities",
                        "agent_task_capabilities",
                        "agent_connector_requests",
                        "usage_events",
                        "agents",
                        "agent_memories",
                        "browser_sessions",
                    }.issubset(tables)
                )
            finally:
                database.DB_PATH = original_db_path

    def test_browser_sessions_migration_is_idempotent(self):
        original_db_path = database.DB_PATH

        with tempfile.TemporaryDirectory() as tmp_dir:
            database.DB_PATH = Path(tmp_dir) / "code_bridge_test.db"
            try:
                database.init_db()
                database.init_db()

                with sqlite3.connect(database.DB_PATH) as conn:
                    columns = {
                        row[1]
                        for row in conn.execute(
                            "PRAGMA table_info(browser_sessions)"
                        ).fetchall()
                    }
                    migration_count = conn.execute(
                        "SELECT COUNT(*) FROM schema_migrations WHERE version = ?",
                        (database.BROWSER_SESSIONS_SCHEMA_VERSION,),
                    ).fetchone()[0]
                    index_names = {
                        str(row[1])
                        for row in conn.execute(
                            "PRAGMA index_list(browser_sessions)"
                        ).fetchall()
                    }

                self.assertEqual(migration_count, 1)
                self.assertIn("run_id", columns)
                self.assertIn("workflow_step_id", columns)
                self.assertIn("storage_state_path", columns)
                self.assertIn("expires_at", columns)
                self.assertIn("idx_browser_sessions_run_status", index_names)
                self.assertIn("idx_browser_sessions_task_step", index_names)
            finally:
                database.DB_PATH = original_db_path

    def test_agents_and_memories_migration_is_idempotent(self):
        original_db_path = database.DB_PATH

        with tempfile.TemporaryDirectory() as tmp_dir:
            database.DB_PATH = Path(tmp_dir) / "code_bridge_test.db"
            try:
                database.init_db()
                database.init_db()

                with sqlite3.connect(database.DB_PATH) as conn:
                    pseudo_count = conn.execute(
                        "SELECT COUNT(*) FROM agents WHERE is_pseudo = 1"
                    ).fetchone()[0]
                    migration_count = conn.execute(
                        """
                        SELECT COUNT(*) FROM schema_migrations
                        WHERE version = 2026053100
                        """
                    ).fetchone()[0]
                    run_columns = {
                        row[1]
                        for row in conn.execute(
                            "PRAGMA table_info(agent_runs)"
                        ).fetchall()
                    }
                    task_columns = {
                        row[1]
                        for row in conn.execute(
                            "PRAGMA table_info(agent_tasks)"
                        ).fetchall()
                    }

                self.assertEqual(pseudo_count, 2)
                self.assertEqual(migration_count, 1)
                self.assertIn("agent_id", run_columns)
                self.assertIn("assigned_agent_id", task_columns)
            finally:
                database.DB_PATH = original_db_path

    def test_event_pairing_migration_is_idempotent(self):
        """`_migrate_event_pairing` must run cleanly twice (no dup columns)."""
        original_db_path = database.DB_PATH

        with tempfile.TemporaryDirectory() as tmp_dir:
            database.DB_PATH = Path(tmp_dir) / "code_bridge_test.db"
            try:
                database.init_db()
                database.init_db()

                with sqlite3.connect(database.DB_PATH) as conn:
                    columns = [
                        row[1]
                        for row in conn.execute(
                            "PRAGMA table_info(agent_events)"
                        ).fetchall()
                    ]
                    migration_count = conn.execute(
                        "SELECT COUNT(*) FROM schema_migrations WHERE version = ?",
                        (database.EVENT_PAIRING_COLUMNS_SCHEMA_VERSION,),
                    ).fetchone()[0]
                    index_names = {
                        str(row[1])
                        for row in conn.execute(
                            "PRAGMA index_list(agent_events)"
                        ).fetchall()
                    }

                self.assertEqual(columns.count("call_id"), 1)
                self.assertEqual(columns.count("parent_event_id"), 1)
                self.assertEqual(migration_count, 1)
                self.assertIn("idx_agent_events_run_call", index_names)
                self.assertIn("idx_agent_events_parent", index_names)
            finally:
                database.DB_PATH = original_db_path

    def test_event_pairing_migration_leaves_existing_rows_null(self):
        """Migrating an existing DB must not back-fill — legacy rows stay NULL."""
        original_db_path = database.DB_PATH

        with tempfile.TemporaryDirectory() as tmp_dir:
            database.DB_PATH = Path(tmp_dir) / "code_bridge_test.db"
            try:
                with sqlite3.connect(database.DB_PATH) as conn:
                    database._migrate_legacy_foundation(conn)
                    database._migrate_agent_cockpit_foundation(conn)
                    conn.execute(
                        """
                        INSERT INTO agent_runs (id, status, title)
                        VALUES ('run_legacy', 'completed', 'legacy run')
                        """
                    )
                    conn.execute(
                        """
                        INSERT INTO agent_events
                            (id, run_id, sequence, event_type, app_event_json)
                        VALUES
                            ('evt_legacy', 'run_legacy', 1, 'status', '{"stage":"old"}')
                        """
                    )
                    conn.commit()

                database.init_db()

                with sqlite3.connect(database.DB_PATH) as conn:
                    row = conn.execute(
                        """
                        SELECT call_id, parent_event_id, app_event_json
                        FROM agent_events
                        WHERE id = 'evt_legacy'
                        """
                    ).fetchone()

                self.assertEqual(row, (None, None, '{"stage":"old"}'))
            finally:
                database.DB_PATH = original_db_path

    def test_event_pairing_migration_columns_nullable(self):
        """Schema must allow inserting NULL into both new columns."""
        original_db_path = database.DB_PATH

        with tempfile.TemporaryDirectory() as tmp_dir:
            database.DB_PATH = Path(tmp_dir) / "code_bridge_test.db"
            try:
                database.init_db()
                with sqlite3.connect(database.DB_PATH) as conn:
                    conn.execute(
                        """
                        INSERT INTO agent_runs (id, status, title)
                        VALUES ('run_null', 'queued', 'null run')
                        """
                    )
                    # Null inserts must succeed without constraint errors.
                    conn.execute(
                        """
                        INSERT INTO agent_events (
                            id, run_id, sequence, event_type,
                            call_id, parent_event_id
                        ) VALUES ('evt_null', 'run_null', 1, 'status', NULL, NULL)
                        """
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT call_id, parent_event_id FROM agent_events "
                        "WHERE id = 'evt_null'"
                    ).fetchone()
                self.assertEqual(row, (None, None))
            finally:
                database.DB_PATH = original_db_path

    def test_agents_and_memories_migration_backfills_existing_rows(self):
        original_db_path = database.DB_PATH

        with tempfile.TemporaryDirectory() as tmp_dir:
            database.DB_PATH = Path(tmp_dir) / "code_bridge_test.db"
            try:
                with sqlite3.connect(database.DB_PATH) as conn:
                    database._migrate_legacy_foundation(conn)
                    database._migrate_agent_cockpit_foundation(conn)
                    database._migrate_work_cockpit_foundation(conn)
                    database._migrate_task_schedules(conn)
                    conn.execute(
                        """
                        INSERT INTO agent_runs (id, status, title)
                        VALUES ('run_existing', 'completed', 'existing run')
                        """
                    )
                    conn.execute(
                        """
                        INSERT INTO agent_tasks (id, title, status)
                        VALUES ('task_existing', 'existing task', 'queued')
                        """
                    )
                    conn.commit()

                database.init_db()

                with sqlite3.connect(database.DB_PATH) as conn:
                    run_row = conn.execute(
                        """
                        SELECT title, status, agent_id
                        FROM agent_runs
                        WHERE id = 'run_existing'
                        """
                    ).fetchone()
                    task_row = conn.execute(
                        """
                        SELECT title, status, assigned_agent_id
                        FROM agent_tasks
                        WHERE id = 'task_existing'
                        """
                    ).fetchone()
                    pseudo_ids = {
                        row[0]
                        for row in conn.execute(
                            "SELECT id FROM agents WHERE is_pseudo = 1"
                        ).fetchall()
                    }

                self.assertEqual(run_row, ("existing run", "completed", "agent_legacy_chat"))
                self.assertEqual(task_row, ("existing task", "queued", "agent_legacy_chat"))
                self.assertEqual(pseudo_ids, {"agent_legacy_chat", "agent_adhoc_dev"})
            finally:
                database.DB_PATH = original_db_path


if __name__ == "__main__":
    unittest.main()
