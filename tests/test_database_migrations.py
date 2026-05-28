import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from core import database


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
                    }.issubset(tables)
                )
            finally:
                database.DB_PATH = original_db_path


if __name__ == "__main__":
    unittest.main()
