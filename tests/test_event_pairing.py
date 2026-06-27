"""Coverage for agent_events call_id / parent_event_id pairing columns.

These tests focus on the migration + storage + codex normalization seams
introduced for v2 flow visualization (TASK_001). The eight cases here
match the brief: column existence, migration idempotency, partial
indexes, nullable storage, call_id round-trip, tool_use ↔ tool_result
pairing, backward-compatible append_event signature, and codex
``_normalize_event`` carrying ``call_id``.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from core import database  # noqa: E402
from llm.codex_session import CodexSession  # noqa: E402


def _columns(conn: sqlite3.Connection, table: str) -> dict[str, sqlite3.Row]:
    """Return column metadata keyed by column name."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1]: row for row in rows}


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    return {str(row[1]) for row in rows}


class EventPairingMigrationTest(unittest.TestCase):
    """Direct migration-level checks (no AgentStore involvement)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_event_pairing.db"

    def tearDown(self) -> None:
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # Case 1
    def test_migration_adds_call_id_and_parent_event_id_columns(self) -> None:
        database.init_db()
        with sqlite3.connect(database.DB_PATH) as conn:
            cols = _columns(conn, "agent_events")
        self.assertIn("call_id", cols)
        self.assertIn("parent_event_id", cols)
        # Both columns must be TEXT and nullable (notnull == 0).
        self.assertEqual(cols["call_id"][2].upper(), "TEXT")
        self.assertEqual(cols["parent_event_id"][2].upper(), "TEXT")
        self.assertEqual(cols["call_id"][3], 0)
        self.assertEqual(cols["parent_event_id"][3], 0)

    # Case 2
    def test_migration_is_idempotent(self) -> None:
        database.init_db()
        # Second run must not raise (no duplicate column / index errors)
        database.init_db()
        with sqlite3.connect(database.DB_PATH) as conn:
            cols = _columns(conn, "agent_events")
            applied = conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = ?",
                (database.EVENT_PAIRING_COLUMNS_SCHEMA_VERSION,),
            ).fetchone()[0]
        # Column should appear exactly once.
        self.assertEqual(
            sum(1 for name in cols if name == "call_id"),
            1,
        )
        self.assertEqual(applied, 1)

    # Case 3
    def test_partial_indexes_present(self) -> None:
        database.init_db()
        with sqlite3.connect(database.DB_PATH) as conn:
            names = _index_names(conn, "agent_events")
            # Verify partial-index filter is applied.
            run_call_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'idx_agent_events_run_call'"
            ).fetchone()
            parent_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'idx_agent_events_parent'"
            ).fetchone()
        self.assertIn("idx_agent_events_run_call", names)
        self.assertIn("idx_agent_events_parent", names)
        self.assertIsNotNone(run_call_sql)
        self.assertIsNotNone(parent_sql)
        self.assertIn("call_id IS NOT NULL", run_call_sql[0])
        self.assertIn("parent_event_id IS NOT NULL", parent_sql[0])


class EventPairingStorageTest(unittest.TestCase):
    """End-to-end AgentStore round-trips against the new columns."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_event_pairing.db"
        agent_store._agent_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()
        self.run = self.store.create_run(project_name="demo", title="Pairing")

    def tearDown(self) -> None:
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # Case 4
    def test_new_columns_nullable_when_unset(self) -> None:
        event = self.store.append_event(
            run_id=self.run["id"],
            event_type="status",
            app_event={"stage": "queued"},
            call_id=None,
            parent_event_id=None,
        )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertIsNone(event["call_id"])
        self.assertIsNone(event["parent_event_id"])
        # list_events returns the same shape.
        rows = self.store.list_events(self.run["id"])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["call_id"])
        self.assertIsNone(rows[0]["parent_event_id"])

    # Case 5
    def test_call_id_round_trips_through_get_and_list(self) -> None:
        event = self.store.append_event(
            run_id=self.run["id"],
            event_type="tool_use",
            call_id="call_abc123",
            app_event={"name": "browser_navigate"},
        )
        assert event is not None
        self.assertEqual(event["call_id"], "call_abc123")
        fetched = self.store.get_event(event["id"])
        assert fetched is not None
        self.assertEqual(fetched["call_id"], "call_abc123")
        rows = self.store.list_events(self.run["id"], after_sequence=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["call_id"], "call_abc123")

    # Case 6
    def test_tool_use_and_tool_result_pair_by_call_id(self) -> None:
        use = self.store.append_event(
            run_id=self.run["id"],
            event_type="tool_use",
            call_id="call_pair_1",
            app_event={"name": "browser_navigate"},
        )
        result = self.store.append_event(
            run_id=self.run["id"],
            event_type="tool_result",
            call_id="call_pair_1",
            parent_event_id=use["id"] if use else None,
            app_event={"output": "ok"},
        )
        assert use is not None and result is not None
        rows = self.store.list_events(self.run["id"])
        by_call: dict[str, list[dict]] = {}
        for row in rows:
            cid = row.get("call_id")
            if cid is None:
                continue
            by_call.setdefault(cid, []).append(row)
        self.assertIn("call_pair_1", by_call)
        pair = by_call["call_pair_1"]
        self.assertEqual(len(pair), 2)
        types = sorted(row["event_type"] for row in pair)
        self.assertEqual(types, ["tool_result", "tool_use"])
        # parent_event_id of the result must reference the originating use.
        result_row = next(row for row in pair if row["event_type"] == "tool_result")
        self.assertEqual(result_row["parent_event_id"], use["id"])

    # Case 7
    def test_legacy_append_event_call_sites_still_work(self) -> None:
        """Existing callers (no call_id / parent_event_id) must keep working."""
        event = self.store.append_event(
            run_id=self.run["id"],
            event_type="status",
            provider_id="openai",
            app_event={"stage": "planning"},
        )
        assert event is not None
        self.assertIsNone(event["call_id"])
        self.assertIsNone(event["parent_event_id"])
        # And get_event still surfaces the new keys as None.
        fetched = self.store.get_event(event["id"])
        assert fetched is not None
        self.assertIn("call_id", fetched)
        self.assertIn("parent_event_id", fetched)
        self.assertIsNone(fetched["call_id"])
        self.assertIsNone(fetched["parent_event_id"])


class CodexNormalizeEventCallIdTest(unittest.TestCase):
    """Codex ``_normalize_event`` must surface call_id under any of the 3 keys."""

    def _session(self) -> CodexSession:
        return CodexSession(
            project_path="/tmp/code-bridge",
            model="gpt-5",
            sandbox_mode="workspace-write",
        )

    # Case 8 (a) — id key
    def test_tool_use_with_id_key(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "tool_use",
                "id": "call_id_key",
                "name": "shell",
                "input": {"command": "ls"},
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["call_id"], "call_id_key")
        block = normalized["message"]["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "call_id_key")
        self.assertEqual(block["name"], "shell")
        self.assertEqual(block["input"], {"command": "ls"})

    # Case 8 (b) — call_id key
    def test_tool_use_with_call_id_key(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "function_call",
                "call_id": "call_via_call_id",
                "function": "browser_navigate",
                "arguments": {"url": "https://example.com"},
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["call_id"], "call_via_call_id")
        block = normalized["message"]["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "call_via_call_id")
        self.assertEqual(block["name"], "browser_navigate")

    # Case 8 (c) — tool_call_id key
    def test_tool_use_with_tool_call_id_key(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "tool_use",
                "tool_call_id": "tcid_42",
                "name": "shell",
                "input": {},
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["call_id"], "tcid_42")

    def test_tool_use_without_any_id_key_yields_none(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "tool_use",
                "name": "shell",
                "input": {},
            }
        )
        assert normalized is not None
        self.assertIsNone(normalized["call_id"])

    def test_tool_result_normalization_carries_call_id(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "tool_result",
                "call_id": "call_pair_1",
                "result": "done",
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["call_id"], "call_pair_1")
        block = normalized["message"]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "call_pair_1")
        self.assertEqual(block["content"], "done")

    def test_tool_result_falls_back_to_tool_use_id_key(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "function_result",
                "tool_use_id": "tu_99",
                "output": "ok",
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["call_id"], "tu_99")
        block = normalized["message"]["content"][0]
        self.assertEqual(block["tool_use_id"], "tu_99")


if __name__ == "__main__":
    unittest.main()
