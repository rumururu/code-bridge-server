"""Coverage for ``AgentTaskRunSink`` decision_marker injection.

TASK_005 introduces a heuristic: if an ``assistant`` or ``output``
reasoning event contains a branching phrase (English ``if`` / ``otherwise``
/ ``depending on`` / ``else`` or Korean ``만약`` / ``결과에 따라`` /
``조건``) and the *next* event from the sink is a ``tool_use``, the sink
inserts a synthetic ``decision_marker`` row before the tool call and
points the tool call's ``parent_event_id`` at that marker.

The six cases below cover:

1. Reasoning without branching keywords — tool_use is appended with no
   ``parent_event_id`` (regression guard for TASK_001).
2. English branching reasoning ``if X then Y`` + next tool_use — marker
   inserted, tool_use parent links to it.
3. Korean branching reasoning ``만약 X면 Y`` + next tool_use — same.
4. Reasoning with branching keywords followed by a non-tool event — the
   cache is preserved until the *next* tool_use arrives (so the marker
   still fires).
5. Two consecutive branching reasonings — only the most recent one
   becomes the marker text (cache replacement).
6. A side-channel event (``error``) between reasoning and tool_use
   invalidates the cached decision so no marker is emitted.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from agent.task_orchestrator import AgentTaskRunSink  # noqa: E402
from core import database  # noqa: E402


def _assistant_text(text: str, *, provider_id: str = "anthropic") -> dict:
    return {
        "type": "assistant",
        "provider_id": provider_id,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


def _tool_use(
    *,
    call_id: str,
    name: str = "shell",
    inputs: dict | None = None,
    provider_id: str = "anthropic",
) -> dict:
    return {
        "type": "assistant",
        "provider_id": provider_id,
        "call_id": call_id,
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": inputs or {"command": "ls"},
                }
            ],
        },
    }


def _error_event(message: str) -> dict:
    return {"type": "error", "message": message}


class DecisionMarkerEmissionTest(unittest.TestCase):
    """End-to-end checks against a fresh AgentStore."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_decision_marker.db"
        agent_store._agent_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()
        run = self.store.create_run(
            project_name="demo",
            title="Decision marker tests",
            provider_id="anthropic",
        )
        self.run_id = run["id"]

    def tearDown(self) -> None:
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _run(self, events: list[dict]) -> list[dict]:
        sink = AgentTaskRunSink(run_id=self.run_id)

        async def drive() -> None:
            for event in events:
                await sink.send_json(event)

        asyncio.run(drive())
        return self.store.list_events(self.run_id)

    # Case 1 — non-branching reasoning -> no marker injection.
    def test_non_branching_reasoning_does_not_inject_marker(self) -> None:
        events = [
            _assistant_text("Calling ls to inspect the workspace."),
            _tool_use(call_id="call-1"),
        ]
        rows = self._run(events)
        types = [row["event_type"] for row in rows]
        self.assertEqual(types, ["assistant", "assistant"])
        # tool_use must NOT be linked to any parent event.
        tool_row = rows[1]
        self.assertEqual(tool_row["call_id"], "call-1")
        self.assertIsNone(tool_row["parent_event_id"])

    # Case 2 — English "if ... then ..." reasoning + tool_use -> marker.
    def test_english_if_reasoning_injects_marker(self) -> None:
        events = [
            _assistant_text(
                "If the package is missing then install it via apt, "
                "otherwise reuse the existing binary."
            ),
            _tool_use(call_id="call-2", name="shell"),
        ]
        rows = self._run(events)
        types = [row["event_type"] for row in rows]
        self.assertEqual(types, ["assistant", "decision_marker", "assistant"])
        reasoning_row = rows[0]
        marker_row = rows[1]
        tool_row = rows[2]
        # Marker carries the condition snippet.
        marker_app = marker_row["app_event"]
        self.assertIn("condition_text", marker_app)
        self.assertIn("if the package", marker_app["condition_text"].lower())
        # Marker hangs off the reasoning event; tool_use hangs off the marker.
        self.assertEqual(marker_row["parent_event_id"], reasoning_row["id"])
        self.assertEqual(tool_row["parent_event_id"], marker_row["id"])

    # Case 3 — Korean "만약 X면 Y" reasoning + tool_use -> marker.
    def test_korean_branching_reasoning_injects_marker(self) -> None:
        events = [
            _assistant_text(
                "만약 캐시가 비어 있다면 빌드를 새로 시작하고, "
                "결과에 따라 다음 스텝을 진행하겠습니다."
            ),
            _tool_use(call_id="call-3", name="shell"),
        ]
        rows = self._run(events)
        types = [row["event_type"] for row in rows]
        self.assertEqual(types, ["assistant", "decision_marker", "assistant"])
        marker_row = rows[1]
        tool_row = rows[2]
        marker_app = marker_row["app_event"]
        self.assertIn("condition_text", marker_app)
        self.assertIn("만약", marker_app["condition_text"])
        self.assertEqual(tool_row["parent_event_id"], marker_row["id"])

    # Case 4 — branching reasoning then a non-tool event then tool_use.
    def test_cache_survives_intervening_non_tool_event(self) -> None:
        """A descriptive (non-tool) assistant turn between the reasoning
        and the tool_use should NOT discard the cached condition. Today
        the sink replaces the cache on the next reasoning event but
        leaves it alone for non-text envelopes that aren't tool_use or
        tool_result.
        """
        # Non-text envelope that isn't a tool_use: e.g. a stop_reason
        # heartbeat with empty content. Our extractor returns "" for it,
        # which triggers the "non-branching reasoning" branch and clears
        # the cache. To exercise the survival path we route through a
        # bare ``background`` envelope instead.
        events = [
            _assistant_text("If the file does not exist, create it."),
            {"type": "background", "note": "heartbeat"},
            _tool_use(call_id="call-4", name="file_write"),
        ]
        rows = self._run(events)
        types = [row["event_type"] for row in rows]
        self.assertEqual(
            types,
            ["assistant", "background", "decision_marker", "assistant"],
        )
        marker_row = rows[2]
        tool_row = rows[3]
        self.assertEqual(tool_row["parent_event_id"], marker_row["id"])

    # Case 5 — two reasonings in a row, only the most recent wins.
    def test_two_consecutive_reasonings_only_latest_wins(self) -> None:
        events = [
            _assistant_text("If we hit a 401, retry with sudo."),
            _assistant_text(
                "Depending on whether the cache exists we either build or skip."
            ),
            _tool_use(call_id="call-5", name="shell"),
        ]
        rows = self._run(events)
        types = [row["event_type"] for row in rows]
        self.assertEqual(
            types,
            ["assistant", "assistant", "decision_marker", "assistant"],
        )
        marker_row = rows[2]
        # The marker must echo the *second* reasoning, not the first.
        self.assertIn(
            "depending on",
            str(marker_row["app_event"]["condition_text"]).lower(),
        )
        # And parent_event_id of the marker must point to the latest
        # reasoning (rows[1]), not rows[0].
        self.assertEqual(marker_row["parent_event_id"], rows[1]["id"])

    # Case 6 — side-channel event clears the cache.
    def test_side_channel_event_invalidates_cache(self) -> None:
        events = [
            _assistant_text("If the build succeeds we move on."),
            _error_event("provider 429"),
            _tool_use(call_id="call-6", name="shell"),
        ]
        rows = self._run(events)
        types = [row["event_type"] for row in rows]
        # No decision_marker — the error between reasoning and tool_use
        # should have discarded the cached condition.
        self.assertEqual(types, ["assistant", "error", "assistant"])
        tool_row = rows[2]
        self.assertIsNone(tool_row["parent_event_id"])


if __name__ == "__main__":
    unittest.main()
