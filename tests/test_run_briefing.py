"""The failure briefing reaches every Configurator entry point (AGENT_SELF_REPAIR_SPEC §1).

An agent that had failed six cycles in a row, with a diagnosis step that had
written the cause each time, reached the Configurator as a blank intent box:
`build_suggest_prompt` took the agent's name, its prompt, the flow and what
the person typed, and nothing about what the agent had done. These pin the
briefing's facts, its masking, and that a healthy agent's prompt is unchanged.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from agent.run_briefing import build_run_briefing, failure_briefing_block  # noqa: E402
from code_bridge_core import graph_suggest  # noqa: E402
from core import database  # noqa: E402

FLOW = [
    {"id": "cycle", "type": "llm", "name": "Cycle", "instruction": "run"},
    {"id": "diagnose", "type": "llm", "name": "Diagnose", "instruction": "why"},
]


class RunBriefingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "briefing.db"
        agent_store._agent_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = agent_store.get_agent_store()
        self.agent = self.store.create_agent(name="Device cycle", flow_json=FLOW)
        self.task = self.store.create_task(title="Device cycle", assigned_agent_id=self.agent["id"])

    def _restore(self) -> None:
        agent_store._agent_store = None
        database.DB_PATH = self._original

    def _failed_run(self, *, stderr: str, exit_code: int = 1, diagnosis: str | None = None):
        run = self.store.create_run(agent_id=self.agent["id"], task_id=self.task["id"], title="Run task: Device cycle")
        self.store.update_run_status(run["id"], "running")
        self.store.create_task_step(
            task_id=self.task["id"], run_id=run["id"], title="M205N 사이클", status="failed",
            input={"workflow_step_id": "cycle", "workflow_type": "shell"},
            output={"shell": {"status": "failed", "exit_code": exit_code, "timed_out": False, "stderr": stderr}},
        )
        if diagnosis is not None:
            self.store.create_task_step(
                task_id=self.task["id"], run_id=run["id"], title="실패 진단", status="completed",
                input={"workflow_step_id": "diagnose", "workflow_type": "llm"},
                output={"message": "Workflow step completed.", "result": diagnosis},
            )
        self.store.update_run_status(run["id"], "failed")
        return run

    def test_the_failed_step_and_its_last_line_reach_the_prompt(self):
        self._failed_run(
            stderr="arg: -p\n[smalldev-cycle] R59N3035LQL home tab not found\n",
            diagnosis="홈 탭 앵커를 못 찾음 — 온보딩 화면일 가능성",
        )
        briefing = build_run_briefing(self.agent["id"])
        prompt = graph_suggest.build_suggest_prompt(
            agent_name="Device cycle", agent_system_prompt="", flow=FLOW, intent="fix it", briefing=briefing
        )
        self.assertIn("M205N 사이클", prompt)
        self.assertIn("exit=1", prompt)
        self.assertIn("home tab not found", prompt)
        self.assertIn("홈 탭 앵커를 못 찾음", prompt)
        self.assertEqual(briefing["streak"], 1)

    def test_no_failures_means_the_prompt_is_byte_for_byte_unchanged(self):
        run = self.store.create_run(agent_id=self.agent["id"], task_id=self.task["id"], title="ok")
        self.store.update_run_status(run["id"], "completed")
        briefing = build_run_briefing(self.agent["id"])
        self.assertEqual(briefing["runs"], [])
        self.assertEqual(failure_briefing_block(briefing), "")
        with_b = graph_suggest.build_suggest_prompt(agent_name="a", agent_system_prompt="", flow=FLOW, intent="x", briefing=briefing)
        without = graph_suggest.build_suggest_prompt(agent_name="a", agent_system_prompt="", flow=FLOW, intent="x")
        self.assertEqual(with_b, without)

    def test_streak_counts_consecutive_failures_and_the_same_step(self):
        self._failed_run(stderr="home tab not found")
        self._failed_run(stderr="home tab not found")
        self._failed_run(stderr="home tab not found")
        briefing = build_run_briefing(self.agent["id"])
        self.assertEqual(briefing["streak"], 3)
        self.assertTrue(briefing["same_step_as_previous"])
        self.assertEqual(len(briefing["runs"]), 3)

    def test_a_secret_in_the_output_tail_is_masked(self):
        self._failed_run(stderr="auth failed for key AKIAIOSFODNN7EXAMPLE at step 3")
        briefing = build_run_briefing(self.agent["id"])
        tail = briefing["runs"][0]["failed_step"]["output_tail"]
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", tail)
        self.assertIn("at step 3", tail)

    def test_a_restart_kill_is_marked_and_does_not_count_toward_the_streak(self):
        run = self.store.create_run(agent_id=self.agent["id"], task_id=self.task["id"], title="cut")
        self.store.update_run_status(run["id"], "running")
        self.store.create_task_step(
            task_id=self.task["id"], run_id=run["id"], title="M205N 사이클", status="failed",
            input={"workflow_step_id": "cycle", "workflow_type": "shell"},
            output={"shell": {"exit_code": -15}, "error": {"message": "Interrupted: the server stopped while this run was in progress."}},
        )
        self.store.update_run_status(run["id"], "failed")
        briefing = build_run_briefing(self.agent["id"])
        self.assertTrue(briefing["runs"][0]["interrupted_by_shutdown"])
        self.assertEqual(briefing["streak"], 0)
        self.assertIn("interrupted_by_shutdown", failure_briefing_block(briefing))

    def test_building_a_briefing_writes_nothing(self):
        self._failed_run(stderr="x")
        before = [(r["id"], r["status"], r["updated_at"]) for r in self.store.list_runs(agent_id=self.agent["id"])]
        build_run_briefing(self.agent["id"])
        after = [(r["id"], r["status"], r["updated_at"]) for r in self.store.list_runs(agent_id=self.agent["id"])]
        self.assertEqual(before, after)
