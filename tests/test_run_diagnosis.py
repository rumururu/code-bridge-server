"""Every failed run gets a diagnosis (AGENT_SELF_REPAIR_SPEC §3).

Deterministic class first, always; a model pass only for classes a model can
reason about; the result recorded once as ``run.diagnosis``. A cause the
model does not back with a quoted line is kept but marked unsupported; a
model that fails leaves the class and the error, never an invented cause.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, run_diagnosis  # noqa: E402
from agent.run_diagnosis import classify, diagnose_run  # noqa: E402
from core import database  # noqa: E402


class ClassifyTest(unittest.TestCase):
    def _brief(self, **step):
        return {"interrupted_by_shutdown": False, "failed_step": step}

    def test_interrupted_wins_over_everything(self):
        self.assertEqual(classify({"interrupted_by_shutdown": True, "failed_step": {"exit_code": -15}}), "interrupted")

    def test_shell_exit_is_script_failed(self):
        self.assertEqual(classify(self._brief(type="shell", exit_code=1, output_tail="home tab not found")), "script_failed")

    def test_missing_script_is_its_own_class(self):
        self.assertEqual(classify(self._brief(type="shell", exit_code=127, output_tail="bash: cycle.sh: No such file")), "script_missing")

    def test_timeout(self):
        self.assertEqual(classify(self._brief(type="shell", exit_code=124, timed_out=True)), "timed_out")

    def test_llm_provider_error(self):
        self.assertEqual(classify(self._brief(type="llm", error_message="provider returned 401")), "provider_error")

    def test_unknown(self):
        self.assertEqual(classify(self._brief(type="notify", error_message="???")), "unknown")


class DiagnoseRunTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "diag.db"
        agent_store._agent_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = agent_store.get_agent_store()
        self.agent = self.store.create_agent(name="Cycle", flow_json=[{"id": "c", "type": "llm", "name": "c", "instruction": "x"}])
        self.task = self.store.create_task(title="Cycle", assigned_agent_id=self.agent["id"])

    def _restore(self) -> None:
        agent_store._agent_store = None
        database.DB_PATH = self._original

    def _failed_run(self, stderr="home tab not found", interrupted=False):
        run = self.store.create_run(agent_id=self.agent["id"], task_id=self.task["id"], title="t")
        self.store.update_run_status(run["id"], "running")
        output = {"shell": {"exit_code": -15 if interrupted else 1, "stderr": stderr}}
        if interrupted:
            output["error"] = {"message": "Interrupted: the server stopped while this run was in progress."}
        self.store.create_task_step(task_id=self.task["id"], run_id=run["id"], title="cycle", status="failed",
                                    input={"workflow_step_id": "c", "workflow_type": "shell"}, output=output)
        self.store.update_run_status(run["id"], "failed")
        return run

    def _events(self, run_id):
        return [e for e in self.store.list_events(run_id) if e["event_type"] == "run.diagnosis"]

    async def test_deterministic_only_records_the_class(self):
        run = self._failed_run()
        diagnosis = await diagnose_run(run["id"], use_llm=False)
        self.assertEqual(diagnosis["class"], "script_failed")
        self.assertEqual(diagnosis["llm"], "skipped")
        self.assertEqual(len(self._events(run["id"])), 1)

    async def test_the_model_answer_is_parsed_and_its_evidence_kept(self):
        run = self._failed_run()
        answer = '```json\n{"cause": "onboarding screen is showing", "evidence": ["home tab not found"], "suggested_change": {"kind": "environment", "what": "finish onboarding", "where": "M205N"}, "needs_human": true}\n```'
        with patch.object(run_diagnosis, "_one_shot", new=AsyncMock(return_value=answer)):
            diagnosis = await diagnose_run(run["id"], use_llm=True)
        self.assertEqual(diagnosis["llm"], "ok")
        self.assertEqual(diagnosis["cause"], "onboarding screen is showing")
        self.assertEqual(diagnosis["evidence"], ["home tab not found"])
        self.assertFalse(diagnosis["unsupported"])
        self.assertEqual(diagnosis["suggested_change"]["kind"], "environment")
        self.assertTrue(diagnosis["needs_human"])

    async def test_a_cause_without_evidence_is_marked_unsupported(self):
        run = self._failed_run()
        answer = '```json\n{"cause": "probably the network", "evidence": [], "suggested_change": {"kind": "workflow"}}\n```'
        with patch.object(run_diagnosis, "_one_shot", new=AsyncMock(return_value=answer)):
            diagnosis = await diagnose_run(run["id"], use_llm=True)
        self.assertTrue(diagnosis["unsupported"])

    async def test_a_failed_model_call_leaves_the_class_and_the_error(self):
        run = self._failed_run()
        with patch.object(run_diagnosis, "_one_shot", new=AsyncMock(side_effect=RuntimeError("quota"))):
            diagnosis = await diagnose_run(run["id"], use_llm=True)
        self.assertEqual(diagnosis["class"], "script_failed")
        self.assertEqual(diagnosis["llm"], "failed")
        self.assertIn("quota", diagnosis["error"])
        self.assertIsNone(diagnosis["cause"])

    async def test_an_interrupted_run_never_calls_the_model(self):
        run = self._failed_run(interrupted=True)
        with patch.object(run_diagnosis, "_one_shot", new=AsyncMock()) as one_shot:
            diagnosis = await diagnose_run(run["id"], use_llm=True)
        one_shot.assert_not_awaited()
        self.assertEqual(diagnosis["class"], "interrupted")

    async def test_diagnosis_happens_once_per_run(self):
        run = self._failed_run()
        await diagnose_run(run["id"], use_llm=False)
        again = await diagnose_run(run["id"], use_llm=False)
        self.assertEqual(again["class"], "script_failed")
        self.assertEqual(len(self._events(run["id"])), 1)

    async def test_a_completed_run_is_not_diagnosed(self):
        run = self.store.create_run(agent_id=self.agent["id"], task_id=self.task["id"], title="ok")
        self.store.update_run_status(run["id"], "completed")
        self.assertIsNone(await diagnose_run(run["id"], use_llm=False))


class DeviceContentionTest(unittest.TestCase):
    """A second automation on the same phone is named, not guessed at."""

    def test_contention_outranks_the_timeout(self):
        brief = {"interrupted_by_shutdown": False, "failed_step": {
            "type": "shell", "exit_code": None, "timed_out": True,
            "contention": ["59166 05:19 /bin/bash smalldev_exchange_cycle.sh R59N3035LQL com.x"],
        }}
        self.assertEqual(classify(brief), "device_contention")

    def test_the_executor_names_other_processes_on_the_same_serial(self):
        from agent.shell_step_executor import device_contention
        found = device_contention(
            ["bash", "/x/cycle.sh", "R59N3035LQL", "com.a|com.b"],
            serials={"R59N3035LQL", "24c2b6bcef0d7ece"},
            processes=[
                "59166 05:19 /bin/bash /x/smalldev_exchange_cycle.sh R59N3035LQL com.a",
                "59170 00:01 adb -s R59N3035LQL shell input tap 1 2",
                "60000 00:03 /bin/bash /x/smalldev_exchange_cycle.sh 24c2b6bcef0d7ece com.a",
            ],
        )
        self.assertEqual(len(found), 2)
        self.assertTrue(all("R59N3035LQL" in f for f in found))

    def test_a_script_naming_no_connected_device_reports_nothing(self):
        from agent.shell_step_executor import device_contention
        self.assertEqual(device_contention(["bash", "/x/backup.sh", "nightly"], serials={"R59N3035LQL"}, processes=["1 00:01 something nightly"]), [])
