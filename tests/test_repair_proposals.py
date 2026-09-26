"""A failed run becomes a repair proposal; only a person applies it (spec §2, §4; ADR-003)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, notification_store, repair_proposals  # noqa: E402
from agent.repair_proposals import (  # noqa: E402
    KIND_ENVIRONMENT, KIND_UNKNOWN, KIND_WORKFLOW, classify_kind, get_repair_proposal_store, on_run_failed,
)
from code_bridge_core.graph_suggest import GraphProposal  # noqa: E402
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402

FLOW = [{"id": "cycle", "type": "llm", "name": "Cycle", "instruction": "run the cycle"}]
FIXED_FLOW = [{"id": "cycle", "type": "llm", "name": "Cycle", "instruction": "run the cycle, then check the home tab"}]


class _Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "repair.db"
        agent_store._agent_store = None
        repair_proposals._store = None
        notification_store._notification_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = agent_store.get_agent_store()
        self.proposals = get_repair_proposal_store()
        self.agent = self.store.create_agent(name="Device cycle", flow_json=FLOW)
        self.task = self.store.create_task(title="Device cycle", assigned_agent_id=self.agent["id"])

    def _restore(self) -> None:
        agent_store._agent_store = None
        repair_proposals._store = None
        notification_store._notification_store = None
        database.DB_PATH = self._original

    def _failed_run(self, stderr="home tab not found"):
        run = self.store.create_run(agent_id=self.agent["id"], task_id=self.task["id"], title="t")
        self.store.update_run_status(run["id"], "running")
        self.store.create_task_step(task_id=self.task["id"], run_id=run["id"], title="cycle", status="failed",
                                    input={"workflow_step_id": "cycle", "workflow_type": "shell"},
                                    output={"shell": {"exit_code": 1, "stderr": stderr}})
        self.store.update_run_status(run["id"], "failed")
        return run

    @staticmethod
    def _diag(kind="workflow", what="check the home tab first", klass="script_failed"):
        return {"class": klass, "cause": "home tab anchor missing", "evidence": ["home tab not found"],
                "suggested_change": {"kind": kind, "what": what, "where": "M205N"}, "needs_human": kind in ("environment", "script_change")}

    def _suggest_ok(self):
        proposal = GraphProposal(summary="Check the home tab before the cycle", flow=FIXED_FLOW)
        return AsyncMock(return_value=([proposal], [], None))


class TriggerTest(_Base):
    async def test_a_failed_run_becomes_a_workflow_proposal_and_a_notification(self):
        run = self._failed_run()
        with patch.object(repair_proposals, "_suggest", new=self._suggest_ok()):
            proposal = await on_run_failed(run["id"], self._diag())
        self.assertEqual(proposal["kind"], KIND_WORKFLOW)
        self.assertTrue(proposal["applicable"])
        self.assertEqual(proposal["flow_json"][0]["instruction"], FIXED_FLOW[0]["instruction"])
        self.assertEqual(proposal["status"], "proposed")
        notes = notification_store.get_notification_store().list_recent(limit=5) if hasattr(notification_store.get_notification_store(), "list_recent") else None
        if notes is not None:
            self.assertTrue(any(n.get("reason") == "repair_proposal" for n in notes))

    async def test_a_second_failure_joins_the_open_proposal_instead_of_making_another(self):
        first = self._failed_run(); second = self._failed_run()
        suggest = self._suggest_ok()
        with patch.object(repair_proposals, "_suggest", new=suggest):
            p1 = await on_run_failed(first["id"], self._diag())
            p2 = await on_run_failed(second["id"], self._diag())
        self.assertEqual(p1["id"], p2["id"])
        self.assertEqual(sorted(self.proposals.get(p1["id"])["seen_runs"]), sorted([first["id"], second["id"]]))
        suggest.assert_awaited_once()

    async def test_an_interrupted_run_makes_no_proposal(self):
        run = self._failed_run()
        with patch.object(repair_proposals, "_suggest", new=self._suggest_ok()) as suggest:
            self.assertIsNone(await on_run_failed(run["id"], {"class": "interrupted"}))
        suggest.assert_not_awaited()

    async def test_an_environment_diagnosis_asks_no_model_and_names_the_person_s_job(self):
        run = self._failed_run()
        with patch.object(repair_proposals, "_suggest", new=self._suggest_ok()) as suggest:
            proposal = await on_run_failed(run["id"], self._diag(kind="environment", what="finish onboarding on the device"))
        suggest.assert_not_awaited()
        self.assertEqual(proposal["kind"], KIND_ENVIRONMENT)
        self.assertFalse(proposal["applicable"])
        self.assertEqual(proposal["human_actions"][0]["what"], "finish onboarding on the device")

    async def test_a_failed_model_call_leaves_an_unknown_proposal_with_the_reason(self):
        run = self._failed_run()
        with patch.object(repair_proposals, "_suggest", new=AsyncMock(side_effect=RuntimeError("quota"))):
            proposal = await on_run_failed(run["id"], self._diag())
        self.assertEqual(proposal["kind"], KIND_UNKNOWN)
        self.assertIn("quota", proposal["summary"])
        self.assertIsNone(proposal["flow_json"])

    async def test_the_daily_cap_stops_new_proposals(self):
        with patch.dict("os.environ", {"CODEBRIDGE_REPAIR_PROPOSALS_PER_DAY": "1"}):
            r1 = self._failed_run()
            with patch.object(repair_proposals, "_suggest", new=self._suggest_ok()):
                p1 = await on_run_failed(r1["id"], self._diag())
            self.proposals.resolve(p1["id"], status="rejected", resolved_by="user", reject_reason="no")
            r2 = self._failed_run()
            with patch.object(repair_proposals, "_suggest", new=self._suggest_ok()) as suggest:
                self.assertIsNone(await on_run_failed(r2["id"], self._diag()))
            suggest.assert_not_awaited()

    async def test_a_rejection_reason_rides_into_the_next_briefing(self):
        r1 = self._failed_run()
        with patch.object(repair_proposals, "_suggest", new=self._suggest_ok()):
            p1 = await on_run_failed(r1["id"], self._diag())
        self.proposals.resolve(p1["id"], status="rejected", resolved_by="user", reject_reason="that step is fine")
        r2 = self._failed_run()
        captured = {}
        async def suggest(agent, briefing, diagnosis):
            captured["briefing"] = briefing
            return [], [], None
        with patch.object(repair_proposals, "_suggest", new=suggest):
            await on_run_failed(r2["id"], self._diag())
        self.assertEqual(captured["briefing"]["previous_rejected_proposal"]["reason"], "that step is fine")

    async def test_expiry_closes_a_week_old_proposal(self):
        run = self._failed_run()
        with patch.object(repair_proposals, "_suggest", new=self._suggest_ok()):
            proposal = await on_run_failed(run["id"], self._diag())
        with database.get_db_connection() as conn:
            conn.execute("UPDATE agent_repair_proposals SET created_at = '2026-01-01T00:00:00+00:00' WHERE id = ?", (proposal["id"],))
            conn.commit()
        expired = self.proposals.expire_stale()
        self.assertEqual([p["id"] for p in expired], [proposal["id"]])
        self.assertEqual(self.proposals.get(proposal["id"])["status"], "expired")


class ClassifyKindTest(unittest.TestCase):
    def test_diagnosis_kind_wins_for_human_kinds(self):
        self.assertEqual(classify_kind({"suggested_change": {"kind": "script_change"}}, {}, [object()]), "script_change")

    def test_a_generated_proposal_is_workflow(self):
        self.assertEqual(classify_kind({"suggested_change": {"kind": "workflow"}}, {}, [object()]), "workflow")

    def test_shell_failure_in_an_external_script_with_no_proposal_is_script_change(self):
        briefing = {"runs": [{"failed_step": {"type": "shell"}}], "registered_scripts": [{"script_id": "s", "origin": "registered"}]}
        self.assertEqual(classify_kind({"suggested_change": {"kind": "workflow"}}, briefing, []), "script_change")


class RoutesTest(_Base):
    def setUp(self) -> None:
        super().setUp()
        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "k"
        self.client = TestClient(app)

    async def _proposal(self):
        run = self._failed_run()
        with patch.object(repair_proposals, "_suggest", new=self._suggest_ok()):
            return await on_run_failed(run["id"], self._diag())

    async def test_accept_applies_the_flow_through_the_save_gate(self):
        proposal = await self._proposal()
        r = self.client.post(f"/api/agent/agents/{self.agent['id']}/repair-proposals/{proposal['id']}/accept", json={})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["proposal"]["status"], "accepted")
        self.assertEqual(self.store.get_agent(self.agent["id"])["flow_json"][0]["instruction"], FIXED_FLOW[0]["instruction"])
        listed = self.client.get(f"/api/agent/agents/{self.agent['id']}/repair-proposals").json()["proposals"]
        self.assertEqual(listed[0]["status"], "accepted")

    async def test_accept_on_a_flow_that_moved_is_a_409_and_the_proposal_is_superseded(self):
        proposal = await self._proposal()
        self.store.update_agent(self.agent["id"], {"flow_json": [{**FLOW[0], "name": "Renamed"}]})
        r = self.client.post(f"/api/agent/agents/{self.agent['id']}/repair-proposals/{proposal['id']}/accept", json={})
        self.assertEqual(r.status_code, 409, r.text)
        self.assertEqual(self.proposals.get(proposal["id"])["status"], "superseded")

    async def test_reject_records_the_reason(self):
        proposal = await self._proposal()
        r = self.client.post(f"/api/agent/agents/{self.agent['id']}/repair-proposals/{proposal['id']}/reject", json={"reason": "wrong step"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["proposal"]["reject_reason"], "wrong step")

    async def test_a_human_only_proposal_cannot_be_applied(self):
        run = self._failed_run()
        with patch.object(repair_proposals, "_suggest", new=self._suggest_ok()):
            proposal = await on_run_failed(run["id"], self._diag(kind="environment", what="tap through onboarding"))
        r = self.client.post(f"/api/agent/agents/{self.agent['id']}/repair-proposals/{proposal['id']}/accept", json={})
        self.assertEqual(r.status_code, 409)

    async def test_the_agent_read_carries_the_open_proposal(self):
        proposal = await self._proposal()
        r = self.client.get(f"/api/agent/agents/{self.agent['id']}")
        agent = r.json().get("agent", r.json())
        self.assertEqual(agent["open_repair_proposal"]["id"], proposal["id"])
