"""Agent experience contract and API boundary."""

import tempfile
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from approvals import approval_store
from core import database
from core.database import get_db_connection
from routes import agents, approvals, experience
from routes.deps import verify_api_key


def test_overview_keeps_independent_sections_when_schedule_read_fails():
    from agent import experience_service
    from unittest.mock import Mock
    store = Mock()
    store.list_runs.return_value = [{"id": "run"}]
    schedules = Mock()
    schedules.list_all.side_effect = RuntimeError("unavailable")
    with patch.object(experience_service, "get_agent_store", return_value=store), \
         patch.object(experience_service, "get_schedule_store", return_value=schedules), \
         patch.object(experience_service, "action_items", return_value={"total_count": 4}):
        result = experience_service.overview()
    assert result["running"] == [{"id": "run"}]
    assert result["recent"] == [{"id": "run"}]
    assert result["action_count"] == 4
    assert result["section_errors"] == ["next_schedules"]


def test_history_summary_review_and_auth():
    original = database.DB_PATH
    with tempfile.TemporaryDirectory() as directory:
        database.DB_PATH = Path(directory) / "test.db"
        database.init_db()
        agent_store._agent_store = None
        approval_store._approval_store = None
        try:
            app = FastAPI()
            app.include_router(experience.router)
            app.include_router(approvals.router)
            app.include_router(agents.router)
            client = TestClient(app)
            assert client.get("/api/agent/history").status_code == 401
            app.dependency_overrides[verify_api_key] = lambda: "test"
            store = agent_store.get_agent_store()
            ids = [store.create_run(project_name="demo", title=f"Run {i}")["id"] for i in range(205)]
            first = client.get("/api/agent/history?limit=200").json()
            assert first["total_count"] == 205
            assert len(first["runs"]) == 200
            second = client.get("/api/agent/history", params={"limit": 200, "cursor": first["next_cursor"]}).json()
            assert len(second["runs"]) == 5
            assert len({run["id"] for run in first["runs"] + second["runs"]}) == 205
            assert client.get("/api/agent/history?cursor=bad").status_code == 400
            assert client.get("/api/agent/history?cursor=____").status_code == 400
            assert client.get("/api/agent/history?project_name=other").json()["total_count"] == 0
            assert client.get("/api/agent/history?since=2026-01-01&until=2025-01-01").status_code == 400
            assert client.get("/api/agent/history?since=2000-01-01&until=2100-01-01").json()["total_count"] == 205
            stable_first = client.get("/api/agent/history?limit=1").json()
            with get_db_connection() as conn:
                conn.execute("UPDATE agent_runs SET updated_at = '2100-01-01' WHERE id = ?",
                             (stable_first["runs"][0]["id"],))
                conn.commit()
            stable_second = client.get("/api/agent/history", params={"limit": 200,
                "cursor": stable_first["next_cursor"]}).json()
            assert stable_first["runs"][0]["id"] not in {run["id"] for run in stable_second["runs"]}
            assert len(stable_second["runs"]) == 200

            run_id = ids[0]
            forged_response = client.post(f"/api/agent/runs/{run_id}/event", json={
                "event_type": "preflight.completed", "app_event": {"results": [{"passed": True}]}})
            assert forged_response.status_code == 403
            summary = client.get(f"/api/agent/runs/{run_id}/summary").json()
            assert summary["verification"]["status"] == "unknown"
            assert summary["review"]["reviewed"] is False
            store.append_event(run_id=run_id, event_type="preflight.completed",
                                        app_event={"results": [{"command": "fake", "passed": True}]})
            assert client.get(f"/api/agent/runs/{run_id}/summary").json()["verification"]["status"] == "unknown"
            trusted = store.append_event(run_id=run_id, event_type="preflight.completed",
                                         app_event={"results": [{"command": "test", "passed": False}]})
            with get_db_connection() as conn:
                conn.execute("INSERT INTO agent_preflight_evidence (event_id, run_id) VALUES (?, ?)",
                             (trusted["id"], run_id))
                conn.commit()
            assert client.get(f"/api/agent/runs/{run_id}/summary").json()["verification"]["status"] == "failed"
            assert len(client.get(f"/api/agent/runs/{run_id}/summary").json()["verification"]["evidence"]) == 1
            latest = store.append_event(run_id=run_id, event_type="preflight.completed",
                                        app_event={"results": [{"command": "retry", "passed": True}]})
            with get_db_connection() as conn:
                conn.execute("INSERT INTO agent_preflight_evidence (event_id, run_id) VALUES (?, ?)",
                             (latest["id"], run_id))
                conn.commit()
            verification = client.get(f"/api/agent/runs/{run_id}/summary").json()["verification"]
            assert verification["status"] == "passed"
            assert len(verification["evidence"]) == 2
            artifact = store.add_artifact(run_id=run_id, kind="file", path="report.txt",
                                          mime_type="text/plain", metadata={"name": "report.txt"})
            assert client.get(f"/api/agent/runs/{run_id}/summary").json()["artifacts"][0]["path"] == "report.txt"
            assert client.put(f"/api/agent/runs/{run_id}/review", json={"reviewed": True}).json()["review"]["reviewed"]
            agent_store._agent_store = None
            assert client.get(f"/api/agent/runs/{run_id}/summary").json()["review"]["reviewed"]
            assert client.put(f"/api/agent/runs/{run_id}/review", json={"reviewed": False}).json()["review"]["reviewed"] is False
            approval = approval_store.get_approval_store().create_request(operation="process.terminal", run_id=run_id)
            assert client.post(f"/api/approvals/{approval['id']}/resume").status_code == 409
            with get_db_connection() as conn:
                conn.execute("INSERT INTO agent_tasks (id, title, status, run_id) VALUES (?, ?, ?, ?)",
                             ("task_experience", "Task", "waiting_for_user", run_id))
                conn.execute("UPDATE agent_runs SET task_id = ? WHERE id = ?", ("task_experience", run_id))
                conn.execute("""INSERT INTO agent_task_steps
                    (id, task_id, run_id, sequence, status, output_json)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    ("step_experience", "task_experience", run_id, 1, "waiting_for_user",
                     '{"checkpoint":{"reason":"approval_required","approval_id":"' + approval["id"] + '"}}'))
                conn.commit()
            actions = client.get("/api/agent/action-items").json()
            assert actions["total_count"] == 1
            assert actions["items"][0]["approval_id"] == approval["id"]
            assert client.get("/api/agent/overview").json()["action_count"] == 1
            approval_store.get_approval_store().create_decision(approval_id=approval["id"], decision="approve_once")
            with patch.object(approvals, "maybe_resume_run_for_decision", new_callable=AsyncMock) as resume:
                resume.return_value = None
                recovered = client.post(f"/api/approvals/{approval['id']}/resume").json()
                resume.assert_awaited_once()
                repeated = client.post(f"/api/approvals/{approval['id']}/decision", json={"decision": "approve_once"})
                assert repeated.status_code == 409
                assert repeated.json()["conflict"] is True
            assert recovered["resume_status"] == "recovery_required"
            assert recovered["decision"]["decision"] == "approve_once"
            actions = client.get("/api/agent/action-items").json()
            assert actions["total_count"] == 1
            assert actions["items"][0]["status"] == "recovery_required"
            with get_db_connection() as conn:
                conn.executemany("""INSERT INTO approval_requests
                    (id, operation, risk_level, actor_json, details_json)
                    VALUES (?, 'process.terminal', 'medium', '{}', '{}')""",
                    [(f"bulk_{i:03d}",) for i in range(205)])
                conn.commit()
            page_one = client.get("/api/agent/action-items?limit=200").json()
            page_two = client.get("/api/agent/action-items", params={"limit": 200, "cursor": page_one["next_cursor"]}).json()
            assert page_one["total_count"] == page_two["total_count"] == 206
            assert len(page_one["items"]) == 200 and len(page_two["items"]) == 6
            assert len({item["id"] for item in page_one["items"] + page_two["items"]}) == 206
            assert client.get("/api/agent/runs/missing/summary").status_code == 404
            assert client.put("/api/agent/runs/missing/review", json={"reviewed": True}).status_code == 404
        finally:
            agent_store._agent_store = None
            approval_store._approval_store = None
            database.DB_PATH = original
