"""Local Dashboard experience mirrors share durable state with the paired API."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from approvals import approval_store
from core import database
from routes import agents, approvals, dashboard_agents, experience, projects, register_api_routers
from routes.deps import verify_api_key


@pytest.fixture
def clients(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "dashboard.db")
    database.init_db()
    agent_store._agent_store = None
    approval_store._approval_store = None
    dashboard = FastAPI()
    dashboard.include_router(dashboard_agents.router)
    app = FastAPI()
    app.include_router(experience.router)
    app.include_router(approvals.router)
    app.include_router(agents.router)
    app.dependency_overrides[verify_api_key] = lambda: "paired"
    yield TestClient(dashboard), TestClient(app)
    agent_store._agent_store = None
    approval_store._approval_store = None


def test_shared_history_review_and_local_boundary(clients):
    dashboard, app = clients
    store = agent_store.get_agent_store()
    first = store.create_run(project_name="one", title="First")
    second = store.create_run(project_name="two", title="Second")
    base = "/api/dashboard/agent"
    assert dashboard.get(f"{base}/history", params={"project_name": "one"}).json()["runs"][0]["id"] == first["id"]
    assert dashboard.get(f"{base}/history", params={"project_name": "one"}).json()["total_count"] == 1
    assert dashboard.get(f"{base}/history", params={"project_name": "two"}).json()["runs"][0]["id"] == second["id"]
    assert dashboard.get(f"{base}/overview").status_code == 200
    assert dashboard.get(f"{base}/action-items").status_code == 200
    assert dashboard.put(f"{base}/runs/{first['id']}/review", json={"reviewed": True}).status_code == 200
    assert dashboard.get(f"{base}/runs/{first['id']}/summary").json()["review"]["actor"] == "desktop_owner"
    assert app.get(f"/api/agent/runs/{first['id']}/summary").json()["review"]["reviewed"] is True
    assert dashboard.get(f"{base}/runs/{first['id']}/summary").json()["review"] == app.get(
        f"/api/agent/runs/{first['id']}/summary").json()["review"]
    assert dashboard.get(f"{base}/history", params={"limit": 0}).status_code == 422
    assert dashboard.get(f"{base}/history", params={"limit": 201}).status_code == 422
    assert dashboard.get(f"{base}/history", params={"since": "invalid"}).status_code == 422
    assert dashboard.get(f"{base}/history", params={"since": "2026-01-02", "until": "2026-01-01"}).status_code == 400
    assert dashboard.get(f"{base}/action-items", params={"cursor": "bad"}).status_code == 400
    assert dashboard.put(f"{base}/runs/missing/review", json={"reviewed": True}).status_code == 404
    assert dashboard.put(f"{base}/runs/{first['id']}/review", json={"reviewed": "invalid"}).status_code == 422
    assert dashboard.get(f"{base}/overview", headers={"CF-Connecting-IP": "203.0.113.1"}).status_code == 403
    foreign = FastAPI()
    register_api_routers(foreign)
    assert not any(route.path.startswith(base) for route in foreign.routes)


def test_artifact_read_failure_preserves_shared_run_summary(clients):
    dashboard, app = clients
    store = agent_store.get_agent_store()
    run = store.create_run(project_name="one", title="Artifact failure")
    dashboard.put(f"/api/dashboard/agent/runs/{run['id']}/review", json={"reviewed": True})
    paths = (f"/api/dashboard/agent/runs/{run['id']}/summary",
             f"/api/agent/runs/{run['id']}/summary")
    normal = dashboard.get(paths[0]).json()
    assert normal["artifacts"] == []
    assert normal["section_errors"] == []
    with patch.object(store, "list_artifacts", side_effect=OSError("artifact store unavailable")):
        responses = (dashboard.get(paths[0]), app.get(paths[1]))
    assert all(response.status_code == 200 for response in responses)
    first, second = (response.json() for response in responses)
    assert first == second
    assert first["artifacts"] is None
    assert first["section_errors"] == ["artifacts"]
    assert first["run"]["id"] == run["id"]
    assert first["verification"] == normal["verification"]
    assert first["review"] == normal["review"]
    assert dashboard.get(paths[0]).json()["section_errors"] == []


def test_every_new_mirror_rejects_tunnel_and_is_unregistered_on_api_listener(clients):
    dashboard, _ = clients
    base = "/api/dashboard/agent"
    paths = [
        ("GET", "/overview", None),
        ("GET", "/action-items", None),
        ("GET", "/history", None),
        ("GET", "/runs/missing/summary", None),
        ("PUT", "/runs/missing/review", {"reviewed": True}),
        ("POST", "/approvals/missing/resume", None),
        ("GET", "/runs/missing/checkpoint", None),
        ("GET", "/runs/missing/artifacts", None),
        ("GET", "/runs/missing/artifacts/missing/content", None),
        ("POST", "/tasks/missing/steps/missing/respond", {"message": "Ready"}),
        ("GET", "/projects", None),
        ("GET", "/projects/missing", None),
        ("POST", "/tasks/missing/start", {"dry_run": True}),
    ]
    foreign = FastAPI()
    register_api_routers(foreign)
    foreign_paths = {route.path for route in foreign.routes}
    for method, suffix, body in paths:
        response = dashboard.request(method, base + suffix, json=body,
                                     headers={"CF-Connecting-IP": "203.0.113.1"})
        assert response.status_code == 403, (method, suffix, response.text)
        assert base + suffix not in foreign_paths


def test_stored_decision_resume_and_artifact_isolation(clients, tmp_path):
    dashboard, app = clients
    base = "/api/dashboard/agent"
    store = agent_store.get_agent_store()
    first = store.create_run(project_name="one", title="First")
    second = store.create_run(project_name="two", title="Second")
    artifact_path = tmp_path / "output.txt"
    artifact_path.write_text("run one", encoding="utf-8")
    artifact = store.add_artifact(run_id=first["id"], kind="file", path=str(artifact_path),
                                  mime_type="text/plain", metadata={})
    assert dashboard.get(f"{base}/runs/{first['id']}/artifacts").json()["artifacts"][0]["id"] == artifact["id"]
    assert dashboard.get(f"{base}/runs/{second['id']}/artifacts").json()["artifacts"] == []
    assert dashboard.get(f"{base}/runs/{second['id']}/artifacts/{artifact['id']}/content").status_code == 404
    assert dashboard.get(f"{base}/runs/{first['id']}/artifacts/{artifact['id']}/content").status_code == 200
    assert dashboard.get(f"{base}/runs/{first['id']}/artifacts/{artifact['id']}/content",
                         params={"max_chars": 0}).status_code == 422
    assert dashboard.get(f"{base}/runs/missing/checkpoint").status_code == 404

    approval = approval_store.get_approval_store().create_request(operation="process.terminal", run_id=first["id"])
    assert dashboard.post(f"{base}/approvals/{approval['id']}/resume").status_code == 409
    approval_store.get_approval_store().create_decision(approval_id=approval["id"], decision="approve_once")
    with patch.object(approvals, "maybe_resume_run_for_decision", new_callable=AsyncMock) as resume:
        result = dashboard.post(f"{base}/approvals/{approval['id']}/resume")
        assert result.status_code == 200
        assert result.json()["decision"]["decision"] == "approve_once"
    assert approval_store.get_approval_store().get_latest_decision(approval["id"])["decision"] == "approve_once"


def test_project_reads_delegate_to_existing_routes(clients):
    dashboard, _ = clients
    base = "/api/dashboard/agent"
    with patch.object(projects, "list_projects_for_current_server", return_value=[{"name": "one"}]), \
         patch.object(projects, "get_project_for_current_server", side_effect=lambda name: {"name": name} if name == "one" else None):
        assert dashboard.get(f"{base}/projects").json() == {"projects": [{"name": "one"}]}
        assert dashboard.get(f"{base}/projects/one").json() == {"name": "one"}
        assert dashboard.get(f"{base}/projects/missing").status_code == 404


def test_decisions_share_http_state_across_app_and_dashboard(clients):
    dashboard, app = clients
    base = "/api/dashboard/agent"
    approval = approval_store.get_approval_store().create_request(operation="test.operation")
    pending = f"{base}/approvals/pending"
    assert approval["id"] in {item["id"] for item in dashboard.get(pending).json()["approvals"]}
    assert approval["id"] in {item["id"] for item in app.get("/api/approvals/pending").json()["approvals"]}
    with patch.object(approvals, "maybe_resume_run_for_decision", new_callable=AsyncMock) as resume:
        resume.return_value = None
        decided = app.post(f"/api/approvals/{approval['id']}/decision", json={"decision": "deny"})
        assert decided.status_code == 200, decided.text
        decision_id = decided.json()["decision"]["id"]
        app_duplicate = app.post(f"/api/approvals/{approval['id']}/decision", json={"decision": "deny"})
        assert app_duplicate.status_code == 200, app_duplicate.text
        assert app_duplicate.json()["decision"]["id"] == decision_id
        duplicate = dashboard.post(f"{base}/approvals/{approval['id']}/decision", json={"decision": "deny"})
        assert duplicate.status_code == 409
        assert duplicate.json()["conflict"] is True
        conflict = dashboard.post(f"{base}/approvals/{approval['id']}/decision", json={"decision": "approve_once"})
        assert conflict.status_code == 409
        assert conflict.json()["conflict"] is True
    assert app.get(f"/api/approvals/{approval['id']}").json()["decision"]["id"] == decision_id
    assert approval["id"] not in {item["id"] for item in dashboard.get(pending).json()["approvals"]}
    assert approval["id"] not in {item["id"] for item in app.get("/api/approvals/pending").json()["approvals"]}
    second = approval_store.get_approval_store().create_request(operation="test.operation")
    assert second["id"] in {item["id"] for item in app.get("/api/approvals/pending").json()["approvals"]}
    with patch.object(approvals, "maybe_resume_run_for_decision", new_callable=AsyncMock) as resume:
        resume.return_value = None
        dashboard_decision = dashboard.post(f"{base}/approvals/{second['id']}/decision",
                                            json={"decision": "deny"})
        assert dashboard_decision.status_code == 200, dashboard_decision.text
    assert second["id"] not in {item["id"] for item in app.get("/api/approvals/pending").json()["approvals"]}
    assert app.get(f"/api/approvals/{second['id']}").json()["decision"]["id"] == dashboard_decision.json()["decision"]["id"]


def test_checkpoint_response_matches_app_and_validates_body(clients):
    dashboard, app = clients
    store = agent_store.get_agent_store()
    task = store.create_task(title="Respond to checkpoint")
    run = store.create_run(task_id=task["id"], title="Waiting")
    step = store.create_task_step(task_id=task["id"], run_id=run["id"],
                                  title="User action", status="waiting_for_user",
                                  output={"checkpoint": {"reason": "manual_handoff", "prompt": "Continue"}})
    store.update_task(task["id"], {"run_id": run["id"], "status": "waiting_for_user"})
    store.update_run_status(run["id"], "waiting_for_user")
    base = f"/api/dashboard/agent/tasks/{task['id']}/steps/{step['id']}/respond"
    assert dashboard.get(f"/api/dashboard/agent/runs/{run['id']}/checkpoint").json() == app.get(
        f"/api/agent/runs/{run['id']}/checkpoint").json()
    assert dashboard.post(base, json={"message": ""}).status_code == 422
    assert dashboard.post(base, json={"message": "Ready", "metadata": []}).status_code == 422
    assert dashboard.post(base.replace(step["id"], "missing"), json={"message": "Ready"}).status_code == 404
    response = dashboard.post(base, json={"message": "Ready", "metadata": {"source": "desktop"},
                                      "remember": False, "resume": False})
    assert response.status_code == 200, response.text
    assert response.json()["response"]["message"] == "Ready"
    assert response.json()["resume_requested"] is False
    assert app.get(f"/api/agent/runs/{run['id']}/checkpoint").json()["checkpoint"]["reason"] == "manual_handoff"
    assert store.get_task_step(step["id"])["status"] == "waiting_for_user"


def test_task_start_typed_delegate_without_execution(clients):
    dashboard, _ = clients
    received = []
    executed = []

    async def fake_start(task_id, body, background_tasks):
        received.append((task_id, body.model_dump(), background_tasks))
        background_tasks.add_task(executed.append, task_id)
        return {"task_id": task_id, "dry_run": body.dry_run}

    with patch.object(dashboard_agents.agents_routes, "start_task", side_effect=fake_start):
        path = "/api/dashboard/agent/tasks/task-one/start"
        assert dashboard.post(path, json={"dry_run": True, "auto_start": False}).json() == {
            "task_id": "task-one", "dry_run": True}
        assert received[0][0] == "task-one"
        assert received[0][1]["auto_start"] is False
        assert len(received[0][2].tasks) == 1
        assert executed == ["task-one"]
        assert dashboard.post(path, json={"capabilities": "invalid"}).status_code == 422
