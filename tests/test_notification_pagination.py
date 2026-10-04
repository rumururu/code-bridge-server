"""Notification pagination keeps filtered rows and run links intact."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import notification_store
from core import database
from routes import agents
from routes.deps import verify_api_key


def test_filtered_offset_and_run_id(tmp_path):
    original = database.DB_PATH
    database.DB_PATH = tmp_path / "notifications.db"
    try:
        database.init_db()
        store = notification_store.NotificationStore()
        older = store.create(title="older", agent_id="wanted", run_id="run_old")
        store.create(title="other", agent_id="other", run_id="run_other")
        store.create(title="read", agent_id="wanted", run_id="run_read")
        store.mark_read(store.list_notifications(agent_id="wanted")[0]["id"])
        newer = store.create(title="newer", agent_id="wanted", run_id="run_new")
        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test"
        response = TestClient(app).get("/api/agent/notifications", params={
            "unread_only": True, "agent_id": "wanted", "offset": 1, "limit": 1,
        })
        assert response.status_code == 200
        assert response.json()["notifications"] == [older]
        assert response.json()["notifications"][0]["run_id"] == "run_old"
        assert newer["run_id"] == "run_new"
        assert TestClient(app).get("/api/agent/notifications?offset=-1").status_code == 422
    finally:
        database.DB_PATH = original
