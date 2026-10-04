"""The Dashboard illustration is served only by the local console."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes import dashboard, register_api_routers


ASSET_PATH = "/dashboard/assets/server-device-v1.png"
SOURCE_PNG = SERVER_DIR.parent / "docs/agent-collaboration/product-experience/preview/assets/server-device-v1.png"


def test_dashboard_visual_asset_content_and_local_access():
    app = FastAPI()
    app.include_router(dashboard.router)
    with TestClient(app) as client:
        response = client.get(ASSET_PATH)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content == SOURCE_PNG.read_bytes()

        denied = client.get(ASSET_PATH, headers={"CF-Connecting-IP": "203.0.113.5"})
        assert denied.status_code == 403


def test_dashboard_visual_asset_is_absent_from_external_api():
    app = FastAPI()
    register_api_routers(app)
    assert ASSET_PATH not in {route.path for route in app.routes}
