"""Bookmarkable console pages preserve the local listener and legacy editors."""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routes import dashboard, register_api_routers


@pytest.mark.parametrize('path', ['/dashboard', '/inbox', '/projects', '/agents', '/settings', '/runs/taskless-run'])
def test_shell_route_and_tunnel_boundary(path):
    app = FastAPI()
    app.include_router(dashboard.router)
    client = TestClient(app)
    result = client.get(path)
    assert result.status_code == 200
    assert 'id="agentsFrame"' in result.text
    assert 'id="settingsFrame"' in result.text
    assert 'const ApprovalReview' in result.text
    assert client.get(path, headers={'CF-Connecting-IP':'203.0.113.5'}).status_code == 403


def test_legacy_forms_and_management_groups_remain_reachable():
    app = FastAPI()
    app.include_router(dashboard.router)
    client = TestClient(app)
    agents = client.get('/agents?embedded=1').text
    assert 'id="agentName"' in agents
    assert 'window.dashboardReady = reloadAll()' in agents
    assert 'window.dashboardNavigate' in agents
    settings = client.get('/settings?embedded=1').text
    for control in ['folderList', 'passwordModal', 'pairingModal', 'serverLogContent', 'deviceList', 'autostartToggle']:
        assert f'id="{control}"' in settings
    assert 'data-management-hidden' in settings
    assert client.get('/agents?embedded=1', headers={'CF-Connecting-IP':'203.0.113.5'}).status_code == 403


def test_console_pages_are_never_registered_on_external_api():
    app = FastAPI()
    register_api_routers(app)
    paths = {route.path for route in app.routes}
    assert not paths.intersection({'/dashboard','/inbox','/projects','/agents','/settings','/runs/{run_id}'})
