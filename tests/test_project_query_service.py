import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_query_service import (
    build_project_list_view,
    build_single_project_view,
    detect_project_running_server_port,
)
from project_models import ProjectType


class ProjectQueryServiceTest(unittest.TestCase):
    def test_build_project_list_view_omits_command_field(self):
        projects = [
            {
                "name": "demo",
                "path": "/tmp/demo",
                "type": "nextjs",
                "dev_server": {"command": "npm run dev", "port": 5173},
            }
        ]

        result = build_project_list_view(projects, get_server_port=lambda _name: 3000)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["dev_server"]["port"], 3000)
        self.assertNotIn("command", result[0]["dev_server"])
        self.assertTrue(result[0]["dev_server"]["running"])

    def test_build_single_project_view_returns_none_when_missing(self):
        result = build_single_project_view("demo", None, get_server_port=lambda _name: None)
        self.assertIsNone(result)

    def test_build_single_project_view_includes_command(self):
        project = {
            "name": "demo",
            "path": "/tmp/demo",
            "type": "nextjs",
            "dev_server": {"command": "npm run dev", "port": 5173},
        }

        result = build_single_project_view("demo", project, get_server_port=lambda _name: None)

        assert result is not None
        self.assertEqual(result["dev_server"]["port"], 5173)
        self.assertEqual(result["dev_server"]["command"], "npm run dev")
        self.assertFalse(result["dev_server"]["running"])

    def test_detect_project_running_server_port_uses_detected_port(self):
        project = {
            "name": "demo",
            "path": "/tmp/demo",
            "type": "nextjs",
            "dev_server": {"port": 5173},
        }
        detect_mock = lambda _path, _ptype: 3000

        result = detect_project_running_server_port(project, detect_port_for_project=detect_mock)

        self.assertEqual(result, 3000)

    def test_detect_project_running_server_port_uses_configured_open_port(self):
        project = {
            "name": "demo",
            "path": "/tmp/demo",
            "type": "nextjs",
            "dev_server": {"port": 5173},
        }

        with patch("project_query_service.is_local_port_open", return_value=True):
            result = detect_project_running_server_port(
                project,
                detect_port_for_project=lambda _path, _ptype: None,
            )

        self.assertEqual(result, 5173)

    def test_detect_project_running_server_port_returns_none_for_invalid_shape(self):
        self.assertIsNone(
            detect_project_running_server_port(
                None,
                detect_port_for_project=lambda _path, _ptype: 3000,
            )
        )
        self.assertIsNone(
            detect_project_running_server_port(
                {"name": "demo", "type": "nextjs"},
                detect_port_for_project=lambda _path, _ptype: 3000,
            )
        )

    def test_detect_project_running_server_port_passes_project_type(self):
        captured: dict[str, object] = {}

        def detect_stub(path: str, project_type: ProjectType) -> int | None:
            captured["path"] = path
            captured["project_type"] = project_type
            return None

        project = {
            "name": "demo",
            "path": "/tmp/demo",
            "type": "nextjs",
            "dev_server": {"port": 5173},
        }

        with patch("project_query_service.is_local_port_open", return_value=False):
            result = detect_project_running_server_port(
                project,
                detect_port_for_project=detect_stub,
            )

        self.assertIsNone(result)
        self.assertEqual(captured["path"], "/tmp/demo")
        self.assertEqual(captured["project_type"], ProjectType.NEXTJS)


if __name__ == "__main__":
    unittest.main()
