import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from projects import ProjectManager


class ProjectManagerCoreTest(unittest.TestCase):
    def test_get_project_uses_injected_db_factory(self):
        fake_db = MagicMock()
        fake_db.get.return_value = {
            "name": "demo",
            "path": "/tmp/demo",
            "type": "nextjs",
            "dev_server": {"command": "npm run dev", "port": 5173},
        }

        manager = ProjectManager(_project_db_factory=lambda: fake_db)
        with patch.object(manager, "get_server_port", return_value=3000), \
             patch.object(manager, "is_server_running", return_value=True):
            result = manager.get_project("demo")

        fake_db.get.assert_called_once_with("demo")
        assert result is not None
        self.assertEqual(result["name"], "demo")
        self.assertEqual(result["dev_server"]["port"], 3000)
        self.assertTrue(result["dev_server"]["running"])

    def test_get_all_projects_uses_injected_db_factory(self):
        fake_db = MagicMock()
        fake_db.get_all.return_value = [
            {"name": "a", "path": "/tmp/a", "type": "nextjs", "dev_server": {"port": 3000}},
            {"name": "b", "path": "/tmp/b", "type": "flutter", "dev_server": None},
        ]
        fake_db.get.side_effect = lambda name: fake_db.get_all.return_value[0] if name == "a" else fake_db.get_all.return_value[1]
        manager = ProjectManager(_project_db_factory=lambda: fake_db)

        with patch("project_manager.list_listening_processes", return_value={}), \
             patch("project_manager.list_process_cwds", return_value={}), \
             patch("project_manager.detect_port_for_project", side_effect=lambda p, t, **kw: 3000 if "a" in p else None):
            result = manager.get_all_projects()

        fake_db.get_all.assert_called_once_with()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "a")
        self.assertEqual(result[0]["dev_server"]["port"], 3000)
        self.assertEqual(result[1]["name"], "b")
        self.assertFalse(result[1]["dev_server"]["running"])

    def test_detect_running_server_port_uses_injected_db_factory(self):
        fake_db = MagicMock()
        fake_db.get.return_value = {
            "name": "demo",
            "path": "/tmp/demo",
            "type": "nextjs",
            "dev_server": {"port": 5173},
        }
        manager = ProjectManager(_project_db_factory=lambda: fake_db)

        with patch("project_manager.detect_project_running_server_port", return_value=5173) as mock_detect:
            result = manager.detect_running_server_port("demo")

        fake_db.get.assert_called_once_with("demo")
        mock_detect.assert_called_once()
        _, kwargs = mock_detect.call_args
        self.assertTrue(callable(kwargs.get("detect_port_for_project")))
        self.assertEqual(result, 5173)


if __name__ == "__main__":
    unittest.main()
