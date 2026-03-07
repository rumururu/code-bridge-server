import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_view import build_project_view


class ProjectViewTest(unittest.TestCase):
    def test_build_project_view_includes_command_by_default(self):
        project = {
            "name": "demo",
            "path": "/tmp/demo",
            "type": "nextjs",
            "dev_server": {"port": 5173, "command": "npm run dev"},
        }

        result = build_project_view(project, detected_port=3000)

        self.assertEqual(result["name"], "demo")
        self.assertEqual(result["path"], "/tmp/demo")
        self.assertEqual(result["type"], "nextjs")
        self.assertEqual(result["dev_server"]["port"], 3000)
        self.assertEqual(result["dev_server"]["command"], "npm run dev")
        self.assertTrue(result["dev_server"]["running"])

    def test_build_project_view_can_omit_command(self):
        project = {
            "name": "demo",
            "path": "/tmp/demo",
            "type": "web",
            "dev_server": {"port": 5173, "command": "npm run dev"},
        }

        result = build_project_view(project, detected_port=None, include_command=False)

        self.assertEqual(result["dev_server"]["port"], 5173)
        self.assertFalse(result["dev_server"]["running"])
        self.assertNotIn("command", result["dev_server"])

    def test_build_project_view_defaults_missing_fields(self):
        result = build_project_view({}, detected_port=None)

        self.assertEqual(result["name"], "")
        self.assertEqual(result["path"], "")
        self.assertEqual(result["type"], "unknown")
        self.assertEqual(result["dev_server"]["port"], None)
        self.assertEqual(result["dev_server"]["command"], None)
        self.assertFalse(result["dev_server"]["running"])


if __name__ == "__main__":
    unittest.main()
