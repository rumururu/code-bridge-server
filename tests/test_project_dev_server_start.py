import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_dev_server_start import (
    NO_COMMAND_TEMPLATE,
    resolve_dev_server_start_plan,
    spawn_dev_server_process,
)
from project_models import ProjectType


class ProjectDevServerStartTest(unittest.TestCase):
    def test_resolve_plan_fails_for_missing_project_path(self):
        result = resolve_dev_server_start_plan("demo", {"type": "nextjs"})

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Invalid dev server configuration for demo")
        self.assertIsNone(result.plan)

    def test_resolve_plan_fails_without_command(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project = {"path": tmp_dir, "type": "nextjs", "dev_server": {}}
            result = resolve_dev_server_start_plan(
                "demo",
                project,
                infer_command=lambda _path, _type: None,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, NO_COMMAND_TEMPLATE.format(name="demo"))

    def test_resolve_plan_fails_when_project_path_missing_on_disk(self):
        project = {
            "path": "/tmp/this-path-should-not-exist-dev-server-plan",
            "type": "nextjs",
            "dev_server": {"command": "npm run dev"},
        }

        result = resolve_dev_server_start_plan("demo", project)

        self.assertFalse(result.success)
        self.assertIn("Project path does not exist", result.error_message or "")

    def test_resolve_plan_infers_command_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project = {
                "path": tmp_dir,
                "type": "nextjs",
                "dev_server": {"port": 5173},
            }
            fake_db = MagicMock()
            result = resolve_dev_server_start_plan(
                "demo",
                project,
                project_db=fake_db,
                infer_command=lambda _path, _type: "pnpm dev",
            )

        self.assertTrue(result.success)
        assert result.plan is not None
        self.assertEqual(result.plan.command, "pnpm dev")
        self.assertEqual(result.plan.port_hint, 5173)
        self.assertEqual(result.plan.project_type, ProjectType.NEXTJS)
        fake_db.update.assert_called_once_with("demo", {"dev_server": {"command": "pnpm dev"}})

    def test_resolve_plan_uses_existing_command_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project = {
                "path": tmp_dir,
                "type": "flutter",
                "dev_server": {"command": "custom start", "port": "5173"},
            }
            infer_mock = MagicMock(return_value="unused")
            result = resolve_dev_server_start_plan(
                "demo",
                project,
                infer_command=infer_mock,
            )

        self.assertTrue(result.success)
        assert result.plan is not None
        self.assertEqual(result.plan.command, "custom start")
        self.assertIsNone(result.plan.port_hint)
        self.assertEqual(result.plan.project_type, ProjectType.FLUTTER)
        infer_mock.assert_not_called()

    def test_spawn_dev_server_process_invokes_popen(self):
        fake_process = MagicMock()
        with patch(
            "project_dev_server_start.subprocess.Popen",
            return_value=fake_process,
        ) as mock_popen:
            result = spawn_dev_server_process("npm run dev", "/tmp/demo")

        self.assertIs(result, fake_process)
        mock_popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
