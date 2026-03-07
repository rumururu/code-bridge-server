import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_device_run_plan import resolve_device_run_plan


class ProjectDeviceRunPlanTest(unittest.TestCase):
    def test_resolve_device_run_plan_requires_device_id(self):
        result = resolve_device_run_plan("demo", "   ", {"type": "flutter", "path": "/tmp"})
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Device ID is required")

    def test_resolve_device_run_plan_requires_project(self):
        result = resolve_device_run_plan("demo", "emulator-5554", None)
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Project demo not found")

    def test_resolve_device_run_plan_requires_flutter_project(self):
        result = resolve_device_run_plan(
            "demo",
            "emulator-5554",
            {"type": "nextjs", "path": "/tmp"},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Only Flutter projects support device run")

    def test_resolve_device_run_plan_requires_existing_path(self):
        result = resolve_device_run_plan(
            "demo",
            "emulator-5554",
            {"type": "flutter", "path": "/tmp/this-path-should-not-exist-device-plan"},
        )
        self.assertFalse(result.success)
        self.assertIn("Project path does not exist", result.error_message or "")

    def test_resolve_device_run_plan_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = resolve_device_run_plan(
                "demo",
                " emulator-5554 ",
                {"type": "flutter", "path": tmp_dir},
            )

        self.assertTrue(result.success)
        assert result.plan is not None
        self.assertEqual(result.plan.project_path, str(Path(tmp_dir)))
        self.assertEqual(result.plan.device_id, "emulator-5554")


if __name__ == "__main__":
    unittest.main()
