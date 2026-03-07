import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_build_state import (
    build_status_payload,
    mark_build_error,
    mark_build_ready,
    mark_building,
    ready_build_path,
)


class ProjectBuildStateTest(unittest.TestCase):
    def test_mark_building_sets_building_info(self):
        state = {}
        mark_building(state, "demo", "flutter")

        payload = build_status_payload(state["demo"])
        self.assertEqual(payload["status"], "building")
        self.assertEqual(payload["project_type"], "flutter")

    def test_mark_build_error_sets_error_info(self):
        state = {}
        mark_build_error(state, "demo", error_message="boom", project_type="nextjs")

        payload = build_status_payload(state["demo"])
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error_message"], "boom")
        self.assertEqual(payload["project_type"], "nextjs")

    def test_mark_build_ready_sets_ready_info(self):
        state = {}
        mark_build_ready(state, "demo", build_path="/tmp/out", project_type="nextjs")

        payload = build_status_payload(state["demo"])
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["build_path"], "/tmp/out")
        self.assertEqual(ready_build_path(state["demo"]), "/tmp/out")

    def test_build_status_payload_defaults_when_missing(self):
        payload = build_status_payload(None)
        self.assertEqual(payload["status"], "none")
        self.assertIsNone(payload["build_path"])
        self.assertIsNone(payload["error_message"])
        self.assertIsNone(payload["project_type"])
        self.assertIsNone(ready_build_path(None))


if __name__ == "__main__":
    unittest.main()
