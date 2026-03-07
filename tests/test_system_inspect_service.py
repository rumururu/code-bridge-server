import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import system_inspect_service


class SystemInspectServiceTest(unittest.TestCase):
    def test_list_system_directories_for_current_server_requires_absolute_path(self):
        with patch(
            "system_inspect_service.validate_accessible_path",
            return_value=True,
        ):
            result = system_inspect_service.list_system_directories_for_current_server("relative/path")

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload.get("error"), "Path must be absolute (start with /)")

    def test_list_system_directories_returns_403_for_inaccessible_path(self):
        with patch(
            "system_inspect_service.validate_accessible_path",
            return_value=False,
        ):
            result = system_inspect_service.list_system_directories_for_current_server("/some/path")

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 403)
        self.assertIn("outside accessible folders", result.payload.get("error", ""))

    def test_list_system_directories_returns_accessible_roots_when_no_path(self):
        fake_roots = [
            {"name": "Projects", "path": "/home/user/Projects"},
            {"name": "Documents", "path": "/home/user/Documents"},
        ]
        with patch(
            "system_inspect_service.get_allowed_roots",
            return_value=fake_roots,
        ):
            result = system_inspect_service.list_system_directories_for_current_server(None)

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.payload.get("is_accessible_roots"))
        self.assertEqual(result.payload.get("folders"), fake_roots)

    def test_list_project_candidates_for_current_server_invalid_depth(self):
        with patch(
            "system_inspect_service.resolve_project_path",
            return_value=(Path("/tmp/root"), None, None),
        ), patch(
            "system_inspect_service.validate_accessible_path",
            return_value=True,
        ):
            result = system_inspect_service.list_project_candidates_for_current_server(
                "/tmp/root",
                max_depth=999,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertIn("max_depth must be between 0", result.payload.get("error", ""))

    def test_list_project_candidates_for_current_server_enriches_registered_flag(self):
        fake_db = MagicMock()
        fake_db.get_all.return_value = [{"name": "alpha", "path": "/tmp/root/a"}]

        with patch(
            "system_inspect_service.resolve_project_path",
            return_value=(Path("/tmp/root"), None, None),
        ), patch(
            "system_inspect_service.validate_accessible_path",
            return_value=True,
        ), patch(
            "system_inspect_service.parse_excluded_dirs",
            return_value={"build"},
        ), patch(
            "system_inspect_service.scan_project_candidates",
            return_value=[
                {"name": "a", "path": "/tmp/root/a"},
                {"name": "b", "path": "/tmp/root/b"},
            ],
        ), patch(
            "system_inspect_service.collect_existing_project_state",
            return_value=(set(), {"/tmp/root/a": "alpha"}),
        ):
            result = system_inspect_service.list_project_candidates_for_current_server(
                "/tmp/root",
                exclude_dirs="build",
                max_depth=1,
                project_db=fake_db,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload.get("excluded_dirs"), ["build"])
        self.assertEqual(result.payload.get("count"), 2)
        self.assertTrue(result.payload["candidates"][0]["registered"])
        self.assertEqual(result.payload["candidates"][0]["registered_project_name"], "alpha")
        self.assertFalse(result.payload["candidates"][1]["registered"])

    def test_list_project_candidates_returns_403_for_inaccessible_path(self):
        with patch(
            "system_inspect_service.resolve_project_path",
            return_value=(Path("/tmp/root"), None, None),
        ), patch(
            "system_inspect_service.validate_accessible_path",
            return_value=False,
        ):
            result = system_inspect_service.list_project_candidates_for_current_server("/tmp/root")

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 403)
        self.assertIn("outside accessible folders", result.payload.get("error", ""))


class SystemInspectServiceAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_system_usage_for_current_server_merges_snapshots(self):
        fake_config = MagicMock(weekly_budget_usd=12.5, usage_window_days=7)
        fake_usage_db = MagicMock()
        fake_usage_db.get_weekly_summary.return_value = {"spent": 3.2}

        with patch(
            "system_inspect_service.fetch_claude_usage_snapshot",
            new=AsyncMock(return_value={"sessions": []}),
        ), patch(
            "system_inspect_service.merge_usage_for_display",
            return_value={"total_spent": 3.2},
        ) as merge_mock:
            result = await system_inspect_service.get_system_usage_for_current_server(
                config=fake_config,
                usage_db=fake_usage_db,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, {"total_spent": 3.2})
        fake_usage_db.get_weekly_summary.assert_called_once_with(budget_usd=12.5, window_days=7)
        merge_mock.assert_called_once_with({"spent": 3.2}, {"sessions": []})


if __name__ == "__main__":
    unittest.main()
