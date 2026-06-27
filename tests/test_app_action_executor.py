import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.app_action_executor import AdbAppActionAdapter  # noqa: E402


class AppActionExecutorTest(unittest.TestCase):
    def test_verify_launch_resolves_activity_and_starts_component(self):
        calls: list[tuple[str, ...]] = []

        async def fake_adb(_adb_path, *args, timeout=30):
            calls.append(tuple(args))
            if args[:4] == (
                "-s",
                "emulator-5554",
                "shell",
                "cmd",
            ):
                return (
                    "priority=0 preferredOrder=0 match=0x108000 specificIndex=-1 isDefault=false\n"
                    "com.android.settings/.Settings\n"
                )
            return ""

        with patch(
            "agent.app_action_executor._resolve_adb_path",
            return_value="/fake/adb",
        ), patch(
            "agent.app_action_executor._resolve_device_id",
            return_value=("emulator-5554", None),
        ), patch(
            "agent.app_action_executor._adb",
            fake_adb,
        ):
            result = asyncio.run(
                AdbAppActionAdapter().run_actions(
                    [{"type": "verify_launch", "package": "com.android.settings"}],
                    context={},
                )
            )

        self.assertEqual(result.status, "completed")
        self.assertIn(
            (
                "-s",
                "emulator-5554",
                "shell",
                "cmd",
                "package",
                "resolve-activity",
                "--brief",
                "-c",
                "android.intent.category.LAUNCHER",
                "com.android.settings",
            ),
            calls,
        )
        self.assertIn(
            (
                "-s",
                "emulator-5554",
                "shell",
                "am",
                "start",
                "-n",
                "com.android.settings/.Settings",
            ),
            calls,
        )
