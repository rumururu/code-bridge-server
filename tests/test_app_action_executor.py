import asyncio
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.app_action_executor import (  # noqa: E402
    SUPPORTED_APP_ACTION_TYPES,
    AdbAppActionAdapter,
    app_action_gap,
)

_UI_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<hierarchy><node text="Continue" bounds="[0,0][100,50]" /></hierarchy>'
)


@contextlib.contextmanager
def _fake_device(calls: list[tuple[str, ...]], *, ui_xml: str = _UI_XML):
    """A device that records every adb call and answers the ones that matter."""

    async def fake_adb(_adb_path, *args, timeout=30):
        calls.append(tuple(args))
        if "resolve-activity" in args:
            return "com.example.app/.MainActivity\n"
        if args[-1].endswith("_cb_agent_ui.xml") and "cat" in args:
            return ui_xml
        return ""

    async def fake_screenshot(*_args, **_kwargs):
        return Path("/tmp/fake_screenshot.png")

    with patch(
        "agent.app_action_executor._resolve_adb_path", return_value="/fake/adb"
    ), patch(
        "agent.app_action_executor._resolve_device_id",
        return_value=("emulator-5554", None),
    ), patch(
        "agent.app_action_executor._adb", fake_adb
    ), patch(
        "agent.app_action_executor._screenshot", fake_screenshot
    ):
        yield


def _run(actions, calls=None, *, ui_xml: str = _UI_XML):
    with _fake_device(calls if calls is not None else [], ui_xml=ui_xml):
        return asyncio.run(AdbAppActionAdapter().run_actions(actions, context={}))


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


class BuilderWrittenActionTest(unittest.TestCase):
    """The payloads the server's own normalizer writes (`configurator.py`)."""

    def test_builder_placeholders_are_refused_not_run(self):
        for action, reason in (
            (
                {"type": "verify_launch", "app": "installed_app_from_previous_step"},
                "app_action_needs_package_name",
            ),
            (
                {"type": "install_app", "source": "user_provided_store_or_package"},
                "app_install_needs_package_or_apk",
            ),
            (
                {"type": "tap_text", "text": "join_or_apply_control_from_current_screen"},
                "app_action_needs_concrete_target",
            ),
            (
                {"type": "open_play_store", "source": "user_provided_store_or_package"},
                "app_action_needs_package_name",
            ),
        ):
            with self.subTest(action=action["type"]):
                gap = app_action_gap(action)
                self.assertIsNotNone(gap)
                self.assertEqual(gap.reason, reason)

    def test_play_store_placeholder_no_longer_reaches_the_device(self):
        # It used to launch `market://details?id=user_provided_store_or_package`
        # — a wrong action that ran, which is worse than one that stops.
        calls: list[tuple[str, ...]] = []

        result = _run(
            [{"type": "open_play_store", "source": "user_provided_store_or_package"}],
            calls,
        )

        self.assertEqual(result.status, "waiting_for_user")
        self.assertEqual(result.wait_reason, "app_action_needs_package_name")
        self.assertEqual([call for call in calls if "am" in call], [])

    def test_app_key_resolves_a_real_package(self):
        # `_verify_launch_actions` writes `app`; the executor read only
        # `package`/`package_name`/`app_id`/`source` and parked on its own step.
        calls: list[tuple[str, ...]] = []

        result = _run([{"type": "verify_launch", "app": "com.example.app"}], calls)

        self.assertEqual(result.status, "completed")
        self.assertIn(
            ("-s", "emulator-5554", "shell", "am", "start", "-n", "com.example.app/.MainActivity"),
            calls,
        )


class NewlyImplementedActionTest(unittest.TestCase):
    """Types that normalized, passed the gate, and hit `_unsupported_action`."""

    def test_launch_and_open_app_start_the_resolved_component(self):
        for action_type in ("launch_app", "open_app"):
            with self.subTest(action_type=action_type):
                calls: list[tuple[str, ...]] = []

                result = _run(
                    [{"type": action_type, "package": "com.example.app"}], calls
                )

                self.assertEqual(result.status, "completed")
                self.assertIn(
                    (
                        "-s",
                        "emulator-5554",
                        "shell",
                        "am",
                        "start",
                        "-n",
                        "com.example.app/.MainActivity",
                    ),
                    calls,
                )

    def test_close_app_force_stops(self):
        calls: list[tuple[str, ...]] = []

        result = _run([{"type": "close_app", "package": "com.example.app"}], calls)

        self.assertEqual(result.status, "completed")
        self.assertIn(
            ("-s", "emulator-5554", "shell", "am", "force-stop", "com.example.app"),
            calls,
        )

    def test_tap_uses_coordinates_when_given(self):
        calls: list[tuple[str, ...]] = []

        result = _run([{"type": "tap", "x": 120, "y": 340}], calls)

        self.assertEqual(result.status, "completed")
        self.assertIn(
            ("-s", "emulator-5554", "shell", "input", "tap", "120", "340"), calls
        )

    def test_tap_without_coordinates_resolves_the_text_on_screen(self):
        calls: list[tuple[str, ...]] = []

        result = _run([{"type": "tap", "target": "Continue"}], calls)

        self.assertEqual(result.status, "completed")
        self.assertIn(
            ("-s", "emulator-5554", "shell", "input", "tap", "50", "25"), calls
        )

    def test_input_text_escapes_spaces_for_the_device_shell(self):
        for action_type in ("input_text", "type_text"):
            with self.subTest(action_type=action_type):
                calls: list[tuple[str, ...]] = []

                result = _run([{"type": action_type, "text": "hello there"}], calls)

                self.assertEqual(result.status, "completed")
                self.assertIn(
                    ("-s", "emulator-5554", "shell", "input", "text", "hello%sthere"),
                    calls,
                )

    def test_non_ascii_input_is_refused_rather_than_silently_dropped(self):
        # `adb shell input text` returns success and types nothing for these.
        calls: list[tuple[str, ...]] = []

        result = _run([{"type": "input_text", "text": "안녕하세요"}], calls)

        self.assertEqual(result.status, "waiting_for_user")
        self.assertEqual(result.wait_reason, "app_action_text_not_typeable")
        self.assertEqual([call for call in calls if "input" in call], [])

    def test_press_key_accepts_aliases_and_keycodes(self):
        for value, expected in (
            ("back", "KEYCODE_BACK"),
            ("KEYCODE_ENTER", "KEYCODE_ENTER"),
            ("recent", "KEYCODE_APP_SWITCH"),
        ):
            with self.subTest(value=value):
                calls: list[tuple[str, ...]] = []

                result = _run([{"type": "press_key", "key": value}], calls)

                self.assertEqual(result.status, "completed")
                self.assertIn(
                    ("-s", "emulator-5554", "shell", "input", "keyevent", expected),
                    calls,
                )

    def test_unknown_key_stops_instead_of_pressing_something_else(self):
        result = _run([{"type": "press_key", "key": "make_coffee"}])

        self.assertEqual(result.status, "waiting_for_user")
        self.assertEqual(result.wait_reason, "app_action_needs_key")

    def test_wait_text_completes_when_the_text_is_there(self):
        result = _run([{"type": "wait_text", "text": "Continue", "timeout": 2}])

        self.assertEqual(result.status, "completed")

    def test_wait_text_stops_when_the_text_never_appears(self):
        result = _run(
            [{"type": "wait_text", "text": "Continue", "timeout": 1}],
            ui_xml="<hierarchy></hierarchy>",
        )

        self.assertEqual(result.status, "waiting_for_user")
        self.assertEqual(result.wait_reason, "app_action_target_not_found")

    def test_every_supported_type_is_executed_by_the_adapter(self):
        # The gate refuses anything outside `SUPPORTED_APP_ACTION_TYPES`, so a
        # type advertised there and not handled below would be a silent park.
        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "app.apk"
            apk.write_bytes(b"apk")
            payloads = {
                "read_screen": {"type": "read_screen"},
                "read_ui": {"type": "read_ui"},
                "dump_ui": {"type": "dump_ui"},
                "screenshot": {"type": "screenshot"},
                "wait": {"type": "wait", "seconds": 0.01},
                "wait_text": {"type": "wait_text", "text": "Continue", "timeout": 1},
                "tap": {"type": "tap", "x": 1, "y": 2},
                "tap_text": {"type": "tap_text", "text": "Continue"},
                "input_text": {"type": "input_text", "text": "hi"},
                "type_text": {"type": "type_text", "text": "hi"},
                "press_key": {"type": "press_key", "key": "back"},
                "open_play_store": {"type": "open_play_store", "package": "com.example.app"},
                "install_app": {"type": "install_app", "apk_path": str(apk)},
                "launch_app": {"type": "launch_app", "package": "com.example.app"},
                "open_app": {"type": "open_app", "package": "com.example.app"},
                "close_app": {"type": "close_app", "package": "com.example.app"},
                "verify_launch": {"type": "verify_launch", "package": "com.example.app"},
            }
            self.assertEqual(set(payloads), set(SUPPORTED_APP_ACTION_TYPES))

            for action_type, action in payloads.items():
                with self.subTest(action_type=action_type):
                    result = _run([action])

                    self.assertEqual(result.status, "completed", msg=action_type)
