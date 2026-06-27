import asyncio
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import BrowserActionAdapterResult, _is_placeholder  # noqa: E402
from agent.browser_action_executor import execute_browser_actions  # noqa: E402


class FakeBrowserActionAdapter:
    def __init__(self):
        self.actions = None
        self.context = None

    async def run_actions(self, actions, *, context):
        self.actions = actions
        self.context = context
        return BrowserActionAdapterResult(
            status="completed",
            message="done",
            observations=[{"title": "Example"}],
        )


class BrowserActionExecutorTest(unittest.TestCase):
    def test_delegates_to_adapter(self):
        adapter = FakeBrowserActionAdapter()
        result = asyncio.run(
            execute_browser_actions(
                [{"type": "navigate", "url": "https://example.com"}],
                context={"run_id": "run_1"},
                adapter=adapter,
            )
        )

        self.assertTrue(result.completed)
        self.assertEqual(adapter.actions[0]["type"], "navigate")
        self.assertEqual(adapter.context["run_id"], "run_1")
        self.assertEqual(result.observations, [{"title": "Example"}])

    def test_missing_actions_waits_for_user(self):
        result = asyncio.run(
            execute_browser_actions(
                [],
                context={"run_id": "run_1"},
                adapter=FakeBrowserActionAdapter(),
            )
        )

        self.assertTrue(result.waiting_for_user)
        self.assertEqual(result.wait_reason, "browser_actions_missing")

    def test_unresolved_template_values_are_placeholders(self):
        self.assertTrue(_is_placeholder("{{approved_note_body}}"))
        self.assertTrue(_is_placeholder("configured_url"))
        self.assertTrue(_is_placeholder("recipient_required"))
        self.assertFalse(_is_placeholder("rumururu@naver.com"))


if __name__ == "__main__":
    unittest.main()
