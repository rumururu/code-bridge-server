"""An errored provider turn is a failed step, not an answer.

On 2026-09-04 the Claude CLI answered an llm step with a `result` whose text
was its own error — "API Error: 400 Claude Code 2.1.220 does not support this
model…". The sink took the text as the step's result, the step completed,
the next step typed it into a cafe article's title and published it, and
the run reported success. The sink now treats an `is_error` result — or a
result whose text is the CLI's error line — as the step's error.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.task_orchestrator import AgentTaskRunSink  # noqa: E402

ERROR_TEXT = (
    "API Error: 400 Claude Code 2.1.220 does not support this model; version"
    " 2.1.251 or newer is required. Run 'claude update', or update the Claude"
    " desktop app, then try again."
)


def _sink() -> AgentTaskRunSink:
    return AgentTaskRunSink(run_id="run_x")


class ErroredResultTest(unittest.TestCase):
    def test_an_is_error_result_is_the_step_error_not_its_answer(self):
        sink = _sink()
        asyncio.run(sink.send_json({"type": "result", "subtype": "error", "is_error": True, "result": ERROR_TEXT}))
        self.assertEqual(sink.error_message, ERROR_TEXT)
        self.assertIsNone(sink.result_text)

    def test_the_cli_error_line_is_caught_even_without_the_flag(self):
        sink = _sink()
        asyncio.run(sink.send_json({"type": "result", "subtype": "success", "result": ERROR_TEXT}))
        self.assertEqual(sink.error_message, ERROR_TEXT)
        self.assertIsNone(sink.result_text)

    def test_a_real_answer_is_still_the_answer(self):
        sink = _sink()
        asyncio.run(sink.send_json({"type": "result", "subtype": "success", "is_error": False, "result": "셀 안에서 줄바꿈: Alt+Enter"}))
        self.assertEqual(sink.result_text, "셀 안에서 줄바꿈: Alt+Enter")
        self.assertIsNone(sink.error_message)

    def test_an_answer_that_mentions_an_api_error_is_not_thrown_away(self):
        sink = _sink()
        asyncio.run(sink.send_json({"type": "result", "is_error": False, "result": "The log shows 'API Error: 429' at 03:00; retry later."}))
        self.assertEqual(sink.result_text, "The log shows 'API Error: 429' at 03:00; retry later.")
