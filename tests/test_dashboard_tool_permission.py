"""The dashboard has to be able to answer the Configurator's tool question.

The server parks a builder turn in `waiting_for_permission` and rides the
request out on the next poll (`routes/agents.py`). Without a card and two
buttons here, that question reaches nobody and the turn is refused ten minutes
later by the wait timeout — the feature exists on the API and nowhere a person
can see it.

Two properties carry this file.

**The page outlasts the server's wait.** The permission hold is 600 seconds and
the ordinary poll budget is 300; counting a parked tick against the thinking
budget would have the page give up on a job the server is still holding open,
while the user is mid-decision. So the two are counted separately, exactly as
the server keeps two clocks.

**The card says what would happen, not just which tool.** "May it use
playwright?" is not the question a person can answer; the URL it would open is.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard_js import COMMON_STUBS, MARKUP, NODE, js_const, js_function, run_js  # noqa: E402

from routes.agents import BUILDER_PERMISSION_WAIT_SECONDS  # noqa: E402


class ThePageOutlastsTheServersWaitTest(unittest.TestCase):
    def test_the_permission_budget_covers_the_whole_server_hold(self):
        interval_ms = float(re.search(r"BUILDER_POLL_MS\s*=\s*(\d+)", MARKUP).group(1))
        limit = float(
            re.search(r"BUILDER_PERMISSION_POLL_LIMIT\s*=\s*(\d+)", MARKUP).group(1)
        )
        budget = interval_ms / 1000.0 * limit
        self.assertGreater(
            budget,
            BUILDER_PERMISSION_WAIT_SECONDS,
            "the page gives up before the server does, so an answer given late "
            "but in time would land on a job the page had already abandoned",
        )

    def test_a_parked_tick_is_not_counted_as_thinking(self):
        # The two counters are the whole point; one counter cannot express
        # "still working" and "still waiting for you" differently.
        loop = js_function("runBuilderTurn")
        self.assertIn("BUILDER_PERMISSION_POLL_LIMIT", loop)
        self.assertIn("waiting += 1", loop)
        self.assertIn("thinking += 1", loop)
        # A parked tick must skip the thinking increment entirely.
        parked = loop[loop.index("waiting_for_permission") :]
        self.assertLess(parked.index("continue"), parked.index("thinking += 1"))


@unittest.skipIf(NODE is None, "node is not installed on this machine")
class TheCardSaysWhatWouldHappenTest(unittest.TestCase):
    REQUEST = {
        "request_id": "req_1",
        "tool_name": "mcp__playwright__browser_navigate",
        "input": {"url": "https://cafe.naver.com/mycafe/write"},
    }

    def _render(self, request: dict) -> str:
        script = "\n".join(
            [
                COMMON_STUBS,
                """
                const listeners = [];
                const log = { children: [], appendChild(n) { this.children.push(n); },
                              scrollTop: 0, scrollHeight: 0 };
                const document = {
                  getElementById: () => log,
                  querySelectorAll: () => [],
                  createElement: () => ({
                    className: '', innerHTML: '',
                    querySelectorAll: () => [],
                  }),
                };
                globalThis.document = document;
                const api = async () => ({});
                const appendBuilderLine = () => {};
                """,
                js_function("clearBuilderPermission"),
                js_function("renderBuilderPermission"),
                "let builderPermissionShown = null;",
                f"renderBuilderPermission('job_1', {json.dumps(request)});",
                "console.log(log.children[0].innerHTML);",
            ]
        )
        return run_js(script)

    def test_the_tool_and_its_actual_input_are_both_shown(self):
        html = self._render(self.REQUEST)
        self.assertIn("mcp__playwright__browser_navigate", html)
        # The URL is the part a person can actually judge.
        self.assertIn("cafe.naver.com/mycafe/write", html)

    def test_both_answers_are_offered(self):
        html = self._render(self.REQUEST)
        self.assertIn('data-allow="1"', html)
        self.assertIn('data-allow="0"', html)

    def test_a_long_input_does_not_bury_the_buttons(self):
        request = dict(self.REQUEST, input={"script": "x" * 5000})
        html = self._render(request)
        self.assertIn("…", html)
        self.assertLess(len(html), 3000)
        self.assertIn('data-allow="1"', html)


class OnlyMcpToolsEverReachThisCardTest(unittest.TestCase):
    def test_the_copy_says_shell_and_file_tools_are_never_offered(self):
        # The reader has to know the boundary is not "whatever it asks for".
        # Asserted on the copy because that is the only place a user learns it.
        for locale_marker in ("perm_body:",):
            self.assertIn(locale_marker, MARKUP)
        self.assertIn("shell commands and file writes are never available", MARKUP.lower())
        self.assertIn("셸 명령이나 파일 쓰기는 설계 대화에서 절대 쓸 수 없습니다", MARKUP)


if __name__ == "__main__":
    unittest.main()
