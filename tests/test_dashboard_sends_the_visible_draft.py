"""The first turn after "정의 수정" told the server we had nothing.

`editSelectedAgent` starts a clean conversation — deliberately, so a session
about a different agent cannot overwrite this one — and `resetBuilder` clears
`builderDraft` on the way. The form is still showing the saved agent, but the
turn that follows carried no draft, so the server's idea of "what we had" was
empty and the model redrew the workflow from the message alone.

Steps the user could see on screen then vanished from what came back, over and
over in one session: 9 steps to 7 when asked to fix three instruction texts;
4 to 2 when asked to attach a schedule, taking both browser steps and three
selectors that had been read off a live page; 4 to 2 again when asked to
reword one instruction. The Configurator named the cause itself every time —
"초안이 비어 있는 상태로 도착했습니다" — and the disclosure that reports
dropped steps stayed silent, having nothing to compare against.

So the body carries what the screen is showing when the conversation has
nothing of its own.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from tests.dashboard_js import NODE, js_function, json_literal, run_js  # noqa: E402

FIELDS = {
    "agentName": "카페 글 등록",
    "agentDescription": "매일 한 편",
    "agentPrompt": "너는 글을 쓴다",
    "agentProvider": "anthropic",
    "agentModel": "",
}

STEPS = [{"id": "write_post", "type": "browser_action"}]


def _harness(*, fields: dict, steps: list) -> str:
    return f"""
    const values = {json_literal(fields)};
    global.document = {{
      getElementById: (id) => ({{ value: values[id] === undefined ? "" : values[id] }}),
    }};
    function collectSteps() {{ return {json_literal(steps)}; }}
    {js_function("draftFromFields")}
    console.log(JSON.stringify(draftFromFields()));
    """


@unittest.skipUnless(NODE, "node is not installed")
class DraftFromFieldsTests(unittest.TestCase):
    def test_it_reports_what_the_form_is_showing(self) -> None:
        draft = eval(run_js(_harness(fields=FIELDS, steps=STEPS)).replace("null", "None"))

        self.assertEqual(draft["name"], "카페 글 등록")
        self.assertEqual(draft["flow"], STEPS)
        self.assertEqual(draft["system_prompt"], "너는 글을 쓴다")
        self.assertEqual(draft["provider_id"], "anthropic")

    def test_an_empty_form_reports_nothing(self) -> None:
        """A fresh "새 에이전트" has no draft to describe, and saying it has
        one would hand the server an empty flow to treat as the baseline."""

        blank = {key: "" for key in FIELDS}
        out = run_js(_harness(fields=blank, steps=[])).strip()

        self.assertEqual(out, "null")

    def test_the_flow_comes_from_the_same_reader_a_save_uses(self) -> None:
        """`collectSteps` is what `saveAgent` sends, so what the server is
        told we have is what a save would write — not a second reading of the
        DOM that could disagree with it."""

        source = js_function("draftFromFields")

        self.assertIn("collectSteps()", source)


@unittest.skipUnless(NODE, "node is not installed")
class TheTurnBodyCarriesItTests(unittest.TestCase):
    def test_the_body_falls_back_to_the_visible_draft(self) -> None:
        markup = (SERVER_DIR / "dashboard" / "templates" / "agents.html").read_text(
            "utf-8"
        )

        self.assertIn("body.draft = builderDraft || draftFromFields();", markup)
        # The old line said nothing when there was no conversation draft.
        self.assertNotIn("if (builderDraft) body.draft = builderDraft;", markup)


if __name__ == "__main__":
    unittest.main()
