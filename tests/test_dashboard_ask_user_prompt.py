"""The dashboard has to answer the Configurator's questions, not print them.

Every Claude session is told to ask a multiple-choice question as a tag
rather than as prose — `<ask_user question="..." options="A|B"/>`, appended to
the system prompt in `llm/claude_session.py`. The directive is unconditional,
so an Agent Builder turn running on Claude emits it, and this screen printed
the raw tag at the reader: markup where a question should be, and no way to
answer it.

Two things are checked, and the second is the one that rots quietly.

**The parser works.** Not "the file contains the word ask_user" — the actual
functions are lifted out of the template and run, because a regex that no
longer matches what the model emits passes any string-containment test ever
written.

**Both clients read the directive the same way.** The app has its own parser
(`lib/utils/interactive_prompt_parser.dart`) against the same tag from the
same prompt. Two clients disagreeing about one directive is a worse failure
than either handling it badly, because only one of them would be looked at
when the complaint came in.
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

from dashboard_js import MARKUP, NODE, js_function, run_js  # noqa: E402

REPO_ROOT = SERVER_DIR.parent
DART_PARSER = (
    REPO_ROOT / "lib" / "utils" / "interactive_prompt_parser.dart"
).read_text(encoding="utf-8")



def _call_js(function_name: str, argument: str) -> object:
    """Run one of the template's parser functions on `argument`.

    The argument is handed over as a JSON literal rather than interpolated
    into the source. The strings under test are full of double quotes — that
    is what the tag is made of — and pasting them into JS source produces a
    syntax error that looks exactly like a broken parser.
    """
    tag_line = re.search(r"const ASK_USER_TAG = .+;", MARKUP)
    assert tag_line is not None, "ASK_USER_TAG is gone from agents.html"
    script = "\n".join(
        [
            tag_line.group(0),
            js_function("parseAskUser"),
            js_function("stripAskUserTags"),
            f"const INPUT = {json.dumps(argument)};",
            f"console.log(JSON.stringify({function_name}(INPUT)));",
        ]
    )
    output = run_js(script).strip()
    return json.loads(output) if output else None


@unittest.skipIf(NODE is None, "node is not installed on this machine")
class DashboardAskUserParserTest(unittest.TestCase):
    QUESTION = (
        "Which directory should the check look at?\n"
        '<ask_user question="Which directory?" options="Root (/)|Home"/>'
    )

    def test_a_tagged_question_parses_into_question_and_options(self):
        parsed = _call_js("parseAskUser", self.QUESTION)
        self.assertEqual(parsed["question"], "Which directory?")
        self.assertEqual(parsed["options"], ["Root (/)", "Home"])

    def test_the_tag_is_removed_from_the_prose(self):
        prose = _call_js("stripAskUserTags", self.QUESTION)
        self.assertEqual(prose, "Which directory should the check look at?")
        self.assertNotIn("ask_user", prose)

    def test_one_option_is_not_a_choice(self):
        # A lone button invites a click that says nothing; the tag is
        # malformed or prose was written into it. The app's parser refuses
        # this too — see the parity test below.
        self.assertIsNone(
            _call_js("parseAskUser", '<ask_user question="Go?" options="Yes"/>')
        )

    def test_ordinary_prose_is_left_exactly_alone(self):
        plain = "I will check the root filesystem each night at 3am."
        self.assertIsNone(_call_js("parseAskUser", plain))
        self.assertEqual(_call_js("stripAskUserTags", plain), plain)


class BothClientsReadOneDirectiveTest(unittest.TestCase):
    def test_the_two_parsers_use_the_same_pattern(self):
        js = re.search(r"const ASK_USER_TAG = /(.+)/;", MARKUP)
        dart = re.search(r"r'(<ask_user.+?)',", DART_PARSER)
        self.assertIsNotNone(js, "ASK_USER_TAG is gone from agents.html")
        self.assertIsNotNone(dart, "the Dart tag pattern moved or was renamed")
        # `\/` in a JS literal is `/` in the pattern; Dart's raw string needs
        # no escape. Nothing else may differ.
        self.assertEqual(js.group(1).replace("\\/", "/"), dart.group(1))

    def test_both_refuse_a_single_option(self):
        self.assertIn("options.length < 2", MARKUP)
        self.assertIn("optionsList.length < 2", DART_PARSER)


if __name__ == "__main__":
    unittest.main()
