"""The dashboard names the argument boxes from the script that gets them.

A registered script states its own interface — `# @param NAME required
description`, parsed at registration and published on the scripts catalog —
and the schema publishes `script_args` as kind `script_args` carrying
`derives_from: 'script_id'` so a client knows where to read that from
(`agent/workflow_step_schema.KIND_SCRIPT_ARGS`).

Why this is not cosmetic: a step whose script needs a directory and was given
an empty list does not fail, it stops at 3am and asks. "Extra arguments, one
per line" gave nobody a reason to type anything into it.

The renderer is executed rather than grepped. A `switch` case that stopped
matching, or a lookup against the wrong key, passes every string-containment
test ever written and draws a blank form.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard_js import NODE, js_function, run_js  # noqa: E402

from code_bridge_core.workflow_step_schema import build_step_schema  # noqa: E402



#: The page's own helpers this renderer leans on. Stubbed rather than lifted
#: only where they touch the DOM or the i18n table; everything that decides
#: *what is drawn* comes from the template itself.
_STUBS = """
const escapeHtml = (value) => String(value ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');
const t = (key) => key;
const getStepField = (step, key) => step[key];
const stringListValues = (step, key) =>
  Array.isArray(step[key]) ? step[key] : [];
"""


def _render(step: dict, scripts: list[dict]) -> str:
    field = None
    for entry in build_step_schema()["types"]:
        if entry["type"] != "shell":
            continue
        for candidate in entry["fields"]:
            if candidate["key"] == "script_args":
                field = candidate
    assert field is not None, "the shell type no longer publishes script_args"

    script = "\n".join(
        [
            _STUBS,
            f"const scripts = {json.dumps(scripts)};",
            js_function("renderStringListField"),
            js_function("renderScriptArgsField"),
            f"const STEP = {json.dumps(step)};",
            f"const FIELD = {json.dumps(field)};",
            "console.log(renderScriptArgsField(STEP, 0, FIELD, 'Arguments', ''));",
        ]
    )
    return run_js(script)


DECLARING_SCRIPT = {
    "id": "sc_1",
    "name": "Disk check",
    "parameters": [
        {
            "name": "TARGET_PATH",
            "required": True,
            "description": "The directory whose free space is checked",
        },
        {"name": "THRESHOLD", "required": False, "description": ""},
    ],
}


@unittest.skipIf(NODE is None, "node is not installed on this machine")
class ScriptArgsFieldTest(unittest.TestCase):
    def test_the_schema_still_derives_this_field_from_script_id(self):
        # The renderer reads `field.derives_from`; if the schema stopped
        # publishing it the form would silently look up the wrong key and
        # every box would go unnamed.
        for entry in build_step_schema()["types"]:
            if entry["type"] != "shell":
                continue
            field = next(f for f in entry["fields"] if f["key"] == "script_args")
            self.assertEqual(field["kind"], "script_args")
            self.assertEqual(field["derives_from"], "script_id")
            self.assertEqual(field["fallback_kind"], "string_list")

    def test_each_declared_parameter_gets_its_own_named_box(self):
        html = _render(
            {"script_id": "sc_1", "script_args": ["/"]}, [DECLARING_SCRIPT]
        )
        self.assertIn("TARGET_PATH", html)
        self.assertIn("THRESHOLD", html)
        # The script's own words about what the box is for.
        self.assertIn("The directory whose free space is checked", html)
        # Required is marked; optional is not.
        self.assertIn("TARGET_PATH *", html)
        self.assertNotIn("THRESHOLD *", html)
        # And the stored value lands in the first box, not the second.
        first = html.index("TARGET_PATH")
        second = html.index("THRESHOLD")
        self.assertIn('value="/"', html[first:second])

    def test_a_script_that_declares_nothing_keeps_the_plain_list(self):
        # Absent `parameters` is *unknown*, not "needs nothing" — every script
        # registered before declarations existed reads this way. A form with
        # no boxes would be a claim the script never made.
        html = _render(
            {"script_id": "sc_old", "script_args": ["x"]},
            [{"id": "sc_old", "name": "Legacy"}],
        )
        self.assertNotIn("TARGET_PATH", html)
        self.assertIn("s_add_item", html)
        self.assertIn('value="x"', html)

    def test_no_script_chosen_keeps_the_plain_list(self):
        html = _render({"script_args": []}, [DECLARING_SCRIPT])
        self.assertNotIn("TARGET_PATH", html)
        self.assertIn("s_add_item", html)

    def test_arguments_beyond_the_declared_ones_are_still_shown(self):
        # A step carrying more than the script declares is a step somebody had
        # a reason for. Hiding the surplus deletes it on the next save.
        html = _render(
            {"script_id": "sc_1", "script_args": ["/", "80", "--verbose"]},
            [DECLARING_SCRIPT],
        )
        self.assertIn('value="--verbose"', html)

    def test_a_later_box_pads_rather_than_leaving_a_hole(self):
        # Positional arguments: filling the second box first must pad the
        # first with an empty string. Assigning past the end of a JS array
        # leaves a hole, and JSON.stringify writes a hole as `null` — which is
        # not a string and not what a step carrying arguments means.
        script = "\n".join(
            [
                _STUBS,
                "const steps = [{script_id: 'sc_1'}];",
                "const setStepField = (step, key, value) => { step[key] = value; };",
                js_function("updateScriptArgItem"),
                "updateScriptArgItem(0, 'script_args', 1, '80');",
                "console.log(JSON.stringify(steps[0].script_args));",
            ]
        )
        self.assertEqual(json.loads(run_js(script).strip()), ["", "80"])


@unittest.skipIf(NODE is None, "node is not installed on this machine")
class UnknownKindFallsBackTest(unittest.TestCase):
    def test_a_kind_this_build_cannot_draw_takes_the_published_fallback(self):
        # The point of `fallback_kind`: an older dashboard against a newer
        # server keeps the field visible and editable instead of dropping it,
        # and an editor that cannot show a value is one that deletes it.
        field = {
            "key": "script_args",
            "kind": "some_future_kind",
            "required": False,
            "options_source": None,
            "options": None,
            "label": {"en": "Arguments"},
            "help": {"en": ""},
            "derives_from": "script_id",
            "fallback_kind": "string_list",
        }
        script = "\n".join(
            [
                _STUBS,
                "const scripts = [];",
                "const schemaText = (map, fallback) => (map && map.en) || fallback;",
                js_function("renderTextField"),
                js_function("renderSelectField"),
                js_function("renderStringListField"),
                js_function("renderActionListField"),
                js_function("renderScriptArgsField"),
                js_function("renderStepField"),
                "const STEP = {script_args: ['kept']};",
                f"const FIELD = {json.dumps(field)};",
                "console.log(renderStepField(STEP, 0, FIELD));",
            ]
        )
        self.assertIn('value="kept"', run_js(script))


if __name__ == "__main__":
    unittest.main()
