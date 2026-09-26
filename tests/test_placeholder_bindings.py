"""A `{{name}}` an earlier `extract` creates is not an unresolved placeholder.

The runtime binds, then judges: `bind_action(action, bindings)` is the first
thing the adapter's loop does, and `task_orchestrator._bindings_from_earlier_
steps` carries names across step boundaries. So at runtime the reference is a
real value long before `_is_placeholder` sees it.

The save-time contract check ran the *same* predicate with no bindings in
existence, and therefore refused the one pattern the browser vocabulary was
built for — read an id out of a page in one step, use it in the next. The flow
could not be saved at all, and the refusal said it would "stop waiting for
input", which is not true of a name something else supplies.

These tests pin both halves: the pattern saves, and everything that genuinely
does park still does not.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.workflow_contract import _check_placeholder_targets  # noqa: E402


def browser(step_id: str, *actions: dict) -> dict:
    return {"id": step_id, "type": "browser_action", "actions": list(actions)}


def extract(name: str, **rest) -> dict:
    return {"type": "extract", "name": name, "source": "html", **rest}


def navigate(url: str) -> dict:
    return {"type": "navigate", "url": url}


class ResolvableReferencesTest(unittest.TestCase):
    def test_a_later_step_may_use_what_an_earlier_one_extracted(self) -> None:
        findings = _check_placeholder_targets([
            browser("read", extract("cafe_id")),
            navigate_step := browser("go", navigate("https://x/cafes/{{cafe_id}}/menus/0")),
        ])
        self.assertEqual(findings, [], navigate_step)

    def test_a_later_action_in_the_same_step_may_use_it(self) -> None:
        """The adapter shares one `bindings` dict for a whole step."""
        findings = _check_placeholder_targets([
            browser("both", extract("cid"), navigate("https://x/{{cid}}")),
        ])
        self.assertEqual(findings, [])

    def test_two_references_both_supplied(self) -> None:
        findings = _check_placeholder_targets([
            browser("read", extract("host"), extract("cid")),
            browser("go", navigate("https://{{host}}/c/{{cid}}")),
        ])
        self.assertEqual(findings, [])


class StillBlockingTest(unittest.TestCase):
    """Everything that really does park must keep parking."""

    def test_a_name_nobody_creates(self) -> None:
        findings = _check_placeholder_targets([
            browser("go", navigate("https://x/{{nobody}}")),
        ])
        self.assertEqual(len(findings), 1)
        # The reader is told which name is unaccounted for; "it is a
        # placeholder" alone does not say what to do about it.
        self.assertIn("nobody", findings[0].ask)

    def test_order_matters_extract_after_use_does_not_count(self) -> None:
        findings = _check_placeholder_targets([
            browser("go", navigate("https://x/{{cid}}")),
            browser("read", extract("cid")),
        ])
        self.assertEqual(len(findings), 1)

    def test_an_action_cannot_satisfy_its_own_reference(self) -> None:
        findings = _check_placeholder_targets([
            browser("odd", extract("x", selector="{{x}}")),
        ])
        self.assertEqual(len(findings), 1)

    def test_a_configured_stub_is_untouched_by_any_of_this(self) -> None:
        findings = _check_placeholder_targets([
            browser("read", extract("cid")),
            browser("go", navigate("configured_url")),
        ])
        self.assertEqual(len(findings), 1)

    def test_a_required_stub_is_untouched(self) -> None:
        findings = _check_placeholder_targets([
            browser("go", {"type": "click", "selector": "selector_required"}),
        ])
        self.assertEqual(len(findings), 1)

    def test_an_unnamed_extract_supplies_nothing(self) -> None:
        findings = _check_placeholder_targets([
            browser("read", {"type": "extract", "source": "html"}),
            browser("go", navigate("https://x/{{cid}}")),
        ])
        self.assertEqual(len(findings), 1)


if __name__ == "__main__":
    unittest.main()
