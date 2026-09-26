"""The kernel's vocabulary renderer must reproduce our prompt exactly.

`agent_flow_core.authoring.vocabulary` (kernel ticket T-G-03) is the shared
form of a rule this server and Infergraph each arrived at after the same
failure: the component that runs an action publishes what it can run, next to
the code that runs it, and a test fails when the two part ways.

Adopting it is only safe if the shared renderer produces *this* server's
current prompt text byte for byte. The block is load-bearing — it is what
taught the Configurator that `extract` and `url_not_contains` exist at all —
so a renderer that came close would silently reword a prompt that was tuned
against real runs. These tests hold the kernel to the exact strings
`browser_action_vocabulary_block()` and `app_action_vocabulary_block()` return
*today*, read live from the executors rather than from a copied fixture: the
kernel's own snapshot of this block had already gone stale by five days.

Nothing here changes server behaviour. Switching the adapters to call the
kernel renderer is a separate change (T-G-08); this file is the evidence that
it can be made without touching a single character of the prompt.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_flow_core.authoring.vocabulary import (  # noqa: E402
    VocabularyDriftError,
    VocabularyItem,
    assert_vocabulary_matches,
    describe_description_only,
    render_vocabulary_block,
    vocabulary_drift_issues,
)

from agent.app_action_executor import (  # noqa: E402
    APP_ACTION_VOCABULARY,
    app_action_vocabulary_block,
)
from agent.browser_action_adapter import (  # noqa: E402
    BROWSER_ACTION_VOCABULARY,
    BROWSER_ACTIONS_NOT_EXECUTED,
    browser_action_vocabulary_block,
)

from test_app_action_vocabulary import (  # noqa: E402
    _dispatched_action_types as _app_dispatched,
)
from test_browser_action_vocabulary import (  # noqa: E402
    _dispatched_action_types as _browser_dispatched,
)

#: The one line `browser_action_vocabulary_block()` writes for the actions it
#: refuses. Stated once for both actions rather than repeated per action, which
#: is why the kernel takes it as the block's shared note instead of requiring a
#: reason on each item.
BROWSER_UNSUPPORTED_NOTE = "NOT executed — a step using these stops and asks the user"

#: `check`/`uncheck` are the `click` gesture, documented inside the `click`
#: entry rather than given rows of their own. The adapter branches on all three.
BROWSER_CLICK_ALIASES = ("check", "uncheck")


def _browser_items() -> list[VocabularyItem]:
    """What a Code Bridge adapter hands the kernel renderer (T-G-08).

    Built from the live tables, so an action added to the adapter arrives here
    without anyone editing this file — the whole point of the protocol.
    """
    items = [
        VocabularyItem(
            id=action.type,
            label=action.label_en,
            description=action.note,
            aliases=list(BROWSER_CLICK_ALIASES) if action.type == "click" else [],
        )
        for action in BROWSER_ACTION_VOCABULARY
    ]
    items += [
        VocabularyItem(id=name, supported=False)
        for name in BROWSER_ACTIONS_NOT_EXECUTED
    ]
    return items


def _app_items() -> list[VocabularyItem]:
    return [
        VocabularyItem(id=action.type, label=action.label_en, description=action.note)
        for action in APP_ACTION_VOCABULARY
    ]


def _render_browser(items: list[VocabularyItem]) -> str:
    """The kernel call that stands in for `browser_action_vocabulary_block()`.

    The header and footer are the adapter's own words; T-G-08 passes them from
    there instead of repeating them here.
    """
    return render_vocabulary_block(
        items,
        header="browser_action `actions` — the full set:",
        id_width=None,
        describe=describe_description_only,
        unsupported_note=BROWSER_UNSUPPORTED_NOTE,
        footer_notes=[
            "  A URL, selector or value left as a placeholder (configured_… ) or as an "
            "unfilled {{name}} stops the step and asks, so name a real target or an "
            "{{name}} some earlier extract fills.",
            "  Every browser_action step starts on a blank page. Cookies carry over "
            "from the previous browser step (so a login survives), but the page it "
            "left open does not — `_prepare_browser_session_for_execution` hands the next "
            "step the storage state and nothing else. So each browser step must "
            "navigate to the page it works on, even if the step before it was "
            "already there. Splitting `navigate` into one step and `fill` into the "
            "next cannot work: the second step waits for a selector on a blank page "
            "and times out. Put the whole page interaction in one step, or repeat "
            "the navigate at the top of each.",
        ],
    )


def _render_app(items: list[VocabularyItem]) -> str:
    return render_vocabulary_block(
        items,
        header="app_action `actions` — the full set (Android, over adb):",
        id_width=None,
        describe=describe_description_only,
        footer_notes=[
            "  A package name, APK path or on-screen text left as a placeholder "
            "(configured_… , user_provided_… , {{name}}) stops the step and asks, "
            "so name a real one.",
            "  There is no {{binding}} substitution on the device side: an app "
            "action cannot consume a value an earlier action read.",
        ],
    )


class TheKernelRendererReproducesOurPromptTest(unittest.TestCase):
    """T-G-03's acceptance criterion, asserted rather than eyeballed."""

    def test_the_browser_block_is_reproduced_byte_for_byte(self) -> None:
        self.assertEqual(browser_action_vocabulary_block(), _render_browser(_browser_items()))

    def test_the_app_block_is_reproduced_byte_for_byte(self) -> None:
        """A second surface, and a different shape: seventeen actions, a
        sixteen-column name field instead of eleven, two footer lines and
        nothing refused. It is what makes `id_width=None` necessary — a
        renderer with one hard-coded column would need the caller to compute
        the other, which is a copy of the renderer."""
        self.assertEqual(app_action_vocabulary_block(), _render_app(_app_items()))

    def test_the_comparison_is_against_something_substantial(self) -> None:
        """Guard the guard: two empty strings are also byte-identical."""
        block = browser_action_vocabulary_block()
        self.assertGreaterEqual(len(block.splitlines()), 12)
        self.assertIn("url_not_contains", block)
        self.assertIn("{{cafe_id}}", block)

    def test_the_refused_actions_survive_the_round_trip_with_their_reason(self) -> None:
        """The server publishes what cannot be used on purpose. A renderer
        that kept only the working actions would teach an author that `select`
        does not exist, when the truth is that it exists and is refused."""
        rendered = _render_browser(_browser_items())
        self.assertIn(
            f"  {'/'.join(BROWSER_ACTIONS_NOT_EXECUTED)}   {BROWSER_UNSUPPORTED_NOTE}",
            rendered,
        )

    def test_dropping_a_refused_action_breaks_the_reproduction(self) -> None:
        """The same criterion stated as a failure: prove the line above is
        load-bearing, not incidentally present."""
        without = _render_browser([i for i in _browser_items() if i.supported])
        self.assertNotEqual(browser_action_vocabulary_block(), without)
        for name in BROWSER_ACTIONS_NOT_EXECUTED:
            self.assertNotIn(f"  {name}", without)


class TheKernelDriftCheckAgreesWithOurOwnTest(unittest.TestCase):
    """`assert_vocabulary_matches` is meant to replace the hand-written
    comparisons in `test_browser_action_vocabulary.py` and
    `test_app_action_vocabulary.py`. It only can if it reaches the same verdict
    on the real dispatch sets."""

    def test_the_browser_vocabulary_has_no_drift(self) -> None:
        dispatched = _browser_dispatched()
        runnable = dispatched - set(BROWSER_ACTIONS_NOT_EXECUTED)
        self.assertEqual(
            vocabulary_drift_issues(
                _browser_items(),
                runnable,
                refused_keys=BROWSER_ACTIONS_NOT_EXECUTED,
                shared_unsupported_note=BROWSER_UNSUPPORTED_NOTE,
            ),
            [],
        )

    def test_the_app_vocabulary_has_no_drift(self) -> None:
        self.assertEqual(
            vocabulary_drift_issues(_app_items(), _app_dispatched()), []
        )

    def test_an_invented_action_would_fail_the_build(self) -> None:
        """The failure this protocol exists for: a documented action the
        executor cannot run is worse than an undocumented one, because the
        model writes it confidently and the step stops."""
        items = _browser_items() + [
            VocabularyItem(id="teleport", description="go anywhere")
        ]
        with self.assertRaises(VocabularyDriftError):
            assert_vocabulary_matches(
                items,
                _browser_dispatched() - set(BROWSER_ACTIONS_NOT_EXECUTED),
                refused_keys=BROWSER_ACTIONS_NOT_EXECUTED,
                shared_unsupported_note=BROWSER_UNSUPPORTED_NOTE,
            )

    def test_deleting_a_refused_action_from_the_table_would_fail_the_build(self) -> None:
        with self.assertRaises(VocabularyDriftError):
            assert_vocabulary_matches(
                [i for i in _browser_items() if i.supported],
                _browser_dispatched() - set(BROWSER_ACTIONS_NOT_EXECUTED),
                refused_keys=BROWSER_ACTIONS_NOT_EXECUTED,
                shared_unsupported_note=BROWSER_UNSUPPORTED_NOTE,
            )


if __name__ == "__main__":
    unittest.main()
