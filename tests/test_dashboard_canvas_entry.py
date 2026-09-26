"""The canvas has to be reachable by clicking, not by typing its URL.

`/canvas/` has been served, working and complete for a while, and
`agents.html` contained the string "canvas" zero times. The only way in was to
hand-type

    /canvas/?agent=agent_<32 hex>&binding=dashboard&locale=ko

which needs an internal agent id the page shows but nobody would think to
paste. A feature only reachable that way is not shipped, however green its own
tests are — hence this file, which is about the *route into* the canvas rather
than the canvas itself.

Placement is asserted, not just presence: the entry point lives in the detail
pane's workflow section (`#paneSteps`), because that pane is already drawing
the workflow the canvas edits. A link on the page header would pass a
"contains the word canvas" test and still be in the wrong place — the header
is not about any particular agent, and the canvas cannot open without one.

The URL assertions matter for the same reason. Drop `binding=dashboard` and
the page comes up pointed at the key-guarded API it has no key for; drop
`agent` and it opens on nothing; drop `locale` and a Korean dashboard hands off
to an English editor mid-task.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

MARKUP = (SERVER_DIR / "dashboard" / "templates" / "agents.html").read_text(
    encoding="utf-8"
)


class CanvasEntryPointTest(unittest.TestCase):
    def test_the_page_mentions_the_canvas_at_all(self):
        self.assertIn("openWorkflowCanvas", MARKUP)
        self.assertIn("/canvas/", MARKUP)

    def test_the_entry_point_sits_in_the_workflow_pane(self):
        """Inside `#paneSteps`, not bolted onto the page header."""
        pane = MARKUP.split('<div id="paneSteps">', 1)
        self.assertEqual(len(pane), 2, "#paneSteps not found in agents.html")
        # The pane ends where the next sibling pane begins.
        pane_markup = pane[1].split('<div id="paneSchedule"', 1)[0]
        self.assertIn('id="canvasBtn"', pane_markup)
        self.assertIn("openWorkflowCanvas()", pane_markup)

        header = MARKUP.split('<div class="crumb"', 1)[0]
        self.assertNotIn("openWorkflowCanvas", header)

    def test_the_link_carries_agent_binding_and_locale(self):
        builder = re.search(
            r"function canvasUrlFor\(agentId\)\s*\{(.+?)\n        \}",
            MARKUP,
            re.S,
        )
        self.assertIsNotNone(builder, "canvasUrlFor() not found")
        body = builder.group(1)
        self.assertIn("agent: agentId", body)
        # `dashboard` is what points the canvas at the no-key localhost mirror
        # this page itself uses. Any other value needs a credential the page
        # does not have.
        self.assertIn("binding: 'dashboard'", body)
        self.assertIn("locale: currentLang", body)
        self.assertIn("prefers-color-scheme: dark", body)
        self.assertIn("/canvas/?", body)

    def test_it_opens_as_a_full_view_rather_than_an_inline_frame(self):
        """The canvas is full-screen and carries its own save bar; the detail
        pane is the narrow column of a two-column layout. An iframe here would
        clip the editor it is meant to show."""
        self.assertIn("window.open(canvasUrlFor(selectedAgentId)", MARKUP)
        self.assertNotIn("<iframe", MARKUP)

    def test_the_open_call_can_still_report_a_blocked_popup(self):
        """`noopener` makes window.open return null by specification, blocked
        or not — which silently breaks both the blocked-popup message and the
        refresh-on-return that keys off the same result. Caught by clicking the
        button; no unit test would have seen it, so here is the one that will.
        """
        opener = re.search(
            r"const opened = window\.open\(canvasUrlFor\(selectedAgentId\)[^)]*\)",
            MARKUP,
        )
        self.assertIsNotNone(opener, "window.open call not found")
        self.assertNotIn("noopener", opener.group(0))

    def test_a_blocked_popup_says_so(self):
        """window.open's one failure mode looks exactly like a dead button."""
        self.assertIn("open_canvas_blocked", MARKUP)

    def test_the_empty_steps_state_points_at_the_canvas(self):
        """An agent with no steps is the case where the canvas is the answer,
        not an alternative editor."""
        self.assertIn("open_canvas_empty", MARKUP)

    def test_a_canvas_save_reaches_this_page_without_a_manual_refresh(self):
        """The standalone bundle has no host bridge to announce a save, so the
        page has to notice by itself — by watching `flow_revision`, which moves
        when and only when the workflow was written."""
        self.assertIn("checkCanvasSaved", MARKUP)
        self.assertIn("agent.flow_revision", MARKUP)
        self.assertIn("revision !== canvasOpenedRevision", MARKUP)
        # Guarded on the manage view: selectAgent() repopulates the editor
        # fields, which would wipe a half-typed definition.
        self.assertIn("if (currentView !== 'manage') return;", MARKUP)

    def test_the_watch_does_not_rely_on_tab_visibility_alone(self):
        """Two windows side by side never fire a visibility change, and the
        one that *does* fire on open arrives before anything can have been
        saved. The events are an accelerant on top of the poll, not the
        mechanism."""
        self.assertIn("setInterval(checkCanvasSaved", MARKUP)
        self.assertIn("visibilitychange", MARKUP)

    def test_the_watch_stops(self):
        """A poll with no way to end is a leak: it must stop when the canvas
        closes, when another agent is selected, and when the page moves on."""
        self.assertIn("stopCanvasWatch", MARKUP)
        self.assertIn("canvasWindow.closed", MARKUP)

    def test_both_dashboard_languages_label_the_button(self):
        for key in ("open_canvas:", "open_canvas_empty:", "open_canvas_blocked:"):
            with self.subTest(key=key):
                self.assertEqual(
                    MARKUP.count(key),
                    2,
                    f"{key} must be defined in both the en and ko tables",
                )


class QuotaOfferMarkupTest(unittest.TestCase):
    """The other half of the same complaint: a builder turn that dies on a
    provider quota has to offer the way out on screen, not only in the JSON."""

    def test_the_failure_path_renders_the_offer(self):
        self.assertIn("renderProviderSwitchOffer", MARKUP)
        self.assertIn("switchProviderAndRetry", MARKUP)

    def test_a_failed_poll_keeps_the_fields_the_offer_needs(self):
        """`{ error }` used to be all a failed poll returned, which threw away
        the very fields that make the offer possible."""
        self.assertIn("return { ...polled, error: polled.error", MARKUP)

    def test_the_switch_writes_the_app_wide_selection(self):
        """Not a private builder-only override: the same setting Settings
        shows, so the page and the settings screen cannot disagree about which
        model is writing agents."""
        self.assertIn("'/api/system/llm/selection'", MARKUP)
        self.assertIn("company_id: alternative.company_id", MARKUP)

    def test_the_switch_is_announced(self):
        """Changing the model that writes someone's agent is not something to
        do quietly, even when they asked for it."""
        self.assertIn("quota_switched", MARKUP)
        self.assertEqual(MARKUP.count("quota_switched:"), 2)

    def test_it_only_offers_on_a_quota_failure(self):
        self.assertIn("result.error_kind !== 'quota'", MARKUP)


if __name__ == "__main__":
    unittest.main()
