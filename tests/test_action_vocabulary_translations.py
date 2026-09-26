"""A published `ko` that is the English text is worse than no `ko` at all.

`agent.action_vocabulary._localized` used to fill a missing translation with
the English string, so the published payload always had both an `en` and a
`ko` key — even for the twenty-six `ActionType.note` entries, which had no
`note_ko` field to begin with and so were *always* published with `help.ko ==
help.en`. A client cannot tell that from a real (if short) translation: the
canvas renders both as Korean, and shows English on a screen that is
otherwise in Korean.

These tests read the two published vocabularies (`BROWSER_ACTION_VOCABULARY`,
`APP_ACTION_VOCABULARY`) the same way `GET
/api/dashboard/agent/workflow/step-schema` does — through `to_dict()` — and
fail if any entry's Korean help is missing or identical to its English help.
They intentionally do not touch `label`, which was already translated for all
26 actions before this fix.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.app_action_executor import APP_ACTION_VOCABULARY  # noqa: E402
from agent.browser_action_adapter import BROWSER_ACTION_VOCABULARY  # noqa: E402

#: Every action across both executors, tagged with the vocabulary it came
#: from so a failure names the file to fix.
_ALL_ACTIONS: tuple[tuple[str, object], ...] = tuple(
    ("browser", action) for action in BROWSER_ACTION_VOCABULARY
) + tuple(("device", action) for action in APP_ACTION_VOCABULARY)

#: No entry in either table is exempt today: every action has a real English
#: sentence for its note, and every translated one is expected to say
#: something different in Korean, not just repeat a JSON literal. If a future
#: action's *entire* help text turns out to be nothing but a bare JSON
#: example with no English prose around it (so there is nothing to translate
#: and `en == ko` would be legitimate), add `(vocabulary, action_type)` here
#: with a comment saying why — do not weaken the assertion below instead.
_ALLOWED_IDENTICAL_HELP: frozenset[tuple[str, str]] = frozenset()
_ALLOWED_IDENTICAL_PARAM_HELP: frozenset[tuple[str, str, str]] = frozenset()


class ActionHelpIsTranslatedTest(unittest.TestCase):
    """`help.ko` is the substance a developer reads to author an action list
    item in the canvas — it must actually be Korean, not English wearing a
    `ko` label."""

    def test_every_action_publishes_a_korean_help(self) -> None:
        missing = [
            f"{vocab}/{action.type}"
            for vocab, action in _ALL_ACTIONS
            if "ko" not in action.to_dict()["help"]
        ]
        self.assertFalse(
            missing,
            f"no Korean help published at all for: {missing} — "
            "add ActionType(..., note_ko=...)",
        )

    def test_no_action_help_is_english_wearing_a_ko_label(self) -> None:
        identical = [
            f"{vocab}/{action.type}"
            for vocab, action in _ALL_ACTIONS
            if (vocab, action.type) not in _ALLOWED_IDENTICAL_HELP
            and action.to_dict()["help"].get("ko") == action.to_dict()["help"]["en"]
        ]
        self.assertFalse(
            identical,
            f"help.ko is byte-identical to help.en for: {identical} — "
            "translate ActionType.note_ko instead of leaving it as the "
            "English note",
        )

    def test_every_action_label_is_still_translated(self) -> None:
        """The labels were never the defect — pin that a future edit cannot
        regress them while touching help text."""
        missing = [
            f"{vocab}/{action.type}"
            for vocab, action in _ALL_ACTIONS
            if "ko" not in action.to_dict()["label"]
        ]
        self.assertFalse(missing, f"label lost its Korean translation: {missing}")


class ActionParamNotesAreTranslatedTest(unittest.TestCase):
    """The same check one level down: a parameter's `help` (e.g. `extract`'s
    `pattern`, `wait`'s `timeout_ms`) is the only place a developer learns
    what the key actually does."""

    def test_no_param_help_is_english_wearing_a_ko_label(self) -> None:
        identical = []
        for vocab, action in _ALL_ACTIONS:
            for param_dict in action.to_dict()["params"]:
                if param_dict["help"] is None:
                    continue
                key = (vocab, action.type, param_dict["key"])
                if key in _ALLOWED_IDENTICAL_PARAM_HELP:
                    continue
                if param_dict["help"].get("ko") == param_dict["help"]["en"]:
                    identical.append(f"{vocab}/{action.type}.{param_dict['key']}")
        self.assertFalse(
            identical,
            f"param help.ko is byte-identical to help.en for: {identical} — "
            "translate ActionParam(help_ko=...)",
        )

    def test_no_param_help_is_missing_a_korean_translation(self) -> None:
        missing = []
        for vocab, action in _ALL_ACTIONS:
            for param_dict in action.to_dict()["params"]:
                if param_dict["help"] is None:
                    continue
                if "ko" not in param_dict["help"]:
                    missing.append(f"{vocab}/{action.type}.{param_dict['key']}")
        self.assertFalse(
            missing,
            f"no Korean translation published for param help: {missing} — "
            "add ActionParam(help_ko=...)",
        )


class MissingTranslationDegradesToEnglishTest(unittest.TestCase):
    """Pin the shape of the gap-detection fix itself: an untranslated entry
    omits `ko` rather than repeating `en` under that key, and `en` is always
    present so a client with no fallback logic at all still has something to
    show."""

    def test_localized_omits_ko_when_untranslated(self) -> None:
        from agent.action_vocabulary import _localized

        published = _localized("English only", None)
        self.assertEqual(published, {"en": "English only"})
        self.assertNotIn("ko", published)

    def test_localized_publishes_ko_when_translated(self) -> None:
        from agent.action_vocabulary import _localized

        published = _localized("English", "한국어")
        self.assertEqual(published, {"en": "English", "ko": "한국어"})


if __name__ == "__main__":
    unittest.main()
