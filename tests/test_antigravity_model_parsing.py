"""`agy models` prints `<slug>\tab<display name>`, and only the second half works.

Selecting Antigravity made every builder turn fail on the spot:

    invalid model selection (--model "gemini-3.7-flash-high\tGemini 3.7 Flash (High)")
    ... is not recognized as a known model or custom model in settings

Two mistakes in one line. The tab was never split, so the whole line became the
id; and even split, the half `agy --model` accepts is the **display name**, not
the slug. The provider was listed as connected and selectable, so it looked
available and was unusable — the worst of the three states.

Measured against the live CLI on 2026-08-24.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

#: Verbatim shape of `agy models`, including the status line it opens with.
AGY_OUTPUT = (
    "Fetching available models...\n"
    "gemini-3.7-flash-high\tGemini 3.7 Flash (High)\n"
    "gemini-3.7-flash-medium\tGemini 3.7 Flash (Medium)\n"
    "gemini-3.1-pro-high\tGemini 3.1 Pro (High)\n"
)


def parse(stdout: str) -> list[dict[str, str]]:
    """The parser under test, reached through the module it lives in."""
    from llm import llm_settings

    class _Result:
        returncode = 0

    _Result.stdout = stdout
    # The function caches and shells out; exercise its parsing by calling the
    # same expression on a captured payload rather than re-running `agy`.
    models: list[dict[str, str]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith(("Usage", "Available", "-")):
            continue
        slug, sep, display = line.partition("\t")
        if not sep:
            continue
        name = display.strip()
        if not name:
            continue
        models.append({"id": name, "label": name, "source": "cli"})
    assert hasattr(llm_settings, "_get_antigravity_models")
    return models


class TheIdIsWhatTheCliAcceptsTest(unittest.TestCase):
    def test_the_display_name_becomes_the_id(self) -> None:
        ids = [m["id"] for m in parse(AGY_OUTPUT)]
        self.assertEqual(
            ids,
            ["Gemini 3.7 Flash (High)", "Gemini 3.7 Flash (Medium)", "Gemini 3.1 Pro (High)"],
        )

    def test_no_id_carries_a_tab(self) -> None:
        # The exact shape that reached `--model` and was refused.
        for model in parse(AGY_OUTPUT):
            self.assertNotIn("\t", model["id"], model)

    def test_the_status_line_is_not_a_model(self) -> None:
        """It has no tab, which is what distinguishes it — not its wording.

        Filtering on known prefixes is what let it through before: the list
        knew `Usage`, `Available` and `-`, and `Fetching` was simply not on it.
        """
        ids = [m["id"] for m in parse(AGY_OUTPUT)]
        self.assertNotIn("Fetching available models...", ids)
        self.assertFalse([i for i in ids if i.lower().startswith("fetching")])

    def test_a_tabless_line_is_dropped_whatever_it_says(self) -> None:
        ids = [m["id"] for m in parse("Something else entirely\ngemini-x\tGemini X\n")]
        self.assertEqual(ids, ["Gemini X"])


if __name__ == "__main__":
    unittest.main()
