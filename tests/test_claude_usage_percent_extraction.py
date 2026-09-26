"""`/usage` prints the limits on labelled lines and then a page of other percentages.

The extractor returned 91 for the output below — "91% of your usage was at
>150k context", a share of requests — while the weekly limit stood at 51%.
"""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm.claude_usage import _extract_usage_percent_from_text  # noqa: E402

REAL_OUTPUT = """You are currently using your subscription to power your Claude Code usage

Current session: 63% used · resets Sep 20 at 1pm (Asia/Seoul)
Current week (all models): 51% used · resets Sep 24 at 6pm (Asia/Seoul)
Current week (Fable): 53% used · resets Sep 24 at 6pm (Asia/Seoul)

What's contributing to your limits usage?
Approximate, based on local sessions on this machine.

Last 24h · 4674 requests · 19 sessions
  91% of your usage was at >150k context
  87% of your usage was while 4+ sessions ran in parallel
"""


class ClaudeUsagePercentExtractionTest(unittest.TestCase):
    def test_weekly_all_models_line_wins_over_explanatory_percentages(self):
        self.assertEqual(_extract_usage_percent_from_text(REAL_OUTPUT), 51.0)

    def test_falls_back_to_a_per_model_week_line(self):
        text = "Current session: 10% used\nCurrent week (Sonnet): 22.5% used · resets Mon\n"
        self.assertEqual(_extract_usage_percent_from_text(text), 22.5)

    def test_explanatory_sentences_alone_yield_nothing(self):
        text = "Last 7d\n  91% of your usage was at >150k context\n"
        self.assertIsNone(_extract_usage_percent_from_text(text))

    def test_older_unlabelled_output_still_parses(self):
        self.assertEqual(_extract_usage_percent_from_text("Weekly limit: 37% used"), 37.0)


if __name__ == "__main__":
    unittest.main()
