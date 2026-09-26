"""What counts as "out of quota", and — more importantly — what does not.

The classifier exists so the builder can offer a different backend when the
selected one is out of allowance. Its whole value depends on being *wrong in
one direction only*: a genuine defect misread as a quota problem sends the user
provider-shopping for a bug that will follow them, which is worse than the dead
end this feature is meant to remove. So the negative table below is the
important half of this file.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm.provider_errors import (  # noqa: E402
    PROVIDER_ERROR_QUOTA,
    classify_provider_error,
)


class ClassifyProviderErrorTest(unittest.TestCase):
    def test_recognises_real_provider_quota_messages(self):
        # Sample 1 is verbatim what the Codex CLI printed on this machine on
        # 2026-08-23 — the failure that started all of this.
        samples = [
            "You've hit your usage limit. Upgrade to Plus to continue using "
            "Codex (https://chatgpt.com/explore/plus), or try again at "
            "Sep 15th, 2026",
            "Claude usage limit reached. Your limit will reset at 3pm.",
            "Your credit balance is too low to access the Anthropic API.",
            "429 Resource has been exhausted (e.g. check quota).",
            "RESOURCE_EXHAUSTED",
            "Quota exceeded for quota metric 'Generate requests'",
            "Error code: 429 - insufficient_quota",
            "Rate limit reached for gpt-5 in organization org-x",
            "You exceeded your current quota, please check your plan.",
        ]
        for message in samples:
            with self.subTest(message=message[:48]):
                self.assertEqual(
                    classify_provider_error(message),
                    PROVIDER_ERROR_QUOTA,
                )

    def test_makes_no_claim_about_other_failures(self):
        """Anything that is not plainly an allowance problem returns None.

        None is not "healthy" — the turn still fails and still reports the
        provider's own words. It only means no *offer* is attached, because
        switching provider would not fix any of these.
        """
        samples = [
            "LLM provider returned an error",
            "Configurator LLM timed out.",
            "400 Bad Request: unknown field 'flow_json'",
            "model 'gpt-9' does not exist",
            "Invalid API key provided",
            "unexpected EOF while parsing JSON output",
            "Traceback (most recent call last): KeyError: 'result'",
            "Permission denied while reading /etc/shadow",
            "",
        ]
        for message in samples:
            with self.subTest(message=message[:48]):
                self.assertIsNone(classify_provider_error(message))

    def test_none_and_non_string_inputs_are_safe(self):
        self.assertIsNone(classify_provider_error(None))

    def test_a_missing_cli_is_never_a_quota_problem(self):
        """"Install it" and "wait for it" are different instructions.

        A not-installed message can still contain an allowance-ish word (the
        install hint naming a paid plan, say). It must lose to the negative
        rule, or the page would offer to switch away from a provider that was
        never there rather than saying it is missing.
        """
        samples = [
            "codex: command not found",
            "The Codex CLI is not installed. Upgrade to Plus for more usage.",
            "no such file or directory: /usr/local/bin/gemini",
            "'claude' is not recognized as an internal or external command",
        ]
        for message in samples:
            with self.subTest(message=message[:48]):
                self.assertIsNone(classify_provider_error(message))


if __name__ == "__main__":
    unittest.main()
