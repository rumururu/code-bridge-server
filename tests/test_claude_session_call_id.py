"""Coverage for ``ClaudeSession._enrich_with_call_id``.

The Anthropic Messages API places the tool call identifier on the first
content block of an ``assistant`` event (``content[0].id`` for tool_use,
``content[0].tool_use_id`` for tool_result). ``_enrich_with_call_id``
surfaces that identifier as a top-level ``call_id`` so the generic
``AgentTaskRunSink`` can persist it without provider-specific knowledge.

These tests exercise the helper directly without spinning up the actual
Claude SDK subprocess (the helper is a pure function on a static method).
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm.claude_session import ClaudeSession  # noqa: E402


class ClaudeEnrichWithCallIdTest(unittest.TestCase):
    """Direct unit tests for the static enrichment helper."""

    # Case 1
    def test_non_assistant_event_passes_through_unchanged(self) -> None:
        for event in (
            {"type": "result", "result": "done"},
            {"type": "control_request", "request": {"subtype": "can_use_tool"}},
            {"type": "output", "text": "hello"},
            {"type": "error", "error": {"message": "boom"}},
        ):
            with self.subTest(event_type=event["type"]):
                enriched = ClaudeSession._enrich_with_call_id(event)
                self.assertIs(enriched, event)
                self.assertNotIn("call_id", enriched)

    # Case 2
    def test_tool_use_block_with_id_gets_top_level_call_id(self) -> None:
        event = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc123",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
        }
        enriched = ClaudeSession._enrich_with_call_id(event)
        self.assertEqual(enriched["call_id"], "toolu_abc123")
        # Existing nested block id must still be present and untouched.
        self.assertEqual(enriched["message"]["content"][0]["id"], "toolu_abc123")

    # Case 3
    def test_tool_result_block_with_tool_use_id_gets_top_level_call_id(self) -> None:
        event = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc123",
                        "content": "ok",
                        "is_error": False,
                    }
                ],
            },
        }
        enriched = ClaudeSession._enrich_with_call_id(event)
        self.assertEqual(enriched["call_id"], "toolu_abc123")
        self.assertEqual(
            enriched["message"]["content"][0]["tool_use_id"],
            "toolu_abc123",
        )

    # Case 4
    def test_empty_or_missing_content_returns_unchanged(self) -> None:
        cases = [
            {"type": "assistant", "message": {"role": "assistant", "content": []}},
            {"type": "assistant", "message": {"role": "assistant"}},  # no content key
            {"type": "assistant", "message": "not-a-dict"},
            {"type": "assistant"},  # no message key
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "Bash"}  # id missing
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_result", "content": "ok"}  # tool_use_id missing
                    ],
                },
            },
        ]
        for event in cases:
            with self.subTest(event=event):
                enriched = ClaudeSession._enrich_with_call_id(event)
                self.assertNotIn("call_id", enriched)

    # Case 5
    def test_original_event_dict_is_not_mutated(self) -> None:
        event = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_immut",
                        "name": "Bash",
                        "input": {"command": "echo"},
                    }
                ],
            },
        }
        snapshot = copy.deepcopy(event)
        enriched = ClaudeSession._enrich_with_call_id(event)
        # Helper must return a new dict, not the original.
        self.assertIsNot(enriched, event)
        # Original event keys + nested structure must be unchanged.
        self.assertEqual(event, snapshot)
        # And the new top-level field must only exist on the returned copy.
        self.assertNotIn("call_id", event)
        self.assertEqual(enriched["call_id"], "toolu_immut")

    # Bonus — guards against ``None`` or non-string id slipping through.
    def test_non_string_id_returns_unchanged(self) -> None:
        for bad_id in (None, 42, "", {"nested": "id"}):
            with self.subTest(id=bad_id):
                event = {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": bad_id, "name": "Bash"}
                        ],
                    },
                }
                enriched = ClaudeSession._enrich_with_call_id(event)
                self.assertNotIn("call_id", enriched)


if __name__ == "__main__":
    unittest.main()
