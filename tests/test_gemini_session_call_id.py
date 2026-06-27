"""Coverage for top-level ``call_id`` in ``GeminiSession._normalize_event``.

Mirrors the codex coverage in ``test_event_pairing.py``: drive the
normalizer directly with raw event dicts (no Gemini CLI subprocess) and
assert the normalized envelope carries a top-level ``call_id`` so the
generic ``AgentTaskRunSink`` can persist it into ``agent_events.call_id``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm.gemini_session import GeminiSession  # noqa: E402


class GeminiNormalizeEventCallIdTest(unittest.TestCase):
    """Gemini ``_normalize_event`` must surface ``call_id`` (and keep content blocks)."""

    def _session(self) -> GeminiSession:
        return GeminiSession(project_path="/tmp/code-bridge", model="gemini-2.5-pro")

    # Case 1 — tool_use with primary ``id`` key
    def test_tool_use_with_id_key_surfaces_top_level_call_id(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "tool_call",
                "id": "gem_call_001",
                "name": "shell",
                "input": {"command": "ls"},
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["call_id"], "gem_call_001")
        block = normalized["message"]["content"][0]
        self.assertEqual(block["type"], "tool_use")
        # Nested block id must mirror the surfaced call_id for downstream pairing.
        self.assertEqual(block["id"], "gem_call_001")
        self.assertEqual(block["name"], "shell")
        self.assertEqual(block["input"], {"command": "ls"})

    # Case 2 — tool_use with ``tool_use_id`` fallback key
    def test_tool_use_with_tool_use_id_fallback(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "tool_invocation",  # any key containing "tool" but not "result"
                "tool_use_id": "gem_call_fallback",
                "name": "browser_navigate",
                "input": {"url": "https://example.com"},
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["call_id"], "gem_call_fallback")
        block = normalized["message"]["content"][0]
        self.assertEqual(block["id"], "gem_call_fallback")
        self.assertEqual(block["name"], "browser_navigate")

    # Case 3 — tool_use with no id keys -> call_id is None (still normalized)
    def test_tool_use_without_any_id_yields_none_call_id(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "tool_call",
                "name": "shell",
                "input": {},
            }
        )
        assert normalized is not None
        self.assertIn("call_id", normalized)
        self.assertIsNone(normalized["call_id"])
        block = normalized["message"]["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertIsNone(block["id"])

    # Case 4 — tool_result normalization carries call_id and content
    def test_tool_result_normalization_carries_call_id(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "tool_result",
                "tool_use_id": "gem_call_001",
                "result": "ok",
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["call_id"], "gem_call_001")
        block = normalized["message"]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "gem_call_001")
        self.assertEqual(block["content"], "ok")
        self.assertFalse(block["is_error"])

    # Case 5 — tool_result falls back to ``id`` and respects is_error flag
    def test_tool_result_falls_back_to_id_and_respects_is_error(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {
                "type": "tool_result_error",
                "id": "gem_call_err",
                "output": "boom",
                "is_error": True,
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["call_id"], "gem_call_err")
        block = normalized["message"]["content"][0]
        self.assertEqual(block["tool_use_id"], "gem_call_err")
        self.assertEqual(block["content"], "boom")
        self.assertTrue(block["is_error"])

    # Case 6 — regression guard: non-tool events still normalize like before
    def test_non_tool_event_does_not_gain_call_id(self) -> None:
        session = self._session()
        normalized = session._normalize_event(
            {"type": "content", "text": "hi"}
        )
        assert normalized is not None
        self.assertEqual(normalized["type"], "assistant")
        # No top-level call_id should be added on plain text/assistant events.
        self.assertNotIn("call_id", normalized)
        block = normalized["message"]["content"][0]
        self.assertEqual(block["type"], "text")
        self.assertEqual(block["text"], "hi")


if __name__ == "__main__":
    unittest.main()
