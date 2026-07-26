"""The SDK→event-dict contract the chat stream depends on.

``chat_stream_service`` dispatches on ``event["type"]`` and reaches into
specific fields (``message.content`` blocks, ``result``'s duration/turn
counts). The SDK hands back typed objects instead, so these tests pin the
translation: a drift here silently breaks tool rendering or turn completion in
the app rather than raising anywhere on the server.

Shapes were captured from a live ``claude-agent-sdk`` turn, not written from
the type definitions.
"""

import unittest

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from llm.claude_sdk_events import block_to_dict, message_to_event, session_id_of


class BlockConversionTest(unittest.TestCase):
    def test_text_block(self):
        self.assertEqual(
            block_to_dict(TextBlock(text="hello")),
            {"type": "text", "text": "hello"},
        )

    def test_thinking_block_keeps_signature(self):
        block = ThinkingBlock(thinking="reasoning", signature="sig")
        self.assertEqual(
            block_to_dict(block),
            {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
        )

    def test_tool_use_block_matches_what_the_stream_reads(self):
        # _emit_tool_use reads block["id"], ["name"], ["input"].
        block = ToolUseBlock(id="toolu_1", name="Bash", input={"command": "ls"})
        self.assertEqual(
            block_to_dict(block),
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Bash",
                "input": {"command": "ls"},
            },
        )

    def test_tool_result_block_omits_unset_optionals(self):
        converted = block_to_dict(ToolResultBlock(tool_use_id="toolu_1"))
        self.assertEqual(converted, {"type": "tool_result", "tool_use_id": "toolu_1"})

    def test_tool_result_block_carries_error_flag(self):
        block = ToolResultBlock(tool_use_id="toolu_1", content="boom", is_error=True)
        self.assertEqual(
            block_to_dict(block),
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "boom",
                "is_error": True,
            },
        )


class MessageConversionTest(unittest.TestCase):
    def test_assistant_message(self):
        message = AssistantMessage(
            content=[TextBlock(text="OK")],
            model="claude-sonnet-4-5",
            usage={"input_tokens": 3},
            message_id="msg_1",
            session_id="sess_1",
        )
        event = message_to_event(message)
        self.assertEqual(event["type"], "assistant")
        self.assertEqual(event["message"]["content"], [{"type": "text", "text": "OK"}])
        self.assertEqual(event["message"]["role"], "assistant")
        self.assertEqual(event["message"]["id"], "msg_1")
        self.assertEqual(event["session_id"], "sess_1")

    def test_user_message_with_tool_result(self):
        message = UserMessage(content=[ToolResultBlock(tool_use_id="toolu_1", content="ok")])
        event = message_to_event(message)
        self.assertEqual(event["type"], "user")
        self.assertEqual(
            event["message"]["content"],
            [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        )

    def test_user_message_with_plain_text(self):
        event = message_to_event(UserMessage(content="hi"))
        self.assertEqual(event["message"]["content"], "hi")

    def test_result_message_carries_the_fields_the_stream_reads(self):
        message = ResultMessage(
            subtype="success",
            duration_ms=4023,
            duration_api_ms=3900,
            is_error=False,
            num_turns=1,
            session_id="sess_1",
            total_cost_usd=0.01,
            usage={"output_tokens": 5},
            result="OK",
        )
        event = message_to_event(message)
        # _handle_result_event reads exactly these.
        for key in ("duration_ms", "duration_api_ms", "num_turns", "result"):
            self.assertIn(key, event)
        self.assertEqual(event["type"], "result")
        self.assertIs(event["is_error"], False)

    def test_system_message_spreads_raw_payload(self):
        message = SystemMessage(
            subtype="init",
            data={"session_id": "sess_1", "tools": ["Bash"], "cwd": "/tmp"},
        )
        event = message_to_event(message)
        self.assertEqual(event["type"], "system")
        self.assertEqual(event["subtype"], "init")
        # Raw fields survive so nothing the app already reads is dropped.
        self.assertEqual(event["tools"], ["Bash"])
        self.assertEqual(event["cwd"], "/tmp")

    def test_unknown_message_degrades_to_a_dict(self):
        class Surprise:
            def __init__(self):
                self.session_id = "sess_1"

        event = message_to_event(Surprise())
        self.assertIsInstance(event, dict)
        self.assertIn("type", event)


class SessionIdTest(unittest.TestCase):
    def test_reads_attribute(self):
        message = AssistantMessage(content=[], model="m", session_id="sess_1")
        self.assertEqual(session_id_of(message), "sess_1")

    def test_falls_back_to_system_payload(self):
        message = SystemMessage(subtype="init", data={"session_id": "sess_2"})
        self.assertEqual(session_id_of(message), "sess_2")

    def test_none_when_absent_or_blank(self):
        self.assertIsNone(session_id_of(SystemMessage(subtype="init", data={})))
        self.assertIsNone(
            session_id_of(AssistantMessage(content=[], model="m", session_id="  "))
        )


if __name__ == "__main__":
    unittest.main()
