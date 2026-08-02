"""The Antigravity CLI speaks a different envelope, and the app must not care.

`agy --output-format stream-json` discriminates on ``event`` rather than
``type``, puts assistant text in ``step_update.text_delta`` rather than in a
message, and nests its answer at ``result.response``. Reading it with the
Gemini normaliser produces nothing at all — which is what shipped first: the
provider appeared in the picker, a turn ran, and the draft stayed empty with
"응답을 생성하고 있습니다" the only thing on screen.
"""

import json
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm.antigravity_session import AntigravitySession
from llm.llm_session import LlmSessionFactory


def _session(model: str | None = "gemini-3.1-pro-high") -> AntigravitySession:
    return AntigravitySession(project_path="/tmp", model=model)


def _init(conversation_id: str = "conv_1") -> dict:
    return {
        "event": "init",
        "conversation_id": conversation_id,
        "init": {"model": "gemini-3.1-pro-high", "cwd": "/tmp"},
    }


def _delta(text: str, conversation_id: str = "conv_1") -> dict:
    return {
        "event": "step_update",
        "step_update": {
            "conversation_id": conversation_id,
            "step_index": 2,
            "state": "DONE",
            "step_type": "agent_response",
            "text_delta": text,
        },
    }


def _result(
    response: str,
    status: str = "SUCCESS",
    conversation_id: str = "conv_1",
    **extra,
) -> dict:
    payload = {
        "conversation_id": conversation_id,
        "status": status,
        "response": response,
    }
    payload.update(extra)
    return {"event": "result", "result": payload}


class NormalisationTest(unittest.TestCase):
    def test_init_captures_the_conversation_id(self):
        session = _session()
        self.assertIsNone(session._normalize_event(_init("abc")))
        self.assertEqual(session.session_id, "abc")

    def test_text_delta_becomes_an_assistant_message(self):
        session = _session()
        event = session._normalize_event(_delta("Hello"))
        self.assertEqual(event["type"], "assistant")
        self.assertEqual(
            event["message"]["content"], [{"type": "text", "text": "Hello"}]
        )

    def test_deltas_accumulate(self):
        session = _session()
        session._normalize_event(_delta("Hel"))
        session._normalize_event(_delta("lo"))
        self.assertEqual(session._full_response_text, "Hello")

    def test_a_step_without_text_is_not_an_event(self):
        # Most step_updates are bookkeeping; emitting them would flood the UI.
        session = _session()
        self.assertIsNone(
            session._normalize_event(
                {"event": "step_update", "step_update": {"step_type": "checkpoint"}}
            )
        )

    def test_result_carries_the_whole_answer(self):
        session = _session()
        session._normalize_event(_delta("partial"))
        event = session._normalize_event(_result("the full answer"))
        self.assertEqual(event["type"], "result")
        # The closing frame repeats everything; trusting it over the deltas
        # means a dropped frame cannot truncate the answer.
        self.assertEqual(event["result"], "the full answer")

    def test_usage_is_passed_through(self):
        session = _session()
        event = session._normalize_event(
            _result("ok", usage={"input_tokens": 10, "output_tokens": 2})
        )
        self.assertEqual(event["usage"]["output_tokens"], 2)

    def test_a_failed_result_is_an_error(self):
        session = _session()
        event = session._normalize_event(_result("", status="FAILED", error="quota"))
        self.assertEqual(event["type"], "error")
        self.assertIn("quota", event["error"]["message"])

    def test_an_unknown_frame_is_ignored_rather_than_guessed_at(self):
        session = _session()
        self.assertIsNone(session._normalize_event({"event": "telemetry"}))

    def test_a_real_transcript_produces_text_then_result(self):
        # Captured from `agy -p "Say only: OK" --output-format stream-json`.
        lines = [
            json.dumps(_init("6d49b582")),
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {"step_index": 0, "state": "DONE", "step_type": "user_input"},
                }
            ),
            json.dumps(_delta("OK\n", "6d49b582")),
            json.dumps(_result("OK\n", conversation_id="6d49b582")),
        ]
        session = _session()
        events = [
            e
            for e in (session._normalize_event(json.loads(line)) for line in lines)
            if e
        ]
        self.assertEqual([e["type"] for e in events], ["assistant", "result"])
        self.assertEqual(events[-1]["result"], "OK\n")
        self.assertEqual(session.session_id, "6d49b582")


class CommandTest(unittest.TestCase):
    def test_it_runs_print_mode_with_stream_json(self):
        cmd = _session()._build_command("hi")
        self.assertIn("--print", cmd)
        self.assertIn("stream-json", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("gemini-3.1-pro-high", cmd)

    def test_no_model_means_the_cli_default(self):
        self.assertNotIn("--model", _session(model=None)._build_command("hi"))

    def test_a_later_turn_continues_the_same_conversation(self):
        session = _session()
        session._normalize_event(_init("conv_9"))
        cmd = session._build_command("second turn")
        self.assertIn("--conversation", cmd)
        self.assertIn("conv_9", cmd)


class FactoryTest(unittest.TestCase):
    def test_the_factory_knows_the_provider(self):
        # Missing here was the actual failure: the provider was offered in the
        # picker while the session factory refused it as "not supported yet".
        session = LlmSessionFactory.create_session(
            provider_id="antigravity", project_path="/tmp", model=None
        )
        self.assertIsInstance(session, AntigravitySession)

    def test_it_is_listed_as_supported(self):
        self.assertIn("antigravity", LlmSessionFactory.get_supported_providers())


if __name__ == "__main__":
    unittest.main()
