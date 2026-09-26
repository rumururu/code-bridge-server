"""Workflow proposals for a saved agent, and the gate they must pass.

T-I1-18 gives the canvas an AI entry point without a fifth canvas route;
T-I1-19 is the property these tests actually defend: **no proposal reaches a
person unvalidated.** A proposal that fails to normalize, or that carries
blocking contract findings, is dropped and *named* in `dropped` — never
repaired into something nobody wrote, and never silently missing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core import graph_suggest  # noqa: E402


def _proposal_block(summary: str, flow_json: str) -> str:
    return (
        "```proposal\n"
        f'{{"summary": "{summary}", "flow": {flow_json}}}\n'
        "```\n"
    )


_VALID_FLOW = (
    '[{"id": "check", "type": "llm", "name": "Check",'
    ' "instruction": "look at the thing"},'
    ' {"id": "report", "type": "notify",'
    ' "notify": {"title": "done", "body": "ok"}}]'
)


class ParseTest(unittest.TestCase):
    def test_proposals_and_assessment_come_out_of_their_blocks(self) -> None:
        text = (
            "here is prose the model was told not to write\n"
            + _proposal_block("보수적 제안", _VALID_FLOW)
            + _proposal_block("과감한 제안", _VALID_FLOW)
            + '```assessment\n{"open_questions": ["알림 채널은요?"]}\n```\n'
        )
        parsed = graph_suggest.parse_suggest_response(text)
        self.assertEqual(
            [p["summary"] for p in parsed.proposals], ["보수적 제안", "과감한 제안"]
        )
        self.assertEqual(parsed.assessment.open_questions, ["알림 채널은요?"])
        self.assertEqual(parsed.warnings, [])

    def test_invalid_json_is_a_warning_not_a_substitute(self) -> None:
        parsed = graph_suggest.parse_suggest_response(
            "```proposal\n{not json}\n```"
        )
        self.assertEqual(parsed.proposals, [])
        self.assertTrue(any("invalid" in w for w in parsed.warnings))

    def test_a_proposal_without_a_flow_is_refused(self) -> None:
        parsed = graph_suggest.parse_suggest_response(
            '```proposal\n{"summary": "flow 없음"}\n```'
        )
        self.assertEqual(parsed.proposals, [])
        self.assertTrue(any("without a flow" in w for w in parsed.warnings))

    def test_extras_past_the_cap_are_dropped_loudly(self) -> None:
        text = "".join(
            _proposal_block(f"p{i}", _VALID_FLOW)
            for i in range(graph_suggest.MAX_PROPOSALS + 2)
        )
        parsed = graph_suggest.parse_suggest_response(text)
        self.assertEqual(len(parsed.proposals), graph_suggest.MAX_PROPOSALS)
        self.assertTrue(any("extras dropped" in w for w in parsed.warnings))


class ValidateTest(unittest.TestCase):
    def test_a_valid_proposal_survives_with_its_normalized_flow(self) -> None:
        survivors, dropped = graph_suggest.validate_proposals(
            [
                {
                    "summary": "ok",
                    "flow": [
                        {
                            "id": "check",
                            "type": "llm",
                            "name": "Check",
                            "instruction": "look",
                        }
                    ],
                }
            ]
        )
        self.assertEqual(dropped, [])
        self.assertEqual(len(survivors), 1)
        # Normalized, not verbatim: the reader reviews what a save would
        # store, byte for byte.
        step = survivors[0].flow[0]
        self.assertEqual(step["type"], "llm")
        self.assertIn("on_failure", step)

    def test_a_proposal_the_normalizer_refuses_is_dropped_and_named(self) -> None:
        survivors, dropped = graph_suggest.validate_proposals(
            [
                {
                    "summary": "외계 필드",
                    # `device_id` is not a field of an llm step — the exact
                    # class of drift the normalizer refuses.
                    "flow": [
                        {
                            "id": "s1",
                            "type": "llm",
                            "name": "bad",
                            "device_id": "R3CX",
                        }
                    ],
                }
            ]
        )
        self.assertEqual(survivors, [])
        self.assertEqual(len(dropped), 1)
        self.assertIn("외계 필드", dropped[0])

    def test_an_empty_flow_is_dropped(self) -> None:
        survivors, dropped = graph_suggest.validate_proposals(
            [{"summary": "빈 흐름", "flow": []}]
        )
        self.assertEqual(survivors, [])
        self.assertIn("빈 흐름: empty workflow", dropped)

    def test_a_blocking_contract_finding_drops_the_proposal(self) -> None:
        # A browser action typing a {{binding}} nothing ever extracted is the
        # measured blocking case (`unresolved_browser_target`) — a step that
        # would run, report success, and type nothing.
        survivors, dropped = graph_suggest.validate_proposals(
            [
                {
                    "summary": "묶이지 않은 바인딩",
                    "flow": [
                        {
                            "id": "s1",
                            "type": "browser_action",
                            "name": "type it",
                            "actions": [
                                {"type": "navigate", "url": "https://example.com"},
                                {
                                    "type": "type",
                                    "selector": "#q",
                                    "text": "{{never_bound}}",
                                },
                            ],
                        }
                    ],
                }
            ],
        )
        self.assertEqual(survivors, [])
        self.assertEqual(len(dropped), 1)
        self.assertIn("묶이지 않은 바인딩", dropped[0])
        self.assertIn("unresolved_browser_target", dropped[0])


class PromptTest(unittest.TestCase):
    def test_the_prompt_carries_the_generated_vocabulary_not_a_retyped_one(
        self,
    ) -> None:
        prompt = graph_suggest.build_suggest_prompt(
            agent_name="테스트 봇",
            agent_system_prompt="지켜본다",
            flow=[{"id": "s1", "type": "notify", "name": "알림"}],
            intent="실패하면 나한테 알려줘",
        )
        # The same generated blocks the Configurator uses — every step type
        # must appear, none retyped by hand here.
        for step_type in ("shell", "llm", "notify", "condition", "mcp_tool"):
            self.assertIn(f'"{step_type}"', prompt)
        self.assertIn("실패하면 나한테 알려줘", prompt)
        self.assertIn("```proposal", prompt)
        self.assertIn("```assessment", prompt)
        # The current flow rides along as JSON.
        self.assertIn('"id": "s1"', prompt)


class RouteGateTest(unittest.IsolatedAsyncioTestCase):
    """The route runs parse → gate and reports drops; the LLM is scripted."""

    async def test_the_route_returns_survivors_and_names_the_dropped(self) -> None:
        from routes import agents as agents_routes

        scripted = (
            _proposal_block("살아남을 제안", _VALID_FLOW)
            + _proposal_block(
                "죽을 제안",
                '[{"id": "s1", "type": "llm", "name": "bad", "device_id": "x"}]',
            )
            + '```assessment\n{"open_questions": ["언제 돌릴까요?"]}\n```'
        )

        fake_agent = {
            "id": "agent_x",
            "name": "테스트 봇",
            "system_prompt": "지켜본다",
            "flow_json": [],
            "is_pseudo": False,
        }

        with (
            mock.patch.object(
                agents_routes, "_require_agent", return_value=fake_agent
            ),
            mock.patch.object(
                agents_routes, "create_chat_session", new=mock.AsyncMock()
            ),
            mock.patch.object(
                agents_routes,
                "_collect_llm_response_text",
                new=mock.AsyncMock(return_value=scripted),
            ),
        ):
            payload = await agents_routes.suggest_agent_graph(
                "agent_x",
                agents_routes.GraphSuggestBody(intent="알림 붙여줘"),
            )

        self.assertEqual(len(payload["proposals"]), 1)
        self.assertEqual(payload["proposals"][0]["summary"], "살아남을 제안")
        # The normalized flow is what a save would store.
        self.assertEqual(payload["proposals"][0]["flow"][0]["id"], "check")
        self.assertEqual(len(payload["dropped"]), 1)
        self.assertIn("죽을 제안", payload["dropped"][0])
        self.assertEqual(
            payload["assessment"]["open_questions"], ["언제 돌릴까요?"]
        )
        self.assertIn("flow_revision", payload)


if __name__ == "__main__":
    unittest.main()
