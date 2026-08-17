"""Round-trip and linear-subset tests for agent/flow_graph.py (T-B-02/03).

Pins the two round-trip propositions of the mapping spec (agent-flow-core
``docs/LINEAR_FLOW_MAPPING.md`` section 7):

1. ``from_graph(to_graph(L)) == L`` for every normalized linear workflow —
   proven here over the full stored-shape snapshot net of
   ``test_flow_json_snapshot_regression.py`` (the same goldens Phase 2 must
   not break) plus the spec's own section-10 examples.
2. ``to_graph(from_graph(g)) == g`` for canonical graphs (``to_graph``'s
   range), compared as ``model_dump(by_alias=True)`` wire dicts.

Also pins the from_graph discrimination rules: policies are the canon and
edges only a derived view, so edge/policy disagreement is rejected with the
exact ``linear.*`` issue codes of spec section 6.3 — never silently
approximated.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for path in (str(SERVER_DIR), str(TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from agent.flow_graph import (  # noqa: E402
    UnsupportedTopologyError,
    from_graph,
    to_graph,
)
from agent.workflow_v2 import ALLOWED_STEP_TYPES, normalize_workflow  # noqa: E402
from agent_flow_core.model import Flow  # noqa: E402
from agent_flow_core.validate import validate_flow  # noqa: E402
from test_flow_json_snapshot_regression import SNAPSHOTS  # noqa: E402


def _normalized(snapshot: list[dict]) -> list[dict]:
    return normalize_workflow(copy.deepcopy(snapshot))


def _issue_tuples(error: UnsupportedTopologyError) -> list[tuple]:
    return [
        (issue.code, issue.step_id, issue.edge_id, issue.detail)
        for issue in error.issues
    ]


# ---------------------------------------------------------------------------
# Spec section 10 example literals (docs/LINEAR_FLOW_MAPPING.md).
# ---------------------------------------------------------------------------

# 10.1 (a): 4-step browser_action x2 + llm + notify flow (cafe_id binding —
# data passing rides run scope and must not appear as edges).
SPEC_LINEAR_A: list[dict] = [
    {
        "id": "open_cafe",
        "type": "browser_action",
        "name": "카페 첫 화면 열기",
        "description": "카페 홈을 열고 내부 cafe_id를 추출한다.",
        "on_failure": {"type": "ask_user", "resume": "same_step"},
        "on_success": {"type": "continue"},
        "tool_hint": None,
        "success_criteria": "",
        "actions": [
            {"type": "navigate", "url": "https://cafe.naver.com/devsharing"},
            {
                "type": "extract",
                "name": "cafe_id",
                "source": "html",
                "pattern": "clubid=(\\d+)",
            },
        ],
    },
    {
        "id": "open_board",
        "type": "browser_action",
        "name": "게시판 목록 열기",
        "description": "추출한 cafe_id로 게시판 목록에 들어가 본문을 걷는다.",
        "on_failure": {"type": "ask_user", "resume": "same_step"},
        "on_success": {"type": "continue"},
        "tool_hint": None,
        "success_criteria": "",
        "actions": [
            {
                "type": "navigate",
                "url": "https://cafe.naver.com/f-e/cafes/{{cafe_id}}/menus/0",
            },
            {"type": "extract", "name": "board_text", "source": "text"},
        ],
    },
    {
        "id": "summarize",
        "type": "llm",
        "name": "새 글 요약",
        "description": "걷어온 게시판 본문에서 새 글을 요약한다.",
        "on_failure": {"type": "ask_user", "resume": "same_step"},
        "on_success": {"type": "continue"},
        "tool_hint": None,
        "success_criteria": "새 글 유무와 제목 목록이 요약에 포함된다",
        "actions": [],
        "instruction": (
            "이전 스텝이 걷어온 게시판 본문에서 오늘 올라온 글을 찾아 제목과 한"
            " 줄 요약을 만들어라."
        ),
    },
    {
        "id": "notify_user",
        "type": "notify",
        "name": "요약 알림",
        "description": "요약 결과를 사용자에게 알린다.",
        "on_failure": {"type": "ask_user", "resume": "same_step"},
        "on_success": {"type": "continue"},
        "tool_hint": None,
        "success_criteria": "",
        "actions": [],
        "notify": {
            "title": "카페 새 글 요약",
            "body": "오늘의 새 글 요약이 도착했습니다.",
            "level": "info",
        },
    },
]

# The full to_graph wire output the spec fixes for 10.1.
SPEC_GRAPH_A: dict = {
    "name": "",
    "description": "",
    "steps": [
        {
            "id": "open_cafe",
            "stepType": "browser_action",
            "name": "카페 첫 화면 열기",
            "description": "카페 홈을 열고 내부 cafe_id를 추출한다.",
            "connectorRef": None,
            "config": {
                "tool_hint": None,
                "success_criteria": "",
                "actions": [
                    {
                        "type": "navigate",
                        "url": "https://cafe.naver.com/devsharing",
                    },
                    {
                        "type": "extract",
                        "name": "cafe_id",
                        "source": "html",
                        "pattern": "clubid=(\\d+)",
                    },
                ],
            },
            "inputFields": [],
            "onFailure": {"type": "ask_user", "resume": "same_step"},
            "onSuccess": {"type": "continue"},
            "extensions": {},
        },
        {
            "id": "open_board",
            "stepType": "browser_action",
            "name": "게시판 목록 열기",
            "description": "추출한 cafe_id로 게시판 목록에 들어가 본문을 걷는다.",
            "connectorRef": None,
            "config": {
                "tool_hint": None,
                "success_criteria": "",
                "actions": [
                    {
                        "type": "navigate",
                        "url": (
                            "https://cafe.naver.com/f-e/cafes/{{cafe_id}}"
                            "/menus/0"
                        ),
                    },
                    {"type": "extract", "name": "board_text", "source": "text"},
                ],
            },
            "inputFields": [],
            "onFailure": {"type": "ask_user", "resume": "same_step"},
            "onSuccess": {"type": "continue"},
            "extensions": {},
        },
        {
            "id": "summarize",
            "stepType": "llm",
            "name": "새 글 요약",
            "description": "걷어온 게시판 본문에서 새 글을 요약한다.",
            "connectorRef": None,
            "config": {
                "tool_hint": None,
                "success_criteria": "새 글 유무와 제목 목록이 요약에 포함된다",
                "actions": [],
                "instruction": (
                    "이전 스텝이 걷어온 게시판 본문에서 오늘 올라온 글을 찾아"
                    " 제목과 한 줄 요약을 만들어라."
                ),
            },
            "inputFields": [],
            "onFailure": {"type": "ask_user", "resume": "same_step"},
            "onSuccess": {"type": "continue"},
            "extensions": {},
        },
        {
            "id": "notify_user",
            "stepType": "notify",
            "name": "요약 알림",
            "description": "요약 결과를 사용자에게 알린다.",
            "connectorRef": None,
            "config": {
                "tool_hint": None,
                "success_criteria": "",
                "actions": [],
                "notify": {
                    "title": "카페 새 글 요약",
                    "body": "오늘의 새 글 요약이 도착했습니다.",
                    "level": "info",
                },
            },
            "inputFields": [],
            "onFailure": {"type": "ask_user", "resume": "same_step"},
            "onSuccess": {"type": "continue"},
            "extensions": {},
        },
    ],
    "edges": [
        {
            "id": "open_cafe:success",
            "fromStepId": "open_cafe",
            "toStepId": "open_board",
            "fromField": None,
            "toField": None,
            "kind": "seq",
            "extensions": {"codeBridgeLinear": {"on": "success"}},
        },
        {
            "id": "open_board:success",
            "fromStepId": "open_board",
            "toStepId": "summarize",
            "fromField": None,
            "toField": None,
            "kind": "seq",
            "extensions": {"codeBridgeLinear": {"on": "success"}},
        },
        {
            "id": "summarize:success",
            "fromStepId": "summarize",
            "toStepId": "notify_user",
            "fromField": None,
            "toField": None,
            "kind": "seq",
            "extensions": {"codeBridgeLinear": {"on": "success"}},
        },
    ],
    "triggers": [],
    "extensions": {},
}

# 10.2 (b): retry.then goto (E4), end vs natural termination, and a
# goto-only-reachable diagnose step.
SPEC_LINEAR_B: list[dict] = [
    {
        "id": "nightly_check",
        "type": "browser_action",
        "name": "야간 가용성 점검",
        "description": "대상 페이지가 뜨는지 확인한다.",
        "on_failure": {
            "type": "retry",
            "max_attempts": 2,
            "then": {"type": "goto_step", "target_step_id": "diagnose"},
        },
        "on_success": {"type": "continue"},
        "tool_hint": None,
        "success_criteria": "",
        "actions": [
            {"type": "navigate", "url": "https://example.com"},
            {"type": "assert", "selector": "h1"},
        ],
    },
    {
        "id": "report_ok",
        "type": "notify",
        "name": "정상 알림",
        "description": "점검 성공을 알리고 여기서 끝낸다.",
        "on_failure": {"type": "ask_user", "resume": "same_step"},
        "on_success": {"type": "end"},
        "tool_hint": None,
        "success_criteria": "",
        "actions": [],
        "notify": {
            "title": "야간 점검 정상",
            "body": "example.com 정상 응답.",
            "level": "success",
        },
    },
    {
        "id": "diagnose",
        "type": "llm",
        "name": "실패 원인 진단",
        "description": "점검 실패의 원인을 로그에서 추정한다. goto로만 도달한다.",
        "on_failure": {"type": "abort"},
        "on_success": {"type": "continue"},
        "tool_hint": None,
        "success_criteria": "",
        "actions": [],
        "instruction": (
            "직전 browser_action 스텝의 에러를 보고 실패 원인 후보를 정리하라."
        ),
    },
    {
        "id": "report_fail",
        "type": "notify",
        "name": "장애 알림",
        "description": "진단 요약을 알린다.",
        "on_failure": {"type": "ask_user", "resume": "same_step"},
        "on_success": {"type": "continue"},
        "tool_hint": None,
        "success_criteria": "",
        "actions": [],
        "notify": {
            "title": "야간 점검 실패",
            "body": "진단 요약을 확인하세요.",
            "level": "error",
        },
    },
]

SPEC_EDGES_B: list[dict] = [
    {
        "id": "nightly_check:success",
        "fromStepId": "nightly_check",
        "toStepId": "report_ok",
        "fromField": None,
        "toField": None,
        "kind": "seq",
        "extensions": {"codeBridgeLinear": {"on": "success"}},
    },
    {
        "id": "nightly_check:failure",
        "fromStepId": "nightly_check",
        "toStepId": "diagnose",
        "fromField": None,
        "toField": None,
        "kind": "goto",
        "extensions": {"codeBridgeLinear": {"on": "failure", "via": "retry_then"}},
    },
    {
        "id": "diagnose:success",
        "fromStepId": "diagnose",
        "toStepId": "report_fail",
        "fromField": None,
        "toField": None,
        "kind": "seq",
        "extensions": {"codeBridgeLinear": {"on": "success"}},
    },
]

# 10.3 (c): default (implicit-sequential) policies but fan-out/merge edges —
# must be rejected, all issues at once.
SPEC_GRAPH_C: dict = {
    "name": "",
    "description": "",
    "steps": [
        {
            "id": "start",
            "stepType": "llm",
            "name": "준비",
            "description": "",
            "connectorRef": None,
            "config": {},
            "inputFields": [],
            "onFailure": {"type": "ask_user", "resume": "same_step"},
            "onSuccess": {"type": "continue"},
            "extensions": {},
        },
        {
            "id": "fetch_a",
            "stepType": "browser_action",
            "name": "소스 A 수집",
            "description": "",
            "connectorRef": None,
            "config": {
                "actions": [{"type": "navigate", "url": "https://a.example.com"}]
            },
            "inputFields": [],
            "onFailure": {"type": "ask_user", "resume": "same_step"},
            "onSuccess": {"type": "continue"},
            "extensions": {},
        },
        {
            "id": "fetch_b",
            "stepType": "browser_action",
            "name": "소스 B 수집",
            "description": "",
            "connectorRef": None,
            "config": {
                "actions": [{"type": "navigate", "url": "https://b.example.com"}]
            },
            "inputFields": [],
            "onFailure": {"type": "ask_user", "resume": "same_step"},
            "onSuccess": {"type": "continue"},
            "extensions": {},
        },
        {
            "id": "join",
            "stepType": "llm",
            "name": "합치기",
            "description": "",
            "connectorRef": None,
            "config": {},
            "inputFields": [],
            "onFailure": {"type": "ask_user", "resume": "same_step"},
            "onSuccess": {"type": "continue"},
            "extensions": {},
        },
    ],
    "edges": [
        {
            "id": "e1",
            "fromStepId": "start",
            "toStepId": "fetch_a",
            "fromField": None,
            "toField": None,
        },
        {
            "id": "e2",
            "fromStepId": "start",
            "toStepId": "fetch_b",
            "fromField": None,
            "toField": None,
        },
        {
            "id": "e3",
            "fromStepId": "fetch_a",
            "toStepId": "join",
            "fromField": None,
            "toField": None,
        },
        {
            "id": "e4",
            "fromStepId": "fetch_b",
            "toStepId": "join",
            "fromField": None,
            "toField": None,
        },
    ],
    "triggers": [],
    "extensions": {},
}


class SnapshotRoundTripTest(unittest.TestCase):
    """Proposition 1 over the full stored-shape snapshot net (10 flows)."""

    def test_snapshot_net_is_the_expected_ten(self) -> None:
        self.assertEqual(len(SNAPSHOTS), 10)

    def test_prop1_from_graph_to_graph_is_identity_on_normalized(self) -> None:
        for name, snapshot in SNAPSHOTS.items():
            with self.subTest(snapshot=name):
                linear = _normalized(snapshot)
                self.assertEqual(from_graph(to_graph(linear)), linear)

    def test_prop2_to_graph_from_graph_is_identity_on_canonical(self) -> None:
        # Canonical form == to_graph's range (spec section 7, proposition 2);
        # equality is on the camelCase wire dump.
        for name, snapshot in SNAPSHOTS.items():
            with self.subTest(snapshot=name):
                graph = to_graph(_normalized(snapshot))
                round_tripped = to_graph(from_graph(graph))
                self.assertEqual(
                    round_tripped.model_dump(by_alias=True),
                    graph.model_dump(by_alias=True),
                )

    def test_to_graph_does_not_mutate_its_input(self) -> None:
        for name, snapshot in SNAPSHOTS.items():
            with self.subTest(snapshot=name):
                linear = _normalized(snapshot)
                frozen = copy.deepcopy(linear)
                to_graph(linear)
                self.assertEqual(linear, frozen)

    def test_canonical_graphs_pass_the_kernel_gate(self) -> None:
        # Spec 3.2 (post section-9): assert_valid_flow is usable as a gate on
        # to_graph output — no error-severity issues on any snapshot graph.
        for name, snapshot in SNAPSHOTS.items():
            with self.subTest(snapshot=name):
                graph = to_graph(_normalized(snapshot))
                issues = validate_flow(
                    graph, allowed_step_types=ALLOWED_STEP_TYPES
                )
                self.assertEqual(
                    [i for i in issues if i.severity == "error"], []
                )


class SpecExampleATest(unittest.TestCase):
    """Spec 10.1 — 4-step flow: exact wire shape + round trip."""

    def test_linear_input_is_normalize_idempotent(self) -> None:
        self.assertEqual(_normalized(SPEC_LINEAR_A), SPEC_LINEAR_A)

    def test_to_graph_matches_the_spec_wire_json(self) -> None:
        graph = to_graph(_normalized(SPEC_LINEAR_A))
        self.assertEqual(graph.model_dump(by_alias=True), SPEC_GRAPH_A)

    def test_round_trip(self) -> None:
        linear = _normalized(SPEC_LINEAR_A)
        self.assertEqual(from_graph(to_graph(linear)), linear)


class SpecExampleBTest(unittest.TestCase):
    """Spec 10.2 — retry.then goto derives an E4 failure edge."""

    def test_edges_match_the_spec(self) -> None:
        graph = to_graph(_normalized(SPEC_LINEAR_B))
        dump = graph.model_dump(by_alias=True)
        self.assertEqual(dump["edges"], SPEC_EDGES_B)
        # Annotations ride on the edges; flow-level extensions stay empty.
        self.assertEqual(dump["extensions"], {})
        self.assertEqual(dump["triggers"], [])

    def test_retry_itself_is_not_an_edge(self) -> None:
        # report_ok (end) and report_fail (trailing continue) both have no
        # outgoing edge — only the node policies distinguish them.
        graph = to_graph(_normalized(SPEC_LINEAR_B))
        froms = [edge.from_step_id for edge in graph.edges]
        self.assertNotIn("report_ok", froms)
        self.assertNotIn("report_fail", froms)

    def test_round_trip_preserves_end_vs_natural_termination(self) -> None:
        linear = _normalized(SPEC_LINEAR_B)
        restored = from_graph(to_graph(linear))
        self.assertEqual(restored, linear)
        self.assertEqual(restored[1]["on_success"], {"type": "end"})
        self.assertEqual(restored[3]["on_success"], {"type": "continue"})

    def test_kernel_gate_accepts_the_goto_graph(self) -> None:
        graph = to_graph(_normalized(SPEC_LINEAR_B))
        issues = validate_flow(graph, allowed_step_types=ALLOWED_STEP_TYPES)
        self.assertEqual([i for i in issues if i.severity == "error"], [])

    def test_nested_retry_chain_still_derives_one_goto(self) -> None:
        linear = _normalized(
            [
                {
                    "id": "work",
                    "type": "llm",
                    "name": "Work",
                    "on_failure": {
                        "type": "retry",
                        "max_attempts": 2,
                        "then": {
                            "type": "retry",
                            "max_attempts": 1,
                            "then": {
                                "type": "goto_step",
                                "target_step_id": "rescue",
                            },
                        },
                    },
                },
                {"id": "rescue", "type": "llm", "name": "Rescue"},
            ]
        )
        graph = to_graph(linear)
        failure_edges = [
            e for e in graph.edges if e.id == "work:failure"
        ]
        self.assertEqual(len(failure_edges), 1)
        edge = failure_edges[0]
        self.assertEqual(edge.to_step_id, "rescue")
        self.assertEqual(edge.kind, "goto")
        self.assertEqual(
            edge.extensions["codeBridgeLinear"],
            {"on": "failure", "via": "retry_then"},
        )
        self.assertEqual(from_graph(graph), linear)


class SpecExampleCTest(unittest.TestCase):
    """Spec 10.3 — fan-out/merge graph is rejected with all issues at once."""

    def test_fan_out_graph_is_rejected_with_the_exact_codes(self) -> None:
        flow = Flow.model_validate(SPEC_GRAPH_C)
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(flow)
        self.assertEqual(
            sorted(_issue_tuples(ctx.exception), key=repr),
            sorted(
                [
                    (
                        "linear.edge_unbacked",
                        "start",
                        "e2",
                        {"fromStepId": "start", "toStepId": "fetch_b"},
                    ),
                    (
                        "linear.edge_unbacked",
                        "fetch_a",
                        "e3",
                        {"fromStepId": "fetch_a", "toStepId": "join"},
                    ),
                    (
                        "linear.edge_missing",
                        "fetch_a",
                        None,
                        {
                            "fromStepId": "fetch_a",
                            "toStepId": "fetch_b",
                            "on": "success",
                        },
                    ),
                ],
                key=repr,
            ),
        )
        for issue in ctx.exception.issues:
            self.assertEqual(issue.severity, "error")
        # The exception message names the offending nodes/edges (spec 6.2).
        self.assertIn("e2", str(ctx.exception))
        self.assertIn("e3", str(ctx.exception))


class BackwardGotoTest(unittest.TestCase):
    """A legal backward goto must not read as a kernel cycle (spec 3.2/9)."""

    LINEAR = [
        {"id": "a", "type": "llm", "name": "A"},
        {
            "id": "b",
            "type": "llm",
            "name": "B",
            "on_failure": {"type": "goto_step", "target_step_id": "a"},
        },
    ]

    def test_backward_goto_edge_kind_and_no_cycle(self) -> None:
        graph = to_graph(_normalized(self.LINEAR))
        backward = [e for e in graph.edges if e.id == "b:failure"]
        self.assertEqual(len(backward), 1)
        self.assertEqual(backward[0].to_step_id, "a")
        self.assertEqual(backward[0].kind, "goto")
        issues = validate_flow(graph, allowed_step_types=ALLOWED_STEP_TYPES)
        self.assertEqual([i.code for i in issues if i.code == "flow.cycle"], [])
        self.assertEqual([i for i in issues if i.severity == "error"], [])

    def test_backward_goto_round_trips(self) -> None:
        linear = _normalized(self.LINEAR)
        self.assertEqual(from_graph(to_graph(linear)), linear)


class EdgePolicyMismatchTest(unittest.TestCase):
    """Edges are a derived view; disagreement with policies is rejected."""

    LINEAR = [
        {"id": "a", "type": "llm", "name": "A"},
        {"id": "b", "type": "llm", "name": "B"},
    ]

    def _canonical_dump(self) -> dict:
        return to_graph(_normalized(self.LINEAR)).model_dump(by_alias=True)

    def test_unbacked_edge_is_rejected(self) -> None:
        dump = self._canonical_dump()
        dump["edges"].append(
            {
                "id": "sneaky",
                "fromStepId": "b",
                "toStepId": "a",
                "fromField": None,
                "toField": None,
            }
        )
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        self.assertEqual(
            _issue_tuples(ctx.exception),
            [
                (
                    "linear.edge_unbacked",
                    "b",
                    "sneaky",
                    {"fromStepId": "b", "toStepId": "a"},
                )
            ],
        )

    def test_missing_required_edge_is_rejected(self) -> None:
        dump = self._canonical_dump()
        dump["edges"] = []  # a's on_success continue still requires a -> b
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        self.assertEqual(
            _issue_tuples(ctx.exception),
            [
                (
                    "linear.edge_missing",
                    "a",
                    None,
                    {"fromStepId": "a", "toStepId": "b", "on": "success"},
                )
            ],
        )

    def test_contradictory_annotation_is_rejected(self) -> None:
        dump = self._canonical_dump()
        dump["edges"][0]["extensions"] = {"codeBridgeLinear": {"on": "failure"}}
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        self.assertEqual(
            [i.code for i in ctx.exception.issues],
            ["linear.edge_meta_mismatch"],
        )

    def test_explicit_goto_kind_on_a_seq_edge_is_rejected(self) -> None:
        dump = self._canonical_dump()
        dump["edges"][0]["kind"] = "goto"
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        self.assertEqual(
            [i.code for i in ctx.exception.issues],
            ["linear.edge_meta_mismatch"],
        )


class NonCanonicalAcceptanceTest(unittest.TestCase):
    """Structurally valid but non-canonical graphs are accepted (spec 7)."""

    LINEAR = [
        {"id": "a", "type": "llm", "name": "A"},
        {
            "id": "b",
            "type": "llm",
            "name": "B",
            "on_failure": {"type": "goto_step", "target_step_id": "a"},
        },
    ]

    def test_foreign_edge_ids_and_missing_annotations_are_accepted(self) -> None:
        linear = _normalized(self.LINEAR)
        dump = to_graph(linear).model_dump(by_alias=True)
        for index, edge in enumerate(dump["edges"]):
            edge["id"] = f"external-{index}"
            edge["extensions"] = {}
            edge["kind"] = "seq"  # omitted kind == the model default
        dump["name"] = "Envelope name (ignored)"
        dump["description"] = "Envelope description (ignored)"
        self.assertEqual(from_graph(Flow.model_validate(dump)), linear)

    def test_legacy_flow_level_edge_meta_is_accepted_when_consistent(
        self,
    ) -> None:
        linear = _normalized(self.LINEAR)
        dump = to_graph(linear).model_dump(by_alias=True)
        meta = {}
        for edge in dump["edges"]:
            annotation = dict(edge["extensions"]["codeBridgeLinear"])
            annotation["kind"] = edge["kind"]
            meta[edge["id"]] = annotation
            edge["extensions"] = {}
            edge["kind"] = "seq"
        dump["extensions"] = {"codeBridgeLinear": {"edgeMeta": meta}}
        self.assertEqual(from_graph(Flow.model_validate(dump)), linear)

    def test_legacy_edge_meta_contradiction_is_rejected(self) -> None:
        dump = to_graph(_normalized(self.LINEAR)).model_dump(by_alias=True)
        goto_edge = next(e for e in dump["edges"] if e["kind"] == "goto")
        dump["extensions"] = {
            "codeBridgeLinear": {"edgeMeta": {goto_edge["id"]: {"on": "success"}}}
        }
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        self.assertIn(
            "linear.edge_meta_mismatch",
            [i.code for i in ctx.exception.issues],
        )


class KernelOnlyFeatureRejectionTest(unittest.TestCase):
    """Kernel features with no linear slot are rejected, never dropped."""

    LINEAR = [{"id": "only", "type": "llm", "name": "Only"}]

    def _dump(self) -> dict:
        return to_graph(_normalized(self.LINEAR)).model_dump(by_alias=True)

    def _codes(self, dump: dict) -> list[str]:
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        return [i.code for i in ctx.exception.issues]

    def test_data_edge_fields_are_rejected(self) -> None:
        dump = self._dump()
        dump["steps"].append(
            copy.deepcopy(dump["steps"][0]) | {"id": "second", "name": "Second"}
        )
        dump["edges"] = [
            {
                "id": "only:success",
                "fromStepId": "only",
                "toStepId": "second",
                "fromField": "rows",
                "toField": "rows",
                "kind": "seq",
                "extensions": {"codeBridgeLinear": {"on": "success"}},
            }
        ]
        self.assertIn("linear.data_edge", self._codes(dump))

    def test_triggers_are_rejected(self) -> None:
        dump = self._dump()
        dump["triggers"] = [{"triggerType": "schedule", "config": {}}]
        self.assertEqual(self._codes(dump), ["linear.trigger_unsupported"])

    def test_step_extensions_are_rejected(self) -> None:
        dump = self._dump()
        dump["steps"][0]["extensions"] = {"ontologyBinding": {"x": 1}}
        self.assertEqual(self._codes(dump), ["linear.step_field_unsupported"])

    def test_connector_ref_and_input_fields_are_rejected(self) -> None:
        dump = self._dump()
        dump["steps"][0]["connectorRef"] = "conn-1"
        dump["steps"][0]["inputFields"] = [{"name": "x", "dataType": "string"}]
        self.assertEqual(
            self._codes(dump),
            ["linear.step_field_unsupported", "linear.step_field_unsupported"],
        )

    def test_flow_extensions_outside_namespace_are_rejected(self) -> None:
        dump = self._dump()
        dump["extensions"] = {"infergraph": {"x": 1}}
        self.assertEqual(
            self._codes(dump), ["linear.flow_extensions_unsupported"]
        )

    def test_config_collision_with_common_keys_is_rejected(self) -> None:
        dump = self._dump()
        dump["steps"][0]["config"]["name"] = "shadow"
        self.assertEqual(self._codes(dump), ["linear.config_field_collision"])

    def test_unknown_step_type_is_rejected(self) -> None:
        dump = self._dump()
        dump["steps"][0]["stepType"] = "spreadsheet"
        codes = self._codes(dump)
        self.assertIn("step.type_unknown", codes)

    def test_unfoldable_step_shape_fails_normalize(self) -> None:
        # A shell step without script_id passes the graph checks but the
        # fold's normalize_workflow rejects it (spec 6.1 rule 6 — the type
        # field contract is not re-invented here).
        dump = self._dump()
        dump["steps"][0]["stepType"] = "shell"
        self.assertEqual(self._codes(dump), ["linear.normalize_failed"])


class StructuralCornerTest(unittest.TestCase):
    def test_empty_flow_folds_to_the_empty_workflow(self) -> None:
        self.assertEqual(from_graph(Flow()), [])
        self.assertEqual(
            to_graph(normalize_workflow(None)).model_dump(by_alias=True),
            Flow().model_dump(by_alias=True),
        )

    def test_success_and_failure_edges_to_the_same_target(self) -> None:
        # E2+E3 with an identical target — the (from, to) comparison must be
        # a multiset (spec section 4).
        linear = _normalized(
            [
                {
                    "id": "gate",
                    "type": "llm",
                    "name": "Gate",
                    "on_success": {"type": "goto_step", "target_step_id": "wrap"},
                    "on_failure": {"type": "goto_step", "target_step_id": "wrap"},
                },
                {"id": "skipped", "type": "llm", "name": "Skipped"},
                {"id": "wrap", "type": "llm", "name": "Wrap"},
            ]
        )
        graph = to_graph(linear)
        pair_edges = [
            (e.from_step_id, e.to_step_id)
            for e in graph.edges
            if e.from_step_id == "gate"
        ]
        self.assertEqual(pair_edges, [("gate", "wrap"), ("gate", "wrap")])
        self.assertEqual(from_graph(graph), linear)
        # Dropping one of the two is a missing edge, not a silent merge.
        dump = graph.model_dump(by_alias=True)
        dump["edges"] = [e for e in dump["edges"] if e["id"] != "gate:failure"]
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        self.assertEqual(
            [i.code for i in ctx.exception.issues], ["linear.edge_missing"]
        )

    def test_duplicate_step_ids_are_rejected_with_the_kernel_code(self) -> None:
        dump = to_graph(
            _normalized([{"id": "one", "type": "llm", "name": "One"}])
        ).model_dump(by_alias=True)
        dump["steps"].append(copy.deepcopy(dump["steps"][0]))
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        self.assertIn(
            "step.id_duplicate", [i.code for i in ctx.exception.issues]
        )

    def test_policy_goto_target_missing_uses_the_kernel_code(self) -> None:
        dump = to_graph(
            _normalized([{"id": "one", "type": "llm", "name": "One"}])
        ).model_dump(by_alias=True)
        dump["steps"][0]["onFailure"] = {
            "type": "goto_step",
            "target_step_id": "ghost",
        }
        dump["edges"] = [
            {
                "id": "one:failure",
                "fromStepId": "one",
                "toStepId": "ghost",
                "fromField": None,
                "toField": None,
                "kind": "goto",
                "extensions": {"codeBridgeLinear": {"on": "failure"}},
            }
        ]
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        self.assertIn(
            "policy.goto_target_missing",
            [i.code for i in ctx.exception.issues],
        )

    def test_invalid_policy_uses_the_kernel_code(self) -> None:
        dump = to_graph(
            _normalized([{"id": "one", "type": "llm", "name": "One"}])
        ).model_dump(by_alias=True)
        dump["steps"][0]["onFailure"] = {"type": "explode"}
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(Flow.model_validate(dump))
        self.assertEqual(
            [i.code for i in ctx.exception.issues], ["policy.invalid"]
        )


if __name__ == "__main__":
    unittest.main()
