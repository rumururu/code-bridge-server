"""The AI can now improve an agent that already exists — without losing it.

Until this existed the Agent Builder could only create. ``BuilderTurn`` had no
``agent_id``, ``BuilderSession`` remembered no source, and ``builder_commit``
called ``store.create_agent`` and nothing else, so "change the second step of
the agent I already have" produced a *second* agent and left the first one
running unchanged.

The dangerous half of closing that gap is not the conversation. It is that an
agent carries more than an ``AgentDraft`` models — ``policy_overrides_json``,
memories, script bindings on a shell step, the origin that says whether its
prompt executes at all. A converter that projected an agent into a draft and
wrote the draft back would delete, on every save, whatever the draft has no
room for. On an agent that runs unattended every six hours the owner finds out
at 3am and has nothing to point at.

So these tests are mostly about *nothing happening*:

- **Round trip.** Convert an agent to a draft, commit it untouched, read it
  back, and require the record to be what it was — every column, the memories,
  and the policy overrides the draft cannot even express. Not on a fixture
  invented for the occasion: on the three agents that were actually in the
  user's database on 2026-08-23 (see :data:`REAL_AGENT_SHAPES`).
- **Only what was asked.** A revision that changes one step's description
  changes one step's description.
- **A moved workflow refuses.** A proposal is built on a version. Committing
  it onto a version somebody else has since replaced is refused with
  ``flow_revision_conflict`` and writes nothing — the same precondition
  ``PATCH /agents/{id}`` and the canvas already use.
- **A shape that cannot round-trip is not opened at all.** It is refused at
  the door, naming the field, rather than opened and quietly truncated on
  save. That is the rule ``agent/agent_origin.py`` established for a
  file-backed prompt, applied to every other stored value.

The last one is the design claim worth restating: the converter proves the
round trip *before* opening, by projecting its own output back through the very
function the commit path uses and requiring equality. A field nobody thought
about cannot pass that check, so the failure mode of forgetting one is a
refusal with a field name in it rather than silent deletion.
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, cli_agent_sources, schedule_store  # noqa: E402
from agent.agent_models import AgentDraft  # noqa: E402
from agent.agent_origin import AUTHORED_ORIGIN  # noqa: E402
from agent.agent_revision import (  # noqa: E402
    REASON_FIELD_NOT_REPRESENTABLE,
    REASON_PROMPT_NOT_EDITABLE,
    REASON_WORKFLOW_NOT_NORMALIZABLE,
    AgentNotRevisableError,
    agent_patch_from_draft,
    draft_from_agent,
)
from agent.flow_revision import compute_flow_revision  # noqa: E402
from code_bridge_core.workflow_v2 import normalize_workflow  # noqa: E402
from core import database  # noqa: E402
from routes import agents as agents_routes  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


#: The three agents that were in the user's database on 2026-08-23, copied
#: verbatim (read-only) from ``~/.code-bridge/core/code_bridge.db``.
#:
#: They are here rather than as invented fixtures because the whole question is
#: whether the converter survives contact with shapes nobody designed it
#: against, and these are the shapes that exist: a ``shell`` step carrying
#: ``script_id`` and ``script_args`` (fields ``AgentDraft`` does not declare
#: and only keeps because ``WorkflowStep`` allows extras), an ``on_failure``
#: that is a structured ``goto_step`` rather than a string, an ``on_success``
#: of ``end``, a ``tool_hint: null`` and an ``actions: []`` stamped onto a
#: shell step by a normaliser that used to compute them for every type, and —
#: on the third — the reference stub and ``instruction`` of an agent that runs
#: from a file on disk.
#:
#: The ids are the real ones so the provenance is checkable. Nothing here
#: writes to those agents: each test seeds a throwaway of its own into a
#: temporary database and lets the store mint a fresh id.
REAL_AGENT_SHAPES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "agent_9f6af9fdeed24da8804ca8eb5d3368cc",
        "name": "스몰뎁 사이클 · N960N",
        "description": "N960N(24c2b6bcef0d7ece) 테스트 참여/정리 사이클",
        "system_prompt": (
            "You diagnose Android device automation failures from script logs."
        ),
        "provider_id": "anthropic",
        "model": None,
        "tools_json": [],
        "flow_json": [
            {
                "id": "cycle",
                "type": "shell",
                "name": "N960N 사이클",
                "script_id": "script_6655441dfd704e01968c992003925fdf",
                "script_args": [
                    "24c2b6bcef0d7ece",
                    "com.mkideabox.codeBridge|com.mkideabox.lottosignal",
                ],
                "description": "N960N에서 스몰뎁 교환 사이클 1회",
                "success_criteria": "사이클 정상 종료",
                "on_failure": {"type": "goto_step", "target_step_id": "diagnose"},
                "tool_hint": None,
                "actions": [],
                "on_success": {"type": "end"},
            },
            {
                "id": "diagnose",
                "type": "llm",
                "name": "실패 진단",
                "description": (
                    "직전 사이클 스텝의 exit code와 [smalldev-cycle] 로그를 읽고 "
                    "무엇이 막혔는지 한 줄로 보고한다. UI가 바뀐 흔적이면 어느 "
                    "화면인지 짚는다. 도구는 쓰지 않는다."
                ),
                "tool_hint": None,
                "success_criteria": "",
                "on_failure": {"type": "ask_user", "resume": "same_step"},
                "actions": [],
                "on_success": {"type": "continue"},
            },
        ],
        "policy_overrides_json": {},
    },
    {
        "source_id": "agent_c8a86444104f45a0b225aa5de3fae152",
        "name": "스몰뎁 사이클 · M205N",
        "description": "M205N(R59N3035LQL) 테스트 참여/정리 사이클",
        "system_prompt": (
            "You diagnose Android device automation failures from script logs."
        ),
        "provider_id": "anthropic",
        "model": None,
        "tools_json": [],
        "flow_json": [
            {
                "id": "cycle",
                "type": "shell",
                "name": "M205N 사이클",
                "script_id": "script_6655441dfd704e01968c992003925fdf",
                "script_args": [
                    "R59N3035LQL",
                    "com.mkideabox.nametree|com.mkideabox.local_minton_league",
                ],
                "description": "M205N에서 스몰뎁 교환 사이클 1회",
                "success_criteria": "사이클 정상 종료",
                "on_failure": {"type": "goto_step", "target_step_id": "diagnose"},
                "tool_hint": None,
                "actions": [],
                "on_success": {"type": "end"},
            },
            {
                "id": "diagnose",
                "type": "llm",
                "name": "실패 진단",
                "description": (
                    "직전 사이클 스텝의 exit code와 [smalldev-cycle] 로그를 읽고 "
                    "무엇이 막혔는지 한 줄로 보고한다. UI가 바뀐 흔적이면 어느 "
                    "화면인지 짚는다. 도구는 쓰지 않는다."
                ),
                "tool_hint": None,
                "success_criteria": "",
                "on_failure": {"type": "ask_user", "resume": "same_step"},
                "actions": [],
                "on_success": {"type": "continue"},
            },
        ],
        "policy_overrides_json": {},
    },
    {
        "source_id": "agent_104187294bd745b19a24f643d322fb58",
        "name": "agent-sdk-verifier-py",
        "description": (
            "Use this agent to verify that a Python Agent SDK application is "
            "properly configured, follows SDK best practices and documentation "
            "recommendations, and is ready for deployment or testing. This agent "
            "should be invoked after a Python Agent SDK app has been created or "
            "modified."
        ),
        "system_prompt": (
            "This agent runs the Claude Code agent 'agent-sdk-verifier-py', "
            "defined at /example/agents/agent-sdk-verifier-py.md.\n\nThat file is "
            "read at the start of every run and its prompt, declared tools, and "
            "model are what actually execute. Editing this text changes nothing "
            "— edit the file."
        ),
        "provider_id": "anthropic",
        "model": "sonnet",
        "tools_json": [],
        "flow_json": [
            {
                "id": "cli_agent_instruction",
                "type": "llm",
                "name": "agent-sdk-verifier-py",
                "instruction": (
                    "This agent runs the Claude Code agent "
                    "'agent-sdk-verifier-py', defined at "
                    "/example/agents/agent-sdk-verifier-py.md."
                ),
                "description": (
                    "Use this agent to verify that a Python Agent SDK "
                    "application is properly configured."
                ),
                "tool_hint": None,
                "success_criteria": "",
                "on_failure": {"type": "ask_user", "resume": "same_step"},
                "on_success": {"type": "continue"},
                "actions": [],
            }
        ],
        "policy_overrides_json": {},
    },
)


#: A policy the draft has no field for. Written onto every seeded agent so the
#: round-trip assertion has something to lose: this is the concrete answer to
#: "what happens to a stored value ``AgentDraft`` cannot express".
POLICY_OVERRIDES = {
    "requires_user_confirmation_for_payment_or_purchase": True,
    "do_not_bypass_captcha_or_2fa": True,
}

#: Two memories per seeded agent, for the same reason: memories are rows in
#: another table, and the draft flattens them to strings.
MEMORIES = ("이 기기는 밤에 꺼져 있을 수 있다.", "실패하면 로그부터 읽어라.")


def _draft_reply(draft: dict[str, Any], *, message: str = "확인했습니다.") -> str:
    return (
        f"{message}\n```draft\n"
        + json.dumps(draft, ensure_ascii=False)
        + "\n```\n"
    )


class BuilderRevisionTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "builder_revision.db"
        agent_store._agent_store = None
        schedule_store._store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

        app = FastAPI()
        app.include_router(agents_routes.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # --- helpers ---------------------------------------------------------

    def _seed(
        self,
        shape: dict[str, Any],
        *,
        policy: dict[str, Any] | None = None,
        memories: tuple[str, ...] = MEMORIES,
    ) -> dict[str, Any]:
        """A throwaway agent carrying one of the real shapes, byte for byte."""
        agent = self.store.create_agent(
            name=shape["name"],
            description=shape["description"],
            system_prompt=shape["system_prompt"],
            provider_id=shape["provider_id"],
            model=shape["model"],
            tools_json=shape["tools_json"],
            flow_json=shape["flow_json"],
            policy_overrides_json=(
                POLICY_OVERRIDES if policy is None else policy
            ),
        )
        for content in memories:
            self.store.add_memory(agent_id=agent["id"], content=content)
        return self.store.get_agent(agent["id"])

    def _open(
        self,
        agent_id: str,
        *,
        message: str = "이 에이전트 고치고 싶어.",
        reply: Any = None,
    ):
        """First turn of a revision conversation, with a stubbed model.

        ``reply`` receives the live ``BuilderSession`` and returns the raw text
        the Configurator would have produced. The default echoes the session's
        own draft back unchanged — a model that agrees with everything — which
        is exactly the input the round-trip assertion needs.
        """

        async def echo(session, *, timeout: float = 120.0) -> str:
            if reply is not None:
                return reply(session)
            return _draft_reply(session.current_draft.model_dump())

        with mock.patch("routes.agents.run_configurator_turn", echo):
            return self.client.post(
                "/api/agent/builder/converse",
                json={"user_message": message, "agent_id": agent_id},
            )

    def _commit(self, session_id: str, draft: dict[str, Any], **extra: Any):
        body: dict[str, Any] = {"session_id": session_id, "draft": draft}
        body.update(extra)
        return self.client.post("/api/agent/builder/commit", json=body)

    def _memory_contents(self, agent_id: str) -> list[str]:
        return sorted(
            str(memory["content"])
            for memory in (self.store.list_memories(agent_id) or [])
        )


class ConverterRoundTripTest(unittest.TestCase):
    """The converter's own claim, with no database and no HTTP anywhere near it.

    ``draft_from_agent`` refuses unless projecting its output back through
    ``agent_patch_from_draft`` reproduces the agent. This asserts the same
    thing from outside, so the proof is not only the implementation's own.
    """

    def test_every_real_shape_projects_back_to_itself(self) -> None:
        for shape in REAL_AGENT_SHAPES:
            with self.subTest(agent=shape["source_id"]):
                agent = {
                    "id": shape["source_id"],
                    **{
                        key: value
                        for key, value in shape.items()
                        if key != "source_id"
                    },
                }
                draft = draft_from_agent(agent, origin=AUTHORED_ORIGIN)
                written = agent_patch_from_draft(draft)
                written["flow_json"] = normalize_workflow(written["flow_json"])

                self.assertEqual(written["name"], agent["name"])
                self.assertEqual(written["description"], agent["description"])
                self.assertEqual(written["system_prompt"], agent["system_prompt"])
                self.assertEqual(written["provider_id"], agent["provider_id"])
                self.assertEqual(written["model"], agent["model"])
                self.assertEqual(written["tools_json"], agent["tools_json"])
                self.assertEqual(
                    written["flow_json"], normalize_workflow(agent["flow_json"])
                )
                # And the workflow was already in its canonical form, so this
                # particular round trip does not even re-spell it: the stored
                # bytes come back as the stored bytes.
                self.assertEqual(written["flow_json"], agent["flow_json"])

    def test_the_patch_never_names_the_policy_column(self) -> None:
        """The one stored field a draft cannot carry is preserved by omission.

        ``AgentStore.update_agent`` assigns only the columns present in the
        patch. If ``policy_overrides_json`` ever appeared here it would be
        written as whatever the draft happened to imply, which for every draft
        is nothing.
        """
        draft = draft_from_agent(
            {
                "id": "agent_x",
                **{
                    key: value
                    for key, value in REAL_AGENT_SHAPES[0].items()
                    if key != "source_id"
                },
            },
            origin=AUTHORED_ORIGIN,
        )
        self.assertNotIn("policy_overrides_json", agent_patch_from_draft(draft))


class RoundTripThroughTheWholeLoopTest(BuilderRevisionTestBase):
    """Open, agree, commit, re-read — and find the agent unchanged.

    The strongest version of the acceptance criterion: not "the converter is
    lossless" but "the product is". Everything a commit touches is asserted,
    including the two things the draft cannot express (policy overrides,
    memories) and the one thing an update must never do (create a second
    agent).
    """

    def test_an_untouched_revision_changes_nothing(self) -> None:
        for seeded, shape in enumerate(REAL_AGENT_SHAPES, start=1):
            with self.subTest(agent=shape["source_id"]):
                before = self._seed(shape)
                before_memories = self._memory_contents(before["id"])

                opened = self._open(before["id"])
                self.assertEqual(opened.status_code, 200, opened.text)
                turn = opened.json()
                self.assertEqual(turn["source_agent_id"], before["id"])
                self.assertEqual(
                    turn["source_flow_revision"],
                    compute_flow_revision(before["flow_json"]),
                )

                committed = self._commit(turn["session_id"], turn["updated_draft"])
                self.assertEqual(committed.status_code, 200, committed.text)
                result = committed.json()["commit_result"]
                self.assertFalse(result["agent"]["created"])
                self.assertTrue(result["agent"]["updated"])
                self.assertEqual(result["agent"]["id"], before["id"])

                after = self.store.get_agent(before["id"])
                for column in (
                    "name",
                    "description",
                    "system_prompt",
                    "provider_id",
                    "model",
                    "tools_json",
                    "flow_json",
                    "policy_overrides_json",
                ):
                    self.assertEqual(
                        after[column], before[column], f"{column} changed"
                    )
                self.assertEqual(
                    self._memory_contents(before["id"]), before_memories
                )
                # No second agent. This is the bug the whole feature exists to
                # end, so it is asserted on the count, not on the response.
                # (Each subTest seeds one more throwaway into the same
                # database, hence the running total rather than 1.)
                self.assertEqual(self.store.count_agents(), seeded)

    def test_the_workflow_revision_is_unchanged_by_an_untouched_commit(self) -> None:
        """A commit that changes nothing must not look like a change.

        The revision is a content hash, so this holds for free — but it is the
        property another writer depends on: their held revision is still valid
        after somebody else opened the agent in the AI editor and agreed with
        it.
        """
        before = self._seed(REAL_AGENT_SHAPES[0])
        opened = self._open(before["id"]).json()
        self._commit(opened["session_id"], opened["updated_draft"])
        self.assertEqual(
            compute_flow_revision(self.store.get_agent(before["id"])["flow_json"]),
            compute_flow_revision(before["flow_json"]),
        )

    def test_what_is_stable_is_the_content_and_the_revision_not_the_bytes(
        self,
    ) -> None:
        """The stored JSON may come back with its keys in a different order.

        ``WorkflowStep.model_dump()`` emits declared fields in declaration
        order and appends the extras (``script_id``, ``script_args``,
        ``instruction``), so a workflow that reached the column by another
        route can be re-serialised with its keys rearranged. Measured on the
        throwaway copies of all three real agents: parsed content equal,
        ``flow_revision`` equal, raw column text not equal.

        That is stated here rather than fixed because nothing reads key order:
        ``canonical_flow_text`` sorts keys before hashing precisely so a
        difference nobody made cannot refuse a reader, and every consumer
        parses the JSON. The commit path has always serialised this way — it
        is what ``store.create_agent`` receives on the create path too. What
        must not drift is this test's *other* two assertions, and they are the
        ones a client depends on.
        """
        before = self._seed(REAL_AGENT_SHAPES[0])
        opened = self._open(before["id"]).json()
        self._commit(opened["session_id"], opened["updated_draft"])
        after = self.store.get_agent(before["id"])

        self.assertEqual(after["flow_json"], before["flow_json"])
        self.assertEqual(
            compute_flow_revision(after["flow_json"]),
            compute_flow_revision(before["flow_json"]),
        )
        self.assertEqual(
            [sorted(step) for step in after["flow_json"]],
            [sorted(step) for step in before["flow_json"]],
        )

    def test_the_session_is_gone_after_a_successful_commit(self) -> None:
        before = self._seed(REAL_AGENT_SHAPES[0])
        opened = self._open(before["id"]).json()
        self._commit(opened["session_id"], opened["updated_draft"])
        again = self._commit(opened["session_id"], opened["updated_draft"])
        self.assertEqual(again.status_code, 404)


class RevisionAppliesOnlyWhatWasAskedTest(BuilderRevisionTestBase):
    """One change in, one change out — and every other field byte-identical."""

    def test_changing_one_step_changes_only_that_step(self) -> None:
        before = self._seed(REAL_AGENT_SHAPES[0])

        def edited(session):
            draft = session.current_draft.model_dump()
            draft["flow"][1]["description"] = "실패 원인을 두 줄로 보고한다."
            return _draft_reply(draft, message="진단 단계 설명만 바꿨습니다.")

        opened = self._open(
            before["id"], message="진단 단계 설명만 바꿔줘.", reply=edited
        ).json()
        committed = self._commit(opened["session_id"], opened["updated_draft"])
        self.assertEqual(committed.status_code, 200, committed.text)

        after = self.store.get_agent(before["id"])
        self.assertEqual(
            after["flow_json"][1]["description"], "실패 원인을 두 줄로 보고한다."
        )
        # Everything else, including the shell step's script binding.
        self.assertEqual(after["flow_json"][0], before["flow_json"][0])
        self.assertEqual(
            {
                key: value
                for key, value in after["flow_json"][1].items()
                if key != "description"
            },
            {
                key: value
                for key, value in before["flow_json"][1].items()
                if key != "description"
            },
        )
        for column in ("name", "system_prompt", "tools_json", "policy_overrides_json"):
            self.assertEqual(after[column], before[column])
        self.assertEqual(self.store.count_agents(), 1)

    def test_the_word_that_means_step_does_not_revert_the_edit(self) -> None:
        """"단계" must not silently undo a revision.

        ``_preserve_flow_for_additive_request`` keeps the *previous* version of
        any step whose id the model reused, whenever the message looks
        additive — and every marker it looks for (단계, 추가, 기능, "step",
        "add") is ordinary phrasing for asking to change a step. On a creation
        draft the previous version is the model's own earlier guess and
        keeping it is a kindness; on a revision it is the user's saved agent,
        so keeping it means the edit they asked for is thrown away while the
        reply says it was made. This is that case, in the words a user would
        actually type.
        """
        before = self._seed(REAL_AGENT_SHAPES[0])

        def edited(session):
            draft = session.current_draft.model_dump()
            draft["flow"][1]["name"] = "원인 보고"
            return _draft_reply(draft)

        opened = self._open(
            before["id"],
            message="두 번째 단계 이름을 '원인 보고'로 바꿔줘.",
            reply=edited,
        ).json()
        self.assertEqual(opened["updated_draft"]["flow"][1]["name"], "원인 보고")

        committed = self._commit(opened["session_id"], opened["updated_draft"])
        self.assertEqual(committed.status_code, 200, committed.text)
        self.assertEqual(
            self.store.get_agent(before["id"])["flow_json"][1]["name"], "원인 보고"
        )

    def test_a_new_memory_seed_is_added_and_the_old_ones_are_not_duplicated(
        self,
    ) -> None:
        """The draft carries the agent's memories out; commit must not re-add them.

        Writing the draft's seeds the way the create path does would duplicate
        every memory on every commit. Absence still deletes nothing: the draft
        has no way to say "forget this", so inferring deletion from a seed the
        model happened not to repeat would let a forgetful reply erase what the
        user asked the agent to remember.
        """
        before = self._seed(REAL_AGENT_SHAPES[0])

        def edited(session):
            draft = session.current_draft.model_dump()
            self.assertEqual(sorted(draft["memory_seeds"]), sorted(MEMORIES))
            draft["memory_seeds"] = ["새로 기억할 것"]
            return _draft_reply(draft)

        opened = self._open(before["id"], reply=edited).json()
        committed = self._commit(opened["session_id"], opened["updated_draft"])
        self.assertEqual(committed.status_code, 200, committed.text)
        self.assertEqual(
            self._memory_contents(before["id"]),
            sorted([*MEMORIES, "새로 기억할 것"]),
        )


class StaleRevisionIsRefusedTest(BuilderRevisionTestBase):
    """A proposal is built on a version. It is not applied to a different one."""

    def _move_the_workflow(self, agent_id: str) -> list[dict[str, Any]]:
        """Another writer replaces the workflow while the conversation runs."""
        replacement = [
            {
                "id": "diagnose",
                "type": "llm",
                "name": "다른 사람이 바꾼 단계",
                "description": "캔버스에서 저장한 워크플로.",
            }
        ]
        response = self.client.patch(
            f"/api/agent/agents/{agent_id}",
            json={"flow_json": replacement},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return self.store.get_agent(agent_id)["flow_json"]

    def test_a_commit_onto_a_moved_workflow_is_refused_and_writes_nothing(
        self,
    ) -> None:
        before = self._seed(REAL_AGENT_SHAPES[0])
        opened = self._open(before["id"]).json()
        theirs = self._move_the_workflow(before["id"])

        committed = self._commit(opened["session_id"], opened["updated_draft"])
        self.assertEqual(committed.status_code, 409, committed.text)
        payload = committed.json()
        self.assertEqual(payload["error"], "flow_revision_conflict")
        self.assertEqual(
            payload["expected_flow_revision"],
            compute_flow_revision(before["flow_json"]),
        )
        self.assertEqual(
            payload["current_flow_revision"], compute_flow_revision(theirs)
        )
        # The other writer's version is still exactly theirs.
        self.assertEqual(self.store.get_agent(before["id"])["flow_json"], theirs)
        self.assertEqual(self.store.count_agents(), 1)

    def test_the_refused_commit_leaves_the_conversation_alive(self) -> None:
        """A refusal the author can act on keeps the work they can act with.

        Deleting the session on conflict would throw away the conversation
        that produced the proposal, which is the one thing they need in order
        to redo it against the new version.
        """
        before = self._seed(REAL_AGENT_SHAPES[0])
        opened = self._open(before["id"]).json()
        self._move_the_workflow(before["id"])
        self._commit(opened["session_id"], opened["updated_draft"])

        again = self._commit(opened["session_id"], opened["updated_draft"])
        self.assertEqual(again.status_code, 409, again.text)

    def test_a_stale_precondition_from_the_client_is_refused_too(self) -> None:
        """The client may hold a newer baseline than the conversation does.

        Both are checked, so a client that re-read the agent cannot use its own
        fresher revision to slip past the version the proposal was built on,
        and a client holding an older one is told before anything is written.
        """
        before = self._seed(REAL_AGENT_SHAPES[0])
        opened = self._open(before["id"]).json()

        committed = self._commit(
            opened["session_id"],
            opened["updated_draft"],
            if_flow_revision="0" * 16,
        )
        self.assertEqual(committed.status_code, 409, committed.text)
        self.assertEqual(committed.json()["error"], "flow_revision_conflict")
        self.assertEqual(
            self.store.get_agent(before["id"])["flow_json"], before["flow_json"]
        )

    def test_a_matching_precondition_from_the_client_goes_through(self) -> None:
        before = self._seed(REAL_AGENT_SHAPES[0])
        opened = self._open(before["id"]).json()
        committed = self._commit(
            opened["session_id"],
            opened["updated_draft"],
            if_flow_revision=compute_flow_revision(before["flow_json"]),
        )
        self.assertEqual(committed.status_code, 200, committed.text)


class NotOpenedRatherThanTruncatedTest(BuilderRevisionTestBase):
    """A shape that cannot come back is refused at the door, naming the field."""

    def _refusal(self, **overrides: Any) -> dict[str, Any]:
        shape = {
            **{
                key: value
                for key, value in REAL_AGENT_SHAPES[0].items()
                if key != "source_id"
            },
            **overrides,
        }
        agent = self.store.create_agent(
            name=shape["name"],
            description=shape["description"],
            system_prompt=shape["system_prompt"],
            provider_id=shape["provider_id"],
            model=shape["model"],
            tools_json=shape["tools_json"],
            flow_json=shape["flow_json"],
            policy_overrides_json={},
        )
        response = self._open(agent["id"])
        self.assertEqual(response.status_code, 409, response.text)
        return response.json()

    def test_a_provider_the_draft_cannot_spell_is_refused_by_name(self) -> None:
        """42 agents in this project's development database store one of these.

        ``AgentDraft.provider_id`` is a three-value literal and the column is
        free text. Opening one anyway would rewrite ``'poc'`` to whatever the
        model picked, on save, without anybody asking.
        """
        payload = self._refusal(provider_id="poc")
        self.assertEqual(payload["error"], REASON_FIELD_NOT_REPRESENTABLE)
        self.assertEqual(payload["field"], "provider_id")
        self.assertIn("provider_id", payload["detail"])

    def test_a_tool_entry_with_an_undeclared_key_is_refused_by_path(self) -> None:
        """``AgentToolDraft`` ignores keys it does not declare, so it drops them.

        Four agents in this project's databases carry ``name`` or
        ``display_name`` on a tools entry. The refusal names the exact path so
        the user is not left to guess which of their tools it was.
        """
        payload = self._refusal(
            tools_json=[{"mcp_id": "playwright", "display_name": "Playwright"}]
        )
        self.assertEqual(payload["error"], REASON_FIELD_NOT_REPRESENTABLE)
        self.assertEqual(payload["field"], "tools_json")
        self.assertIn("tools_json[0].display_name", payload["detail"])

    def test_declared_defaults_appearing_are_not_treated_as_loss(self) -> None:
        """An older row that omits an optional key is still openable.

        The rule is "what was stored must survive", not "the bytes must match":
        a ``tool_names: []`` the model declares and the row omitted is the same
        configuration spelled canonically, and every ordinary save writes it.
        Refusing over those would close AI editing on agents that lose nothing.
        """
        agent = self.store.create_agent(
            name="tools without defaults",
            system_prompt="p",
            provider_id="openai",
            tools_json=[{"mcp_id": "playwright"}],
            flow_json=REAL_AGENT_SHAPES[0]["flow_json"],
        )
        draft = draft_from_agent(
            self.store.get_agent(agent["id"]), origin=AUTHORED_ORIGIN
        )
        self.assertEqual(draft.tools[0].mcp_id, "playwright")

    def test_a_workflow_the_server_cannot_save_is_refused(self) -> None:
        """22 agents in this project's databases are in this state.

        They cannot be saved through ``PATCH /agents/{id}`` either, so opening
        one for AI editing would be opening an editor whose Save button can
        never work — and whose first successful save would be a rewritten
        workflow replacing the one that will not normalise.
        """
        payload = self._refusal(
            flow_json=[
                {
                    "id": "open",
                    "type": "browser_action",
                    "name": "열기",
                    "instruction": "브라우저를 연다.",
                }
            ]
        )
        self.assertEqual(payload["error"], REASON_WORKFLOW_NOT_NORMALIZABLE)
        self.assertEqual(payload["field"], "flow_json")
        self.assertIn("browser_action", payload["detail"])

    def test_an_unknown_agent_is_a_404_before_any_model_call(self) -> None:
        response = self._open("agent_does_not_exist")
        self.assertEqual(response.status_code, 404)

    def test_a_conversation_cannot_be_pointed_at_a_second_agent(self) -> None:
        """Retargeting mid-conversation would write one agent onto another.

        The briefing in the system prompt and the draft in hand both belong to
        the agent the session opened; only the id would change.
        """
        first = self._seed(REAL_AGENT_SHAPES[0])
        second = self._seed(REAL_AGENT_SHAPES[1])
        opened = self._open(first["id"]).json()

        async def echo(session, *, timeout: float = 120.0) -> str:
            return _draft_reply(session.current_draft.model_dump())

        with mock.patch("routes.agents.run_configurator_turn", echo):
            response = self.client.post(
                "/api/agent/builder/converse",
                json={
                    "session_id": opened["session_id"],
                    "user_message": "이걸로 바꿔줘",
                    "agent_id": second["id"],
                },
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "builder_session_agent_mismatch")

    def test_a_creation_conversation_cannot_become_a_revision(self) -> None:
        async def echo(session, *, timeout: float = 120.0) -> str:
            return _draft_reply(session.current_draft.model_dump())

        target = self._seed(REAL_AGENT_SHAPES[0])
        with mock.patch("routes.agents.run_configurator_turn", echo):
            first = self.client.post(
                "/api/agent/builder/converse",
                json={"user_message": "새 에이전트 만들어줘"},
            ).json()
            response = self.client.post(
                "/api/agent/builder/converse",
                json={
                    "session_id": first["session_id"],
                    "user_message": "아니 이걸 고쳐줘",
                    "agent_id": target["id"],
                },
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "builder_session_agent_mismatch")


CLI_AGENT_FILE = textwrap.dedent(
    """\
    ---
    name: disk-watch
    description: Reports on disk pressure.
    tools: Read, Glob
    ---

    You are the disk watcher. Report free space and stop.
    """
)


class FileBackedAgentIsNotOpenedTest(BuilderRevisionTestBase):
    """The prompt is on disk, so an AI that writes prompts must not be pointed at it.

    ``PATCH /agents/{id}`` already refuses to *write* the stored prompt of a
    file-backed agent (``agent/agent_origin.py``) while deliberately allowing
    the workflow around it. This goes one step earlier and refuses to open the
    conversation at all, because the Configurator's whole output is a prompt
    and a set of step instructions: it would spend the conversation writing
    text that executes nowhere, and the user would approve it believing it did
    something.

    The third of :data:`REAL_AGENT_SHAPES` is exactly this kind of agent in the
    user's own database. Its *shape* round-trips fine (see
    :class:`ConverterRoundTripTest`) — origin is derived from the import
    mapping, not from the shape — which is why this refusal has to be a policy
    decision made explicitly rather than something the loss check would catch.
    """

    def setUp(self) -> None:
        super().setUp()
        self._files = tempfile.TemporaryDirectory()
        self.root = Path(self._files.name)
        self.source = (self.root / "watch.md").resolve()
        self.source.write_text(CLI_AGENT_FILE, encoding="utf-8")
        self._locations = mock.patch.object(
            cli_agent_sources,
            "_all_source_locations",
            return_value=[("user", self.root)],
        )
        self._locations.start()
        self.file_backed = cli_agent_sources.import_cli_agent(str(self.source)).agent

    def tearDown(self) -> None:
        self._locations.stop()
        self._files.cleanup()
        super().tearDown()

    def test_opening_it_is_refused_and_names_the_file(self) -> None:
        response = self._open(self.file_backed["id"])
        self.assertEqual(response.status_code, 409, response.text)
        payload = response.json()
        self.assertEqual(payload["error"], REASON_PROMPT_NOT_EDITABLE)
        self.assertEqual(payload["field"], "system_prompt")
        self.assertIn(str(self.source), payload["detail"])
        self.assertFalse(payload["origin"]["prompt_editable"])

    def test_the_converter_refuses_it_directly_too(self) -> None:
        """The guard is in the converter, not only in the route.

        Every future caller of ``draft_from_agent`` gets it, including one
        written by somebody who never read this route.
        """
        with self.assertRaises(AgentNotRevisableError) as caught:
            draft_from_agent(self.store.get_agent(self.file_backed["id"]))
        self.assertEqual(caught.exception.reason, REASON_PROMPT_NOT_EDITABLE)


class RevisionDoesNotGrowTasksTest(BuilderRevisionTestBase):
    """Editing an agent must not give it a second goal on a second clock."""

    def test_an_agent_that_already_has_a_task_does_not_get_another(self) -> None:
        before = self._seed(REAL_AGENT_SHAPES[0])
        task = self.store.create_task(
            title="사이클 실행",
            description=None,
            assigned_agent_id=before["id"],
            kind="general",
            source="manual",
            goal="6시간마다 사이클을 돈다",
        )
        schedule_store.get_schedule_store().create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 21600},
            enabled=True,
        )

        opened = self._open(
            before["id"], message="진단 단계 설명만 다듬어줘. 6시간마다 돌려줘."
        ).json()
        committed = self._commit(
            opened["session_id"],
            opened["updated_draft"],
            task_draft={"goal": "완전히 다른 목표", "schedule": "0 */2 * * *"},
        )
        self.assertEqual(committed.status_code, 200, committed.text)
        result = committed.json()["commit_result"]
        self.assertFalse(result["task"]["created"])
        self.assertEqual(result["task"]["reason"], "existing_task_kept")
        self.assertFalse(result["schedule"]["created"])
        # The one question the payload exists to answer stays true: the agent
        # still runs by itself, on the schedule this commit did not touch.
        self.assertTrue(result["runs_unattended"])
        self.assertNotIn("스스로 실행되지 않습니다", result["summary"])

        tasks = [
            row
            for row in self.store.list_tasks(limit=50)
            if row.get("assigned_agent_id") == before["id"]
        ]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["id"], task["id"])


class TheBriefingComesFromThePublishedVocabularyTest(BuilderRevisionTestBase):
    """The revision prompt adds context, never a second copy of the field list.

    A hand-written field list here would be the drift
    ``code_bridge_core/workflow_step_schema.py`` exists to end — and it would be the copy
    that goes stale, because nothing generates it. So the briefing is asserted
    to carry the agent's own draft and the revision rules, and the step
    vocabulary is asserted to still be the generated one.
    """

    def test_the_prompt_carries_the_agent_and_the_generated_schema(self) -> None:
        from code_bridge_core.configurator import (
            BUILDER_SESSIONS,
            _workflow_step_schema_block,
        )

        before = self._seed(REAL_AGENT_SHAPES[0])
        opened = self._open(before["id"]).json()
        prompt = BUILDER_SESSIONS[opened["session_id"]].system_prompt

        self.assertIn(_workflow_step_schema_block(), prompt)
        self.assertIn(before["id"], prompt)
        self.assertIn("script_6655441dfd704e01968c992003925fdf", prompt)
        self.assertIn(compute_flow_revision(before["flow_json"]), prompt)
        self.assertIn("NOT creating a new agent", prompt)

    def test_a_creation_session_gets_no_briefing(self) -> None:
        from code_bridge_core.configurator import BUILDER_SESSIONS

        async def echo(session, *, timeout: float = 120.0) -> str:
            return _draft_reply(session.current_draft.model_dump())

        with mock.patch("routes.agents.run_configurator_turn", echo):
            opened = self.client.post(
                "/api/agent/builder/converse",
                json={"user_message": "새 에이전트 만들어줘"},
            ).json()
        prompt = BUILDER_SESSIONS[opened["session_id"]].system_prompt
        self.assertNotIn("NOT creating a new agent", prompt)
        self.assertIsNone(opened.get("source_agent_id"))


class CreationIsUnchangedTest(BuilderRevisionTestBase):
    """A turn with no ``agent_id`` behaves exactly as it did before."""

    def test_a_commit_without_a_source_agent_still_creates(self) -> None:
        draft: dict[str, Any] = AgentDraft(
            name="새 에이전트",
            description="설명",
            system_prompt="You are useful.",
            provider_id="openai",
            flow=[{"id": "one", "name": "Step", "type": "llm"}],
        ).model_dump()

        async def echo(session, *, timeout: float = 120.0) -> str:
            return _draft_reply(draft)

        with mock.patch("routes.agents.run_configurator_turn", echo):
            opened = self.client.post(
                "/api/agent/builder/converse",
                json={"user_message": "만들어줘"},
            ).json()
        self.assertIsNone(opened.get("source_agent_id"))

        committed = self._commit(opened["session_id"], opened["updated_draft"])
        self.assertEqual(committed.status_code, 200, committed.text)
        result = committed.json()["commit_result"]
        self.assertTrue(result["agent"]["created"])
        self.assertFalse(result["agent"]["updated"])
        self.assertEqual(self.store.count_agents(), 1)


if __name__ == "__main__":
    unittest.main()
