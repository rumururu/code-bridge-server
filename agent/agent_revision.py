"""Opening an agent that already exists, so the Configurator can improve it.

Until this module existed the Agent Builder could only ever *create*.
``BuilderTurn`` carried no agent, ``BuilderSession`` remembered no source, and
``builder_commit`` called ``store.create_agent`` and nothing else — so the one
thing a user asks for most ("this agent is nearly right, change the second
step") had no path through the AI at all. The workflow could be edited on the
canvas or in the app's edit screen; the conversation that designed it could
not be resumed against what it produced.

What makes that hard is not the conversation. It is that **an agent carries
more than a draft models**. ``AgentDraft`` has no ``policy_overrides_json``, no
memories, no origin and no notion of which script a shell step is bound to
beyond an opaque extra field. A converter that projected an agent into a draft
and let the commit path write the draft back would therefore delete, on every
save, whatever the draft has no room for — on agents that run unattended every
six hours, whose owner would find out at 3am and have nothing to point at.

So this module's contract is not "convert". It is:

    **An agent is opened for AI editing only if converting it and writing it
    back is provably a no-op.**

:func:`draft_from_agent` builds the draft, immediately projects that draft back
through :func:`agent_patch_from_draft` — the *same* function the commit path
uses, not a parallel one written for the check — and compares the result to the
agent it started from, field by field. A single mismatch raises
:class:`AgentNotRevisableError` naming the field. The agent stays openable in
every other surface; what it does not get is an AI editor that would quietly
drop something.

That inverts the usual failure. The question is no longer "did we remember to
carry field X?" — a question whose wrong answer is silent — but "does the
round trip come out equal?", whose wrong answer is a refusal with a field name
in it. A field added to ``agents`` next year is covered on the day it is added:
it will not survive the projection, so the comparison fails and this refuses,
rather than the field being erased by an agent the AI edited.

## What is deliberately *not* in the patch

``policy_overrides_json`` is not a draft field and is **not written back**. It
is preserved by omission: :func:`agent_patch_from_draft` never names the
column, ``AgentStore.update_agent`` only assigns columns present in the patch,
so the stored overrides are exactly what they were. It is therefore also not
part of the comparison — there is nothing to compare, because nothing is
written. That is the one field where "cannot round-trip" is answered by not
making the trip at all, and it is safe precisely because the store's update is
a partial patch rather than a row replacement.

Memories are the other. They live in ``agent_memories``, one row each, and the
draft flattens them to a list of strings. The conversion carries them out (so
the model can see what the agent already remembers and does not propose
duplicates) and the commit path adds only seeds that are not already stored —
never deletes. See :func:`memory_seeds_to_add`.

## The baseline the flow is compared against

Not the bytes in the column: ``normalize_workflow`` of them. Every other write
path on this router normalises before storing (``_normalize_agent_workflow`` in
``routes/agents.py``), and the runtime re-normalises the stored value on every
run (``task_orchestrator._workflow_steps_for_task``), so the normalised form is
what the agent already *is*; the stored bytes are merely an older spelling of
it. Measured on this project's two databases (2026-08-23, 124 agent rows):
``normalize_workflow(stored)`` equals ``normalize_workflow(WorkflowStep
round-trip of stored)`` for all 102 rows that normalise at all, and
normalisation is idempotent on all 102. The 22 that do not normalise are
refused here — they cannot be saved through ``PATCH /agents/{id}`` either, so
an AI editor for them would be an editor whose Save button never worked.

Two consequences, both disclosed rather than hidden:

* A commit that changes nothing **can** move the agent's ``flow_revision``, if
  the stored spelling was an older one that normalisation fills out. The same
  thing happens when a human opens that agent on the canvas and presses Save.
  (Measured on the user's three real agents: it does not, because all three
  are already stored in canonical form.)
* The stored JSON's **key order** can change even when nothing else does:
  ``WorkflowStep.model_dump()`` emits declared fields first and appends the
  extras. Nothing reads key order — ``flow_revision`` sorts keys before
  hashing exactly so a difference nobody made cannot refuse a reader, and
  every consumer parses the JSON — so this is a re-spelling, not an edit. It
  is stated because "byte-identical" would be the wrong claim to make.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from pydantic import ValidationError

from .agent_models import AgentDraft, AgentToolDraft, WorkflowStep
from .agent_origin import AgentOrigin, resolve_agent_origin
from code_bridge_core.workflow_v2 import WorkflowNormalizationError, normalize_workflow

logger = logging.getLogger(__name__)

__all__ = [
    "REASON_FIELD_NOT_REPRESENTABLE",
    "REASON_PROMPT_NOT_EDITABLE",
    "REASON_WORKFLOW_NOT_NORMALIZABLE",
    "AgentNotRevisableError",
    "agent_patch_from_draft",
    "draft_from_agent",
    "memory_seeds_to_add",
]


#: The agent runs from a file on disk, so the prompt the AI would write is not
#: the prompt that executes. See :mod:`agent.agent_origin`.
REASON_PROMPT_NOT_EDITABLE = "agent_prompt_not_editable"

#: Some stored value does not survive the trip through ``AgentDraft``. The
#: error names the field.
REASON_FIELD_NOT_REPRESENTABLE = "agent_field_not_representable"

#: The stored workflow is not one ``normalize_workflow`` accepts, so no write
#: path on this server could save it back.
REASON_WORKFLOW_NOT_NORMALIZABLE = "agent_workflow_not_normalizable"


class AgentNotRevisableError(RuntimeError):
    """This agent must not be opened for AI editing, and here is which field.

    Three fields, and the middle one is the point:

    ``reason``
        A machine code from the constants above, so a client can branch.
    ``field``
        The stored field that does not survive. ``None`` only for refusals
        that are not about a field (the file-backed prompt). Without this the
        refusal would be "cannot edit this agent", which tells the user
        nothing they can act on and tells the next maintainer nothing about
        what to fix.
    ``detail``
        The sentence a client renders verbatim. Every Code Bridge client reads
        ``detail`` (``lib/services/builder_service.dart`` reads
        ``detail ?? error ?? message``), so a refusal without one is a refusal
        the user sees as a blank error.
    """

    def __init__(
        self,
        detail: str,
        *,
        reason: str,
        field: str | None = None,
        origin: AgentOrigin | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.reason = reason
        self.field = field
        self.origin = origin

    def to_view(self) -> dict[str, Any]:
        view: dict[str, Any] = {
            "error": self.reason,
            "reason": self.reason,
            "detail": self.detail,
            "message": self.detail,
        }
        if self.field is not None:
            view["field"] = self.field
        if self.origin is not None:
            view["origin"] = self.origin.to_view()
        return view


def agent_patch_from_draft(draft: AgentDraft) -> dict[str, Any]:
    """The ``AgentStore.update_agent`` patch a draft represents.

    Exactly the columns a draft can speak for, and no others. Notably absent:
    ``policy_overrides_json``, which the draft cannot carry and which therefore
    must not be written — omitting it from the patch is what preserves it, and
    is the reason this returns a patch rather than a whole row.

    ``flow_json`` is **not** normalised here. The commit route normalises it
    itself (``_normalize_agent_workflow``) so that a normalisation failure is
    answered by the route's own 400 with the normaliser's message, the same as
    every other workflow write. This function is the shape, not the gate.
    """
    return {
        "name": (draft.name or "").strip(),
        "description": draft.description,
        "system_prompt": draft.system_prompt,
        "provider_id": draft.provider_id,
        "model": draft.model,
        "tools_json": [tool.model_dump() for tool in draft.tools],
        "flow_json": [step.model_dump() for step in draft.flow],
    }


def _stored_projection(agent: dict[str, Any]) -> dict[str, Any]:
    """The same seven values, read off the stored agent.

    ``system_prompt`` is read as ``value or ""`` because the column is
    nullable and ``AgentDraft.system_prompt`` is a plain ``str``. An empty
    prompt and a null one are the same absence of instructions, and refusing
    to open every agent that happens to store ``NULL`` would be a refusal
    about a storage detail rather than about the user's configuration.
    """
    return {
        "name": (agent.get("name") or "").strip(),
        "description": agent.get("description"),
        "system_prompt": agent.get("system_prompt") or "",
        "provider_id": agent.get("provider_id"),
        "model": agent.get("model"),
        "tools_json": agent.get("tools_json") or [],
        "flow_json": normalize_workflow(agent.get("flow_json") or []),
    }


#: Sentinel for "the round trip did not produce this key at all", so a dropped
#: key and a key legitimately set to ``None`` are distinguishable in the
#: message the user reads.
_ABSENT = object()


def _describe(value: Any, *, limit: int = 160) -> str:
    if value is _ABSENT:
        return "(없음)"
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def draft_from_agent(
    agent: dict[str, Any],
    *,
    memories: Sequence[dict[str, Any]] | None = None,
    origin: AgentOrigin | None = None,
) -> AgentDraft:
    """Open a stored agent as an ``AgentDraft``, or refuse and say why.

    Raises :class:`AgentNotRevisableError` for anything that would come back
    different from how it went out. Nothing is written and nothing is
    partially converted: the caller either gets a draft it can hand to the
    Configurator, or a refusal it can show the user.

    ``origin`` is injected so the caller can resolve it once and reuse it in
    its response; when omitted it is resolved here. An unresolvable origin is
    :data:`agent.agent_origin.UNKNOWN_ORIGIN`, which is not prompt-editable
    and is therefore refused — "I could not check" must not render as "yes, go
    ahead", the same rule ``PATCH /agents/{id}`` already follows.
    """
    agent_id = str(agent.get("id") or "")
    resolved_origin = origin if origin is not None else resolve_agent_origin(agent_id)
    if not resolved_origin.prompt_editable:
        raise AgentNotRevisableError(
            _prompt_not_editable_detail(resolved_origin),
            reason=REASON_PROMPT_NOT_EDITABLE,
            field="system_prompt",
            origin=resolved_origin,
        )

    try:
        stored = _stored_projection(agent)
    except WorkflowNormalizationError as exc:
        raise AgentNotRevisableError(
            "이 에이전트의 워크플로는 지금 형식 그대로는 저장할 수 없어 AI 편집을 "
            f"시작하지 않았습니다: {exc}. 이 상태로 AI에게 맡기면 저장할 때 거절되거나 "
            "고쳐 쓴 워크플로가 원래 것을 덮어씁니다. 캔버스나 편집 화면에서 먼저 "
            "이 단계를 고친 뒤 다시 시도하세요.",
            reason=REASON_WORKFLOW_NOT_NORMALIZABLE,
            field="flow_json",
        ) from exc

    draft = _build_draft(stored, memories=memories)

    # The proof, not a promise: project the draft back through the very
    # function the commit path will use and require equality. A field this
    # module never heard of cannot pass this, which is the whole point.
    written = agent_patch_from_draft(draft)
    try:
        written["flow_json"] = normalize_workflow(written["flow_json"])
    except WorkflowNormalizationError as exc:  # pragma: no cover - defensive
        raise AgentNotRevisableError(
            "이 에이전트의 워크플로를 드래프트로 옮기면 서버가 다시 저장할 수 없는 "
            f"형태가 되어 AI 편집을 시작하지 않았습니다: {exc}",
            reason=REASON_WORKFLOW_NOT_NORMALIZABLE,
            field="flow_json",
        ) from exc

    for field in stored:
        loss = _first_loss(stored[field], written[field], path=field)
        if loss is None:
            continue
        where, was, now = loss
        raise AgentNotRevisableError(
            f"이 에이전트의 '{where}' 값은 AI 편집용 드래프트로 옮겼다가 그대로 "
            "되돌릴 수 없어서 편집을 시작하지 않았습니다. 지금 시작하면 저장하는 "
            f"순간 이 값이 바뀝니다 (저장된 값 {_describe(was)} → 되돌린 값 "
            f"{_describe(now)}). 이 값을 먼저 편집 화면에서 정리한 뒤 다시 "
            "시도하거나, 이 에이전트는 캔버스/편집 화면에서 직접 고치세요.",
            reason=REASON_FIELD_NOT_REPRESENTABLE,
            field=field,
        )
    return draft


#: What :func:`_first_loss` reports when a value did not survive: the dotted
#: path to it, what was stored, and what came back.
_Loss = tuple[str, Any, Any]


def _first_loss(stored: Any, written: Any, *, path: str) -> _Loss | None:
    """Where the round trip lost or changed something, or ``None``.

    **Additions are not losses.** A key the stored value does not have and the
    round trip does — a ``tool_names: []`` that ``AgentToolDraft`` declares and
    an older row omitted, an ``on_success`` the normaliser fills in — is the
    canonical spelling of what was already there, and every ordinary save
    through ``PATCH /agents/{id}`` writes it too. Refusing over those would
    close AI editing on agents that lose nothing: measured on this project's
    two databases (2026-08-23), 5 of the 9 agents whose tools do not compare
    equal differ *only* by declared defaults, while 4 genuinely drop a stored
    key (``name`` / ``display_name``).

    Everything else is a loss and is reported: a stored key missing from the
    round trip, a stored value that came back different, a list that changed
    length. The recursion is what makes the check reach the field the user
    actually configured rather than stopping at "tools_json differs".
    """
    if written is _ABSENT:
        return (path, stored, _ABSENT)
    if isinstance(stored, dict):
        if not isinstance(written, dict):
            return (path, stored, written)
        for key, value in stored.items():
            loss = _first_loss(
                value, written.get(key, _ABSENT), path=f"{path}.{key}"
            )
            if loss is not None:
                return loss
        return None
    if isinstance(stored, list):
        if not isinstance(written, list) or len(written) != len(stored):
            return (path, stored, written)
        for index, value in enumerate(stored):
            loss = _first_loss(value, written[index], path=f"{path}[{index}]")
            if loss is not None:
                return loss
        return None
    if stored != written:
        return (path, stored, written)
    return None


def _prompt_not_editable_detail(origin: AgentOrigin) -> str:
    where = origin.source_path
    location = (
        f"Claude Code 에이전트 파일 '{where}'"
        if where
        else "이 에이전트가 실행할 때 읽는 정의 파일"
    )
    return (
        f"이 에이전트의 프롬프트는 {location}에 있습니다. 매 실행마다 그 파일을 다시 "
        "읽기 때문에 여기에 저장된 텍스트는 실행되지 않습니다. AI 편집은 프롬프트와 "
        "단계 지시문을 다시 쓰는 일이라, 시작하면 실행되지 않을 글을 쓰게 되므로 "
        "열지 않았습니다. 그 파일을 편집하세요. 워크플로 단계 자체는 캔버스나 편집 "
        "화면에서 그대로 고칠 수 있습니다."
    )


def _build_draft(
    stored: dict[str, Any],
    *,
    memories: Sequence[dict[str, Any]] | None,
) -> AgentDraft:
    """Validate the stored values into a draft, refusing per field.

    Each ``ValidationError`` is turned into a refusal that names the field it
    came from rather than one 'agent could not be converted': the two real
    causes seen in this project's databases are a ``provider_id`` outside the
    three the draft allows (``'poc'``, ``'ui-test'`` — 42 rows in the
    development database) and a tools entry carrying a key ``AgentToolDraft``
    does not declare (``name``, ``display_name`` — 4 rows). A user whose agent
    is refused needs to know which of those it was.
    """
    tools = _tool_drafts(stored["tools_json"])
    flow = _workflow_steps(stored["flow_json"])
    try:
        return AgentDraft(
            name=stored["name"] or None,
            description=stored["description"],
            system_prompt=stored["system_prompt"],
            provider_id=stored["provider_id"],
            model=stored["model"],
            tools=tools,
            flow=flow,
            memory_seeds=_memory_seeds(memories),
            # A saved agent has no outstanding script requests: a shell step
            # either names a registered script or the agent could not have
            # been saved (`_reject_unapproved_shell_steps`). Carrying an empty
            # list is the truth, and it leaves the model free to ask for a new
            # script during the revision.
            script_requests=[],
        )
    except ValidationError as exc:
        raise _validation_refusal(exc, prefix="") from exc


def _tool_drafts(tools_json: Any) -> list[AgentToolDraft]:
    if not isinstance(tools_json, list):
        raise AgentNotRevisableError(
            "이 에이전트의 도구 목록이 AI 편집용 드래프트가 다룰 수 있는 형태가 "
            f"아니라 편집을 시작하지 않았습니다 (tools_json = {_describe(tools_json)}).",
            reason=REASON_FIELD_NOT_REPRESENTABLE,
            field="tools_json",
        )
    drafts: list[AgentToolDraft] = []
    for index, entry in enumerate(tools_json):
        try:
            drafts.append(AgentToolDraft.model_validate(entry))
        except ValidationError as exc:
            raise _validation_refusal(
                exc, prefix=f"tools_json[{index}]: ", field="tools_json"
            ) from exc
    return drafts


def _workflow_steps(flow_json: Any) -> list[WorkflowStep]:
    steps: list[WorkflowStep] = []
    for index, entry in enumerate(flow_json):
        try:
            steps.append(WorkflowStep.model_validate(entry))
        except ValidationError as exc:
            raise _validation_refusal(
                exc, prefix=f"flow_json[{index}]: ", field="flow_json"
            ) from exc
    return steps


def _validation_refusal(
    exc: ValidationError,
    *,
    prefix: str,
    field: str | None = None,
) -> AgentNotRevisableError:
    errors = exc.errors()
    named = field or (str(errors[0]["loc"][0]) if errors and errors[0]["loc"] else None)
    lines = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
        for error in errors
    )
    return AgentNotRevisableError(
        "이 에이전트의 저장된 값이 AI 편집용 드래프트 형식에 맞지 않아 편집을 "
        f"시작하지 않았습니다: {prefix}{lines}. 지금 시작하면 저장할 때 이 값이 "
        "사라지거나 바뀝니다. 편집 화면에서 이 값을 먼저 정리한 뒤 다시 시도하세요.",
        reason=REASON_FIELD_NOT_REPRESENTABLE,
        field=named,
    )


def _memory_seeds(memories: Sequence[dict[str, Any]] | None) -> list[str]:
    """What the agent already remembers, as the draft spells it.

    Carried out so the model can see it. Not carried back as a replacement —
    see :func:`memory_seeds_to_add`, which is additive, because the draft has
    no way to express "delete this memory" and inferring deletion from absence
    would let a model that simply did not repeat a seed erase it.
    """
    if not memories:
        return []
    seeds: list[str] = []
    for memory in memories:
        content = str(memory.get("content") or "").strip()
        if content:
            seeds.append(content)
    return seeds


def memory_seeds_to_add(
    draft: AgentDraft,
    *,
    existing: Iterable[dict[str, Any]] | None,
) -> list[str]:
    """The seeds a revision commit should write: the new ones, once each.

    A revision's draft normally *contains* the agent's current memories,
    because :func:`draft_from_agent` put them there. Writing the draft's seeds
    the way the create path does would therefore duplicate every memory the
    agent has, on every commit, forever. Comparing content is enough to stop
    that: a memory is its text.

    Nothing is deleted. A seed the model dropped leaves its stored memory
    alone, which is the conservative direction — a user who wants a memory
    gone has ``DELETE /agents/{id}/memories/{memory_id}``, and a model that
    forgot to repeat one must not be able to reach it.
    """
    already = {
        str(memory.get("content") or "").strip()
        for memory in (existing or [])
        if str(memory.get("content") or "").strip()
    }
    additions: list[str] = []
    for seed in draft.memory_seeds:
        text = seed.strip()
        if not text or text in already:
            continue
        already.add(text)
        additions.append(text)
    return additions
