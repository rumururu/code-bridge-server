"""Where an agent came from, and which of its fields an edit can still reach.

Two kinds of agent share one table. One was **authored here** — somebody typed
its prompt into Code Bridge, and editing that prompt changes what runs. The
other was **registered from a file**: a Claude Code agent definition sitting on
this machine, run by reference. For that one the server rereads the ``.md`` at
the start of every run (:mod:`agent.cli_agent_runtime`) and hands it to the
session, so the prompt and declared tools that execute are always the file's.
What sits in the record's ``system_prompt`` is the stub from
:func:`agent.cli_agent_sources.cli_agent_reference_prompt`, whose own text says
"Editing this text changes nothing — edit the file."

Until this module existed, that stub was the only thing distinguishing them,
and it was destroyed by the exact act it warned about: overwrite the prompt,
save, and the edits still did nothing *and* the explanation was gone, with
nothing left to say why. No field on ``GET /agents`` told the two apart, so no
client could either.

What this module adds is the field — :func:`resolve_agent_origin` — and the
refusal that makes it more than a hint — :func:`assert_patch_reaches_execution`.

If this regresses, here is what the user experiences:

- **Origin missing or wrong on the payload**: the phone and the dashboard show
  a file-backed agent exactly like an authored one, hand the user a prompt box,
  and they spend an afternoon rewriting instructions that will never execute.
  That is the original defect, restored.
- **The refusal moves to the UI only**: a guard the API does not have is not a
  guard. Phone, dashboard, and anything else all go through ``PATCH
  /agents/{id}``; one unpatched client is enough to wipe the stub.
- **A failed lookup reported as ``authored``**: worse than no field at all. The
  client would render an editable prompt on evidence the server does not have.
  Hence :data:`ORIGIN_UNKNOWN`, which is not editable — see
  :func:`resolve_agent_origin`.

Why the origin lives here and not on ``agents``
-----------------------------------------------
``agent_cli_agent_imports`` already records ``source_path -> agent_id``, and it
is the single place that fact is written. A column on ``agents`` would be a
second copy of it, free to disagree with the first, and disagreement here means
the record claims to be editable while the runtime reads a file. So origin is
*derived*, every time it is asked for, from the mapping that the runtime itself
follows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .cli_agent_sources import find_import_source_path_for_agent

logger = logging.getLogger(__name__)

#: Written in Code Bridge. The stored prompt is the program; editing it changes
#: the next run.
ORIGIN_AUTHORED = "authored"

#: Registered from a Claude Code agent definition file on this machine. The
#: file is the program; the stored prompt is a stub nothing executes.
ORIGIN_CLI_AGENT_FILE = "cli_agent_file"

#: The mapping could not be read. Not a kind of agent — a kind of ignorance,
#: and it is kept distinct from :data:`ORIGIN_AUTHORED` on purpose.
ORIGIN_UNKNOWN = "unknown"


@dataclass(frozen=True)
class AgentOrigin:
    """Where one agent came from, in the shape clients receive it.

    Three fields, and the third is the one that matters for surviving a fourth
    origin later:

    ``kind``
        Which of the constants above. A client that recognises it can say
        something specific ("runs from this file").
    ``prompt_editable``
        Whether writing the stored prompt reaches execution. A client that has
        never heard of ``kind`` still knows from this alone not to offer a
        prompt box — so adding a third origin is a new ``kind`` value plus
        whatever descriptor it needs, never a change to this contract, and
        never a shipped app that starts lying about an agent it cannot name.
    ``source_path``
        The file, for the kinds that have one; ``None`` otherwise. Always
        present as a key so the JSON shape does not change with the kind.
    """

    kind: str
    prompt_editable: bool
    source_path: str | None = None

    def to_view(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "prompt_editable": self.prompt_editable,
            "source_path": self.source_path,
        }


#: What an agent with no import mapping is: written here, editable here.
AUTHORED_ORIGIN = AgentOrigin(kind=ORIGIN_AUTHORED, prompt_editable=True)

#: What an agent whose mapping could not be read is. Deliberately *not*
#: editable: "I could not check" must never render as "yes, go ahead".
UNKNOWN_ORIGIN = AgentOrigin(kind=ORIGIN_UNKNOWN, prompt_editable=False)


class AgentPromptNotEditableError(RuntimeError):
    """A patch tried to write text this agent will never execute.

    Carries the origin so the route can name the file in the response. A
    refusal that only says "not allowed" leaves the user with an edit that
    failed *and* no idea where the real prompt lives, which is the same dead
    end as an edit that silently succeeded.
    """

    def __init__(self, message: str, origin: AgentOrigin) -> None:
        super().__init__(message)
        self.origin = origin


def resolve_agent_origin(agent_id: str | None) -> AgentOrigin:
    """Where this agent came from, derived from the import mapping.

    Returns :data:`AUTHORED_ORIGIN` only when the mapping was read and held
    nothing for this agent — that is a positive answer, not a default. When the
    read itself fails, the answer is :data:`UNKNOWN_ORIGIN`, because the one
    thing this must never do is guess "authored" and hand a client an editable
    prompt box for an agent that runs from a file.

    This deliberately does not reuse
    :func:`agent.cli_agent_runtime.find_cli_agent_source_path`, which swallows
    a failed read into ``None`` so that *planning* never breaks over a mapping
    hiccup. That trade is right there and wrong here: a plan that omits a
    source path is degraded, but an origin that reports the wrong kind is
    false.
    """
    if not agent_id:
        return UNKNOWN_ORIGIN
    try:
        source_path = find_import_source_path_for_agent(agent_id)
    except Exception:  # noqa: BLE001 - the answer is "unknown", never "authored"
        logger.warning(
            "agent origin: could not read the import mapping for %s; "
            "reporting origin as unknown rather than guessing",
            agent_id,
            exc_info=True,
        )
        return UNKNOWN_ORIGIN

    if not source_path:
        return AUTHORED_ORIGIN
    return AgentOrigin(
        kind=ORIGIN_CLI_AGENT_FILE,
        prompt_editable=False,
        source_path=source_path,
    )


def _instruction_text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _llm_instructions(flow: Any) -> dict[str, str]:
    """``step id -> instruction`` for every step whose instruction is prose.

    Only ``llm`` steps (and steps with no type, which default to ``llm`` in
    ``task_orchestrator._workflow_step_message``) are collected. A shell step
    runs a registered script and a notify step writes an inbox entry; neither
    relays this field, so neither is what the guard is about.
    """
    if not isinstance(flow, list):
        return {}
    instructions: dict[str, str] = {}
    for index, step in enumerate(flow, start=1):
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type") or step.get("step_type") or "llm").strip()
        if step_type and step_type != "llm":
            continue
        text = _instruction_text(step.get("instruction"))
        if not text:
            continue
        instructions[str(step.get("id") or f"step_{index}")] = text
    return instructions


def assert_patch_reaches_execution(
    *,
    agent: dict[str, Any],
    patch: dict[str, Any],
    origin: AgentOrigin,
) -> None:
    """Refuse the parts of a patch that would be written and never run.

    Does nothing for an agent whose prompt is editable, which is most of them.
    For the rest it refuses exactly two things, and allows everything else on
    purpose — the goal is to stop edits that silently do nothing, not to freeze
    the record.

    **Refused: ``system_prompt``.** On a file-backed agent
    ``task_orchestrator._compose_assigned_agent_prompt`` replaces the stored
    prompt with a freshly generated reference stub before composing anything,
    so the stored text reaches no run, ever.

    **Refused: an ``llm`` step's ``instruction``.** Same reason, one level
    down: ``task_orchestrator._workflow_step_message`` prints "instruction:
    defined by the Claude Code agent file …" for every step of a file-backed
    agent and never relays the stored text.

    **Allowed: name, description, provider/model, tools, policy overrides, and
    the rest of the workflow.** A step's ``title``, ``success_criteria``,
    ``observation``, ``tool_hint``, ``actions`` and ``on_failure`` *are* relayed
    by that same function, and shell and notify steps run in full — so the
    workflow *around* the definition is genuinely the user's to build, and
    refusing it would freeze the one thing Code Bridge adds to a file-backed
    agent. Name and description are the record's own labels; the name that
    selects the agent inside the session comes from the file
    (``cli_agent_runtime._to_definition``), so renaming here cannot break a run.

    A value that is patched to what it already says is not an edit and is not
    refused. That matters for real clients: the app and the dashboard both send
    the whole form on save, so refusing the mere *presence* of ``system_prompt``
    would make renaming a file-backed agent impossible from either.
    """
    if origin.prompt_editable:
        return

    where = origin.source_path
    location = (
        f"the Claude Code agent file '{where}'"
        if where
        else "the agent definition file this agent runs"
    )

    if "system_prompt" in patch:
        current = _instruction_text(agent.get("system_prompt"))
        incoming = _instruction_text(patch.get("system_prompt"))
        if incoming != current:
            raise AgentPromptNotEditableError(
                f"This agent's prompt is {location}. It is read again at the "
                "start of every run, so the text stored here never executes "
                "and was not changed. Edit that file instead.",
                origin,
            )

    if "flow_json" in patch:
        current_steps = _llm_instructions(agent.get("flow_json"))
        for step_id, incoming in _llm_instructions(patch.get("flow_json")).items():
            if current_steps.get(step_id) == incoming:
                continue
            raise AgentPromptNotEditableError(
                f"Step '{step_id}' asks this agent to follow an instruction, "
                f"but what it follows is {location}, read again at the start "
                "of every run. That text would never execute, so it was not "
                "saved. Edit the file, or say what the step needs in its title "
                "and success criteria.",
                origin,
            )


__all__ = [
    "AUTHORED_ORIGIN",
    "ORIGIN_AUTHORED",
    "ORIGIN_CLI_AGENT_FILE",
    "ORIGIN_UNKNOWN",
    "UNKNOWN_ORIGIN",
    "AgentOrigin",
    "AgentPromptNotEditableError",
    "assert_patch_reaches_execution",
    "resolve_agent_origin",
]
