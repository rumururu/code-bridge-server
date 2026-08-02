"""Scripts the Agent Builder conversation asks for, drafts, and never installs.

A ``shell`` step names a registered script by id. When the Configurator needed
one that did not exist it could only describe it in prose, so the conversation
dead-ended: the user left, opened the dashboard, had a script written there,
registered it, came back, and re-explained what the agent was for. This module
is the path from the conversation to the script writer that already exists in
:mod:`routes.scripts`.

Three properties are the whole design, and each is enforced here rather than
promised in a comment:

**A proposal never registers itself.** Drafting (:func:`draft_pending_proposals`)
calls :func:`routes.scripts.draft_script`, which writes nothing and registers
nothing; it cannot reach the script store at all. Registration lives only in
:func:`approve_script_proposal`, which is exported for the *dashboard* router
to wrap — exactly like :func:`routes.scripts.save_drafted_script` and for the
same reason stated in that module: pairing a phone must not hand out the
ability to put a new executable in a workflow's path. The phone sees the
proposal and the drafted text; a person on the PC approves it.

**Approval carries no content.** The endpoint takes a proposal id and nothing
else — no body, no name, no path. What gets registered is byte-for-byte what
the user was shown, in the managed directory the script writer already uses,
through the same registration call with the same path validation.

**A failure is a failure.** If the script writer errors, times out or returns
nothing, the proposal goes to ``failed`` with the reason. No rule-based script
is synthesised to keep the conversation moving; a hand-rolled stand-in would be
a script nobody wrote, waiting for someone to approve it on the strength of the
LLM's name being on it.

Where drafting runs: as its own job, after the turn is answered, never inside
it. The script writer is allowed 180 seconds; a Configurator turn has a 20
second fast window before the client is told to poll. Drafting inside the turn
would either blow that window on every proposal or hold the conversational
reply — which is ready — hostage to an unrelated LLM call for minutes. It is
also a separate failure: a script that could not be drafted must not turn a
perfectly good turn into an error. So the turn returns the proposal in
``drafting`` immediately, and the body arrives on
``GET /api/agent/builder/scripts/proposals/{id}`` (and on every later turn's
``script_proposals``).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from agent.agent_models import AgentDraft, ScriptProposalView, ScriptRequestDraft
from agent.script_models import ScriptDraftRequest, ScriptDraftSave

from .deps import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent/builder/scripts", tags=["script-proposals"])

# Outlives the 30-minute builder session on purpose: a user who lets the
# conversation lapse while reading the drafted script should still be able to
# open the proposal and approve it.
PROPOSAL_TTL = timedelta(hours=2)

STATUS_DRAFTING = "drafting"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_REGISTERED = "registered"


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class ScriptProposal:
    """One requested script, from asked-for to (maybe) registered."""

    id: str
    session_id: str
    name: str
    purpose: str
    expected_output: str = ""
    step_id: str | None = None
    status: str = STATUS_DRAFTING
    interpreter: str = "bash"
    body: str | None = None
    error: str | None = None
    script_id: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    # Held across the whole of approve_script_proposal so two approvals of the
    # same proposal — a double tap, or the phone and dashboard at once — cannot
    # both get past the status check and register the script twice.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or _utcnow()) - self.updated_at > PROPOSAL_TTL


SCRIPT_PROPOSALS: dict[str, ScriptProposal] = {}


def clear_expired_script_proposals(now: datetime | None = None) -> None:
    reference = now or _utcnow()
    for proposal_id in [
        proposal_id
        for proposal_id, proposal in SCRIPT_PROPOSALS.items()
        if proposal.is_expired(reference)
    ]:
        SCRIPT_PROPOSALS.pop(proposal_id, None)


def _normalized(name: str) -> str:
    return " ".join(name.split()).casefold()


def proposal_view(proposal: ScriptProposal) -> ScriptProposalView:
    return ScriptProposalView(
        proposal_id=proposal.id,
        name=proposal.name,
        purpose=proposal.purpose,
        expected_output=proposal.expected_output,
        step_id=proposal.step_id,
        status=proposal.status,
        interpreter=proposal.interpreter,
        body=proposal.body,
        error=proposal.error,
        script_id=proposal.script_id,
    )


def list_proposals_for_session(session_id: str) -> list[ScriptProposal]:
    clear_expired_script_proposals()
    return [
        proposal
        for proposal in SCRIPT_PROPOSALS.values()
        if proposal.session_id == session_id
    ]


def sync_proposals_for_session(
    session_id: str,
    requests: list[ScriptRequestDraft],
) -> list[ScriptProposal]:
    """Create a proposal for each newly requested script; reuse the rest.

    The Configurator re-emits its whole draft every turn, so the same
    ``script_requests`` entry arrives again on each one. Matching by name means
    a five-turn conversation drafts the script once instead of five times —
    each draft being an LLM call — and, more importantly, that the text the
    user is reading does not silently change under them between turns.
    """
    clear_expired_script_proposals()
    existing = {
        _normalized(proposal.name): proposal
        for proposal in list_proposals_for_session(session_id)
    }
    for request in requests:
        key = _normalized(request.name)
        if key in existing:
            continue
        proposal = ScriptProposal(
            id=f"script_proposal_{uuid.uuid4().hex}",
            session_id=session_id,
            name=request.name.strip(),
            purpose=request.purpose.strip(),
            expected_output=(request.expected_output or "").strip(),
            step_id=(request.step_id or "").strip() or None,
        )
        SCRIPT_PROPOSALS[proposal.id] = proposal
        existing[key] = proposal
        logger.info(
            "script_proposal_requested proposal_id=%s session_id=%s name=%s",
            proposal.id,
            session_id,
            proposal.name,
        )
    return list_proposals_for_session(session_id)


def _drafting_intent(proposal: ScriptProposal) -> str:
    lines = [proposal.purpose]
    if proposal.expected_output:
        lines.append(f"On success it must print: {proposal.expected_output}")
    lines.append(
        "It will be run unattended by a scheduled agent workflow step, so it "
        "must be non-interactive and say why it failed on stderr."
    )
    return "\n".join(lines)


async def draft_pending_proposals(session_id: str) -> None:
    """Draft every proposal of this session that has no body yet.

    Called after the turn has been answered — as a FastAPI background task on
    the synchronous route, and inline at the end of the already-backgrounded
    converse job. Either way the user has their reply before this starts.
    """
    for proposal in list_proposals_for_session(session_id):
        if proposal.status != STATUS_DRAFTING or proposal.body is not None:
            continue
        await _draft_proposal(proposal)


async def _draft_proposal(proposal: ScriptProposal) -> None:
    # Imported here, not at module scope: routes.scripts pulls in the LLM
    # stack lazily and this module is imported at router registration.
    from .scripts import draft_script

    try:
        drafted = await draft_script(ScriptDraftRequest(intent=_drafting_intent(proposal)))
    except HTTPException as exc:
        _record_drafting_failure(proposal, str(exc.detail))
        return
    except Exception as exc:  # noqa: BLE001 - the reason is the payload
        _record_drafting_failure(proposal, str(exc) or exc.__class__.__name__)
        return

    body = str(drafted.get("body") or "")
    if not body.strip():
        # draft_script already refuses to return an empty body, so this is
        # belt-and-braces — but "ready" with nothing in it would be a proposal
        # the user could approve into an empty file.
        _record_drafting_failure(proposal, "The model returned an empty script")
        return
    proposal.body = body
    proposal.interpreter = str(drafted.get("interpreter") or "bash")
    proposal.status = STATUS_READY
    proposal.error = None
    proposal.touch()
    logger.info("script_proposal_drafted proposal_id=%s", proposal.id)


def _record_drafting_failure(proposal: ScriptProposal, reason: str) -> None:
    """Report the failure and leave the proposal empty.

    Deliberately no fallback body. A server-written stand-in would arrive
    looking exactly like a drafted script — same card, same approve button —
    and the one thing the user is being asked to judge is whether the script
    is right.
    """
    proposal.status = STATUS_FAILED
    proposal.error = reason
    proposal.body = None
    proposal.touch()
    logger.warning(
        "script_proposal_draft_failed proposal_id=%s reason=%s", proposal.id, reason
    )


def _require_proposal(proposal_id: str) -> ScriptProposal:
    clear_expired_script_proposals()
    proposal = SCRIPT_PROPOSALS.get(proposal_id)
    if proposal is None:
        raise HTTPException(
            status_code=404, detail=f"Script proposal '{proposal_id}' not found"
        )
    return proposal


@router.get(
    "/proposals/{proposal_id}",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def get_script_proposal(proposal_id: str) -> dict[str, Any]:
    """Poll a proposal. Reading it is not approving it."""
    return {"proposal": proposal_view(_require_proposal(proposal_id)).model_dump()}


async def approve_script_proposal(proposal_id: str) -> dict[str, Any]:
    """Register an approved proposal. Dashboard-only — see module docstring.

    This is the only place in the builder path that writes a script to disk or
    into the registry, and it is reached only by an explicit request naming a
    proposal the user has read. It takes no body: the bytes registered are the
    bytes shown.
    """
    proposal = _require_proposal(proposal_id)
    async with proposal.lock:
        if proposal.status == STATUS_REGISTERED and proposal.script_id:
            # Idempotent: a second approval returns the first one's script
            # rather than writing a second copy under a suffixed name.
            return _approval_payload(proposal, get_script(proposal.script_id))

        if proposal.status != STATUS_READY or not proposal.body:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Script proposal '{proposal_id}' is {proposal.status} and "
                    "cannot be approved" + (f": {proposal.error}" if proposal.error else "")
                ),
            )

        from .scripts import save_drafted_script

        try:
            saved = await save_drafted_script(
                ScriptDraftSave(
                    name=proposal.name,
                    body=proposal.body,
                    description=proposal.purpose,
                    interpreter=proposal.interpreter,
                )
            )
        except HTTPException as exc:
            # A name collision is the common one: something is already
            # registered under that filename. Say so and leave the proposal
            # approvable — nothing was written, and the existing script is
            # untouched. The user renames it, or asks for a different name in
            # the conversation.
            proposal.error = str(exc.detail)
            proposal.touch()
            raise

        script = saved["script"]
        proposal.script_id = str(script["id"])
        proposal.status = STATUS_REGISTERED
        proposal.error = None
        proposal.touch()
        logger.info(
            "script_proposal_approved proposal_id=%s script_id=%s",
            proposal.id,
            proposal.script_id,
        )
        return _approval_payload(proposal, script)


def get_script(script_id: str) -> dict[str, Any] | None:
    from agent.script_store import get_script_store

    return get_script_store().get(script_id)


def _approval_payload(
    proposal: ScriptProposal,
    script: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "proposal": proposal_view(proposal).model_dump(),
        "script": script,
        "script_id": proposal.script_id,
    }
    updated_draft = _teach_session_about_script(proposal, script)
    if updated_draft is not None:
        payload["updated_draft"] = updated_draft.model_dump()
    return payload


def _teach_session_about_script(
    proposal: ScriptProposal,
    script: dict[str, Any] | None,
) -> AgentDraft | None:
    if not script:
        return None
    from agent.configurator import get_builder_session

    session = get_builder_session(proposal.session_id)
    if session is None:
        # The conversation lapsed while the user was reading. The script is
        # registered and the id is in the response; a new conversation will
        # find it in the registered-scripts list like any other.
        return None
    return session.apply_registered_script(
        script=script,
        step_id=proposal.step_id,
        request_name=proposal.name,
    )


__all__ = [
    "PROPOSAL_TTL",
    "SCRIPT_PROPOSALS",
    "ScriptProposal",
    "approve_script_proposal",
    "clear_expired_script_proposals",
    "draft_pending_proposals",
    "get_script_proposal",
    "list_proposals_for_session",
    "proposal_view",
    "router",
    "sync_proposals_for_session",
]
