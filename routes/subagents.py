"""Discover Claude Code subagents on this machine and import one as an agent.

All the actual logic lives in :mod:`agent.subagent_sources` — this module is
the thin HTTP layer over it, following the same split as ``routes/scripts.py``
/ ``agent/script_store.py``.

Both routes below are on the shared, api-key-gated surface
(``dependencies=[Depends(verify_api_key)]``), not dashboard-only. That is a
deliberate difference from script registration
(``routes.scripts.register_script``), which is dashboard-only because
registering a script hands a workflow step the ability to run an arbitrary
executable on this machine — a new capability class a paired phone should
not be able to grant itself.

Importing a subagent grants no such new capability. It produces a normal
Code Bridge agent: a name, a system prompt, a declared tool list, and a
single ``llm`` workflow step — exactly the shape ``POST /api/agent/agents``
(also on this shared, api-key-gated router; see ``routes.agents.create_agent``)
already lets an api-key holder create directly, with an arbitrary
system_prompt and flow_json of their choosing. Importing is a convenience
that pre-fills that same request from a file already sitting on the server's
own disk — it is authoring, but the same kind of authoring the phone can
already do, not a new kind. The GET listing is even further from
scripts-style risk: it is read-only, and (per ``SubagentCandidate.to_view``)
never returns a candidate's body/system-prompt text, only the same class of
metadata ``GET /api/agent/scripts`` already exposes on the same shared
router (names, descriptions, paths).

The two sweep routes sit on the same surface for the same reasons. ``GET
/sweep`` is strictly read-only and, if anything, returns *less* than the live
listing — it reads what :mod:`agent.subagent_sweep` already recorded and never
touches the disk. ``PUT /sweep/settings`` flips auto-import, which does grant
something: newly discovered eligible subagents get turned into agents without
a further request. That is not a new capability class either — it is agent
creation, which this same api-key holder can already do directly via ``POST
/api/agent/agents`` — but it is the one switch here that acts while nobody is
watching, which is why it is off by default and why turning it on is not
retroactive.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from agent.subagent_models import SubagentAutoImportUpdate, SubagentImportRequest
from agent.subagent_sources import (
    SubagentImportError,
    discover_subagent_candidates,
    import_subagent,
)
from agent.subagent_sweep import (
    get_auto_import_enabled,
    get_stored_view,
    set_auto_import_enabled,
)

from .deps import verify_api_key

router = APIRouter(prefix="/api/agent/subagents", tags=["subagents"])


@router.get("", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_subagent_candidates() -> dict[str, Any]:
    """Every Claude Code subagent file on this machine, parsed or not.

    ``candidates`` is what a client offers for import: the subagents that can
    actually be run on their own. ``excluded`` is every subagent that parsed
    fine but exists to be dispatched by something else — scheduling one
    produces a refusal every night — each carrying ``excluded_reason`` and
    ``excluded_detail``. ``skipped`` is every file the sweep could not parse,
    each with a machine-readable ``reason``.

    All three are returned because "the agent I installed is not in the list"
    with no explanation is indistinguishable from a broken sweep. A client can
    show the excluded ones greyed out with their reason; what it must not do is
    offer them for import, and what this must not do is drop them.
    """
    sweep = discover_subagent_candidates()
    return {
        "candidates": [candidate.to_view() for candidate in sweep.candidates],
        "excluded": [excluded.to_view() for excluded in sweep.excluded],
        "skipped": [skipped.to_view() for skipped in sweep.skipped],
    }


@router.get("/sweep", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_subagent_sweep_state() -> dict[str, Any]:
    """What the server found the last time it looked, without looking again.

    The listing above walks the filesystem on every request; this one reads
    what the periodic sweep (:mod:`agent.subagent_sweep`) already wrote. Both
    exist because they answer different questions. ``GET ""`` answers "what is
    on disk right now" — a user who just saved a file and tapped Refresh needs
    that, and would not accept a cached answer. This one answers "what does the
    server know on its own, and when did it last check", which is the only
    question that can be answered while nobody is looking, and the only one
    that can say what *changed*.

    ``known`` is the durable set; ``last_sweep`` carries the timestamp, the
    diff (``new`` / ``missing``), and — when the last attempt failed —
    ``status: "failed"`` with the reason. A failed attempt never empties
    ``known``: "I could not check" and "you have no subagents" are different
    statements, and only one of them is ever true here by accident.
    """
    return get_stored_view()


@router.put("/sweep/settings", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_subagent_sweep_settings(body: SubagentAutoImportUpdate) -> dict[str, Any]:
    """Turn auto-import of newly discovered eligible subagents on or off.

    The sweep runs and records either way — this only decides whether a newly
    discovered *eligible* subagent becomes a Code Bridge agent by itself.
    """
    set_auto_import_enabled(body.enabled)
    return {"auto_import_enabled": get_auto_import_enabled()}


@router.post("/import", dependencies=[Depends(verify_api_key)], response_model=None)
async def import_subagent_route(body: SubagentImportRequest) -> dict[str, Any]:
    """Import one discovered subagent as a Code Bridge agent.

    ``created`` is ``False`` and the previously created agent is returned
    unchanged when this ``source_path`` was already imported — see
    ``agent.subagent_sources.import_subagent`` for why re-import is a no-op
    rather than a duplicate.
    """
    try:
        result = import_subagent(body.source_path)
    except SubagentImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "agent": result.agent,
        "created": result.created,
        "reason": result.reason,
    }


__all__ = ["router"]
