"""Discover CLI agent definitions on this machine and import one as an agent.

All the actual logic lives in :mod:`agent.cli_agent_sources` — this module is
the thin HTTP layer over it, following the same split as ``routes/scripts.py``
/ ``agent/script_store.py``.

Both routes below are on the shared, api-key-gated surface
(``dependencies=[Depends(verify_api_key)]``), not dashboard-only. That is a
deliberate difference from script registration
(``routes.scripts.register_script``), which is dashboard-only because
registering a script hands a workflow step the ability to run an arbitrary
executable on this machine — a new capability class a paired phone should
not be able to grant itself.

Importing an agent definition grants no such new capability. It produces a
normal Code Bridge agent: a name, a system prompt, a declared tool list, and a
single ``llm`` workflow step — exactly the shape ``POST /api/agent/agents``
(also on this shared, api-key-gated router; see ``routes.agents.create_agent``)
already lets an api-key holder create directly, with an arbitrary
system_prompt and flow_json of their choosing. Importing is a convenience
that pre-fills that same request from a file already sitting on the server's
own disk — it is authoring, but the same kind of authoring the phone can
already do, not a new kind. The GET listing is even further from
scripts-style risk: it is read-only, and (per ``CliAgentCandidate.to_view``)
never returns a candidate's body/system-prompt text, only the same class of
metadata ``GET /api/agent/scripts`` already exposes on the same shared
router (names, descriptions, paths).

The two sweep routes sit on the same surface for the same reasons. ``GET
/sweep`` is strictly read-only and, if anything, returns *less* than the live
listing — it reads what :mod:`agent.cli_agent_sweep` already recorded and never
touches the disk. ``PUT /sweep/settings`` flips auto-import, which does grant
something: newly discovered eligible definitions get turned into agents without
a further request. That is not a new capability class either — it is agent
creation, which this same api-key holder can already do directly via ``POST
/api/agent/agents`` — but it is the one switch here that acts while nobody is
watching, which is why it is off by default and why turning it on is not
retroactive.

Two prefixes, one implementation
--------------------------------
The feature was originally called "subagent", so its endpoints were
``/api/agent/subagents``. That name was wrong — see
:mod:`agent.cli_agent_sources` — and the canonical prefix is now
``/api/agent/cli-agents``. The old prefix is still mounted, on the same
handler functions, because it is not this server's to break: a phone running
the previously shipped build calls it on every open of the import screen, and
a 404 there renders as an empty list, i.e. "you have no agents" — the exact
failure mode this feature already learned to avoid once. The alias forwards
rather than reimplements, so the two can never answer differently, and
``test_dashboard_cli_agent_mirrors`` holds both prefixes to the same
dashboard-mirror rule.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from agent.cli_agent_models import CliAgentAutoImportUpdate, CliAgentImportRequest
from agent.cli_agent_sources import (
    CliAgentImportError,
    discover_cli_agents,
    import_cli_agent,
)
from agent.cli_agent_sweep import (
    get_auto_import_enabled,
    get_stored_view,
    set_auto_import_enabled,
)

from .deps import verify_api_key

#: The canonical prefix.
CLI_AGENTS_PREFIX = "/api/agent/cli-agents"

#: The prefix this feature shipped under when it was Claude-only. Kept
#: reachable for already-installed clients; see the module docstring.
LEGACY_CLI_AGENTS_PREFIX = "/api/agent/subagents"

router = APIRouter(prefix=CLI_AGENTS_PREFIX, tags=["cli-agents"])

#: Registered alongside ``router`` (see ``routes/__init__.py``). Every route
#: below is added to both, so there is no path that exists on one and not the
#: other.
legacy_router = APIRouter(prefix=LEGACY_CLI_AGENTS_PREFIX, tags=["cli-agents"])


async def list_cli_agent_candidates() -> dict[str, Any]:
    """Every CLI agent definition file on this machine, parsed or not.

    ``candidates`` is what a client offers for import: the definitions that can
    actually be run on their own. ``excluded`` is every definition that parsed
    fine but exists to be dispatched by something else — scheduling one
    produces a refusal every night — each carrying ``excluded_reason`` and
    ``excluded_detail``. ``skipped`` is every file the sweep could not parse,
    each with a machine-readable ``reason``.

    All three are returned because "the agent I installed is not in the list"
    with no explanation is indistinguishable from a broken sweep. A client can
    show the excluded ones greyed out with their reason; what it must not do is
    offer them for import, and what this must not do is drop them.
    """
    sweep = discover_cli_agents()
    return {
        "candidates": [candidate.to_view() for candidate in sweep.candidates],
        "excluded": [excluded.to_view() for excluded in sweep.excluded],
        "skipped": [skipped.to_view() for skipped in sweep.skipped],
    }


async def get_cli_agent_sweep_state() -> dict[str, Any]:
    """What the server found the last time it looked, without looking again.

    The listing above walks the filesystem on every request; this one reads
    what the periodic sweep (:mod:`agent.cli_agent_sweep`) already wrote. Both
    exist because they answer different questions. ``GET ""`` answers "what is
    on disk right now" — a user who just saved a file and tapped Refresh needs
    that, and would not accept a cached answer. This one answers "what does the
    server know on its own, and when did it last check", which is the only
    question that can be answered while nobody is looking, and the only one
    that can say what *changed*.

    ``known`` is the durable set; ``last_sweep`` carries the timestamp, the
    diff (``new`` / ``missing``), and — when the last attempt failed —
    ``status: "failed"`` with the reason. A failed attempt never empties
    ``known``: "I could not check" and "you have no agents" are different
    statements, and only one of them is ever true here by accident.
    """
    return get_stored_view()


async def update_cli_agent_sweep_settings(
    body: CliAgentAutoImportUpdate,
) -> dict[str, Any]:
    """Turn auto-import of newly discovered eligible definitions on or off.

    The sweep runs and records either way — this only decides whether a newly
    discovered *eligible* definition becomes a Code Bridge agent by itself.
    """
    set_auto_import_enabled(body.enabled)
    return {"auto_import_enabled": get_auto_import_enabled()}


async def import_cli_agent_route(body: CliAgentImportRequest) -> dict[str, Any]:
    """Import one discovered agent definition as a Code Bridge agent.

    ``created`` is ``False`` and the previously created agent is returned
    unchanged when this ``source_path`` was already imported — see
    ``agent.cli_agent_sources.import_cli_agent`` for why re-import is a no-op
    rather than a duplicate.
    """
    try:
        result = import_cli_agent(body.source_path)
    except CliAgentImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "agent": result.agent,
        "created": result.created,
        "reason": result.reason,
    }


#: ``(method, path, handler)`` for every route this module serves. Registering
#: from one table is what makes the canonical and legacy prefixes provably the
#: same surface: a route added here appears on both, and one cannot be added to
#: only one of them by forgetting.
_ROUTES = (
    ("GET", "", list_cli_agent_candidates),
    ("GET", "/sweep", get_cli_agent_sweep_state),
    ("PUT", "/sweep/settings", update_cli_agent_sweep_settings),
    ("POST", "/import", import_cli_agent_route),
)

for _target in (router, legacy_router):
    for _method, _path, _handler in _ROUTES:
        _target.add_api_route(
            _path,
            _handler,
            methods=[_method],
            dependencies=[Depends(verify_api_key)],
            response_model=None,
        )


__all__ = [
    "CLI_AGENTS_PREFIX",
    "LEGACY_CLI_AGENTS_PREFIX",
    "get_cli_agent_sweep_state",
    "import_cli_agent_route",
    "legacy_router",
    "list_cli_agent_candidates",
    "router",
    "update_cli_agent_sweep_settings",
]
