"""The canvas's own API front: four routes, and deliberately only four.

This is the third authentication face in the server, and it exists because the
first two do not fit a browser on a phone:

===================  ==========================  ===========================
face                 gate                        who
===================  ==========================  ===========================
``/api/agent/*``     ``verify_api_key``          the phone app, paired key
``/api/dashboard/*`` ``require_local_access``    the desktop dashboard, no key
``/api/canvas/*``    ``verify_canvas_token``     the canvas page in a browser
===================  ==========================  ===========================

The canvas is JavaScript. Whatever credential it holds is readable by every
script on the page, and no amount of care removes that — so the design does not
try to. It removes the *value* of the leak instead: a canvas token opens one
agent's workflow graph for fifteen minutes and nothing else. It cannot start a
run, execute a ``shell`` step, read the secret store, or reach the builder.

That is a promise about a *list*, so the list is fixed and tested:

===========  =========================================  ===============
method       path                                       delegates to
===========  =========================================  ===============
``GET``      ``/api/canvas/agents/{id}/graph``          ``routes.agents.get_agent``
``PATCH``    ``/api/canvas/agents/{id}/graph``          ``routes.agents.update_agent``
``GET``      ``/api/canvas/workflow/step-schema``       ``routes.agents.get_workflow_step_schema``
``GET``      ``/api/canvas/option-sources/{name}``      ``routes.scripts`` / ``routes.devices``
===========  =========================================  ===============

``tests/test_canvas_api_surface.py`` asserts that table exactly. It is meant to
go red when a fifth route is added: the request that will eventually arrive is
"let me run it straight from the canvas", and granting it silently turns every
leaked canvas token into shell execution on the operator's machine. Growing
this list is a decision someone has to make on purpose, in the open.

Every handler **delegates** to the existing handler rather than
re-implementing it — the same discipline as :mod:`routes.dashboard_agents`, and
for the same reason: validation, pseudo-agent protection, workflow
normalisation and the ``flow_graph`` folding gates stay in one place.

The token travels in the ``X-Canvas-Token`` **header**, never a query string.
Preview tokens ride in the URL (``?preview_token=``) and that is defensible for
a read-only proxy; a canvas token can *write*, and a URL is copied into access
logs, ``Referer`` headers, WebView history and shared links.
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from code_bridge_core.workflow_step_schema import OPTION_SOURCES
from audit.route_audit import record_api_action
from canvas.canvas_access import (
    CANVAS_TOKEN_TTL_MINUTES,
    CanvasSession,
    REASON_MISSING,
    get_canvas_session_manager,
    token_fingerprint,
)
from core.config import get_config

from . import agents as agents_routes
from . import devices as devices_routes
from . import scripts as scripts_routes
from .agents import AgentUpdateBody
from .deps import is_request_from_tunnel, verify_api_key

CANVAS_TOKEN_HEADER = "X-Canvas-Token"

# Query keys that would look like a plausible place to put the token. Refused
# loudly rather than ignored, so a client that tries the preview-style
# ``?token=`` gets told why instead of a bare 401 it will read as "my token is
# broken" and then work around by... putting it somewhere worse.
_REFUSED_QUERY_KEYS = ("token", "canvas_token", "api_key")

# Issuance lives under the api-key-gated prefix: the app trades the credential
# it already has for a weaker one. A separate router (not a fifth route on the
# canvas router) so the surface test can hold ``/api/canvas`` to exactly four.
session_router = APIRouter(prefix="/api/agent/canvas", tags=["canvas"])
router = APIRouter(prefix="/api/canvas", tags=["canvas"])


# --- audit -----------------------------------------------------------------

def _audit(
    operation: str,
    *,
    agent_id: str | None,
    fingerprint: str | None,
    request: Request | None,
    success: bool,
    status_code: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Record a canvas event with a token *fingerprint* — never a token.

    Built from named arguments rather than a caller-supplied dict so there is
    no parameter through which a raw token could be passed by accident. Same
    discipline as ``routes/secrets.py``'s ``_audit``, which will not accept a
    secret value.
    """
    details: dict[str, Any] = {}
    if agent_id is not None:
        details["agent_id"] = agent_id
    if fingerprint is not None:
        details["token_fingerprint"] = fingerprint
    if request is not None:
        details["via_tunnel"] = is_request_from_tunnel(request)
    if extra:
        details.update(extra)
    record_api_action(
        operation=operation,
        details=details,
        success=success,
        status_code=status_code,
    )


# --- gate ------------------------------------------------------------------

def _canvas_external_enabled() -> bool:
    """Whether ``/api/canvas/*`` answers tunnel requests at all.

    Defaults to true: the requirement that produced this whole track is "I use
    it from my phone", and the phone reaches the server through the tunnel.
    The switch exists so an operator who does not want that can close it
    without the canvas having to be un-shipped.
    """
    return bool(getattr(get_config(), "canvas_external_enabled", True))


def _require_external_allowed(request: Request) -> None:
    if is_request_from_tunnel(request) and not _canvas_external_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "The canvas API is disabled for external (tunnel) access on"
                " this server. Set server.canvas_external_enabled: true in"
                " config.yaml to allow it, or open the canvas from the local"
                " network instead."
            ),
        )


async def verify_canvas_token(
    request: Request,
    x_canvas_token: str | None = Header(None, alias=CANVAS_TOKEN_HEADER),
) -> CanvasSession:
    """Resolve the presented canvas token, or refuse.

    Order matters: the external switch is checked before the token store is
    touched, so a server with the switch closed does not become an oracle for
    which tokens are live.
    """
    _require_external_allowed(request)

    present_refused = [key for key in _REFUSED_QUERY_KEYS if key in request.query_params]
    if present_refused:
        raise HTTPException(
            status_code=400,
            detail=(
                "The canvas token must be sent in the"
                f" {CANVAS_TOKEN_HEADER} header, not in the query string"
                f" (saw: {', '.join(present_refused)}). A URL is copied into"
                " access logs, Referer headers and browser history, and this"
                " token can write."
            ),
        )

    lookup = get_canvas_session_manager().resolve(x_canvas_token)
    if lookup.session is None:
        reason = lookup.reason or REASON_MISSING
        _audit(
            "canvas.token.rejected",
            agent_id=None,
            # A rejected token still gets fingerprinted so a burst of retries
            # after expiry is legible as one session, not N mysteries.
            fingerprint=token_fingerprint(x_canvas_token) if x_canvas_token else None,
            request=request,
            success=False,
            status_code=401,
            extra={"reason": reason},
        )
        raise HTTPException(
            status_code=401,
            detail=(
                f"Canvas token {reason}. Ask the app for a new one via"
                " POST /api/agent/canvas/session and send it in the"
                f" {CANVAS_TOKEN_HEADER} header."
            ),
        )
    return lookup.session


def _require_token_agent(
    session: CanvasSession, agent_id: str, request: Request
) -> None:
    """A token is for one agent. Another agent's graph is not in scope."""
    if session.agent_id == agent_id:
        return
    _audit(
        "canvas.token.rejected",
        agent_id=agent_id,
        fingerprint=session.fingerprint,
        request=request,
        success=False,
        status_code=403,
        extra={"reason": "agent_mismatch", "token_agent_id": session.agent_id},
    )
    raise HTTPException(
        status_code=403,
        detail=(
            "This canvas token was issued for a different agent. Tokens are"
            " scoped to the agent named at issue time; request a new one for"
            f" '{agent_id}'."
        ),
    )


# --- issuance --------------------------------------------------------------

class CanvasSessionRequest(BaseModel):
    """Ask for a canvas token for one agent."""

    agent_id: str = Field(min_length=1)


@session_router.post("/session", response_model=None)
async def create_canvas_session(
    body: CanvasSessionRequest,
    request: Request,
    api_key: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Trade a paired API key for a weaker, shorter-lived canvas token.

    The API key never reaches the browser. The app calls this, receives a
    token scoped to one agent, and hands *that* to the page (over a JS
    channel, not a URL — see T-I1-11).
    """
    _require_external_allowed(request)
    # Refuses with 404 before a token exists for an agent that does not. Uses
    # the agents module's own lookup so "which agents exist" has one answer.
    agents_routes._require_agent(body.agent_id)

    session = get_canvas_session_manager().issue(body.agent_id, api_key=api_key)
    _audit(
        "canvas.session.issue",
        agent_id=session.agent_id,
        fingerprint=session.fingerprint,
        request=request,
        success=True,
        status_code=200,
        extra={"ttl_minutes": CANVAS_TOKEN_TTL_MINUTES, "scope": list(session.scope)},
    )
    return {
        "token": session.token,
        "agent_id": session.agent_id,
        "scope": list(session.scope),
        "expires_in_minutes": CANVAS_TOKEN_TTL_MINUTES,
        "expires_at": session.expires_at.isoformat(),
        # Named in the response so a client never has to guess and never has a
        # reason to reach for the query string.
        "token_header": CANVAS_TOKEN_HEADER,
    }


# --- the four routes -------------------------------------------------------

_GRAPH_VIEW_KEYS = (
    "id",
    "name",
    "description",
    "updated_at",
    # The canvas holds this and sends it back as `if_flow_revision` on save,
    # so a graph read that omitted it would leave the browser — the writer
    # most likely to be racing the app's edit screen — the one client with no
    # way to say which version it is editing. It is a hash of the stored
    # workflow and reveals nothing the graph beside it does not.
    "flow_revision",
    "flow_graph",
    "flow_graph_issues",
    "flow_graph_unavailable",
)


def _graph_view(agent_payload: dict[str, Any]) -> dict[str, Any]:
    """The graph-shaped subset of an agent payload.

    ``GET /api/agent/agents/{id}`` answers with the whole agent — system
    prompt, tools, provider, policy overrides. A ``graph:read`` token asked
    for the graph, so it gets the graph plus the labels needed to title the
    screen. Narrowing here rather than trusting the caller keeps the token's
    written scope and its actual reach the same thing.
    """
    return {key: agent_payload[key] for key in _GRAPH_VIEW_KEYS if key in agent_payload}


@router.get("/agents/{agent_id}/graph", response_model=None)
async def read_agent_graph(
    agent_id: str,
    request: Request,
    session: CanvasSession = Depends(verify_canvas_token),
) -> dict[str, Any]:
    """The kernel wire-form graph for one agent (or why there is none).

    Delegates to ``routes.agents.get_agent``, which answers 200 with
    ``flow_graph_unavailable`` when the kernel is absent or the stored
    workflow does not fold — that distinction is worth keeping, so it is
    passed through rather than flattened into an error.
    """
    _require_token_agent(session, agent_id, request)
    payload = await agents_routes.get_agent(agent_id)
    view = _graph_view(payload)
    _audit(
        "canvas.graph.read",
        agent_id=agent_id,
        fingerprint=session.fingerprint,
        request=request,
        success=True,
        status_code=200,
        extra={"graph_available": "flow_graph" in view},
    )
    return view


class CanvasGraphPatch(BaseModel):
    """A whole graph, as the canvas drew it."""

    flow_graph: dict[str, Any]
    # The same escape hatch ``AgentUpdateBody`` offers: "yes, save it even
    # though the contract gate says it will stall". Passed through so the
    # canvas can offer the choice the phone and dashboard already do.
    commit_incomplete: bool = False
    # "Only if the stored workflow is still the one I drew from" — the
    # revision this canvas last read, echoed back. Optional here for exactly
    # the reason it is optional on ``AgentUpdateBody``: a canvas bundle older
    # than this change sends none and keeps working. The shipped canvas always
    # sends it.
    if_flow_revision: str | None = None


@router.patch("/agents/{agent_id}/graph", response_model=None)
async def write_agent_graph(
    agent_id: str,
    body: CanvasGraphPatch,
    request: Request,
    session: CanvasSession = Depends(verify_canvas_token),
) -> dict[str, Any] | JSONResponse:
    """Save the graph, through the existing ``flow_graph`` write path.

    No new validation lives here. ``routes.agents.update_agent`` folds the
    graph with ``_fold_flow_graph_input`` and refuses in three named ways —
    ``kernel_not_installed`` (422), ``invalid_flow_graph`` (400) and
    ``unsupported_topology`` (400, carrying every issue) — and, when the
    request carried ``if_flow_revision``, a fourth: ``flow_revision_conflict``
    (409), the workflow having moved since this canvas read it. Those refusals
    are returned verbatim: the canvas is supposed to list the issues, and a
    summary here would be a summary of something the user needs in full.
    """
    _require_token_agent(session, agent_id, request)
    result = await agents_routes.update_agent(
        agent_id,
        AgentUpdateBody(
            flow_graph=body.flow_graph,
            commit_incomplete=body.commit_incomplete,
            if_flow_revision=body.if_flow_revision,
        ),
    )
    if isinstance(result, JSONResponse):
        _audit(
            "canvas.graph.write",
            agent_id=agent_id,
            fingerprint=session.fingerprint,
            request=request,
            success=False,
            status_code=result.status_code,
            extra={"refused": True},
        )
        return result

    # Re-read rather than reshaping the update's payload: the graph the canvas
    # should now draw is the one the server folded and stored, not the one it
    # sent. Optimistic redraw is what makes a rejected edge reappear.
    fresh = await agents_routes.get_agent(agent_id)
    _audit(
        "canvas.graph.write",
        agent_id=agent_id,
        fingerprint=session.fingerprint,
        request=request,
        success=True,
        status_code=200,
    )
    return _graph_view(fresh)


@router.get("/workflow/step-schema", response_model=None)
async def canvas_step_schema(
    session: CanvasSession = Depends(verify_canvas_token),
) -> dict[str, Any]:
    """The same form schema the phone and dashboard draw from.

    Read-only and identical for every agent, so it is not agent-scoped: a
    token for agent A asking for the step schema is asking about the server's
    step vocabulary, not about agent B.
    """
    return await agents_routes.get_workflow_step_schema()


# The largest page ``routes/scripts.py`` will serve (``le=200``). Passed
# explicitly because these handlers are called as functions, not over HTTP:
# an omitted argument would hand the store FastAPI's ``Query`` default object
# instead of a number — the same trap ``routes/dashboard_agents.py`` avoids by
# always naming what it forwards.
_OPTION_SOURCE_SCRIPT_LIMIT = 200


async def _option_source_scripts() -> dict[str, Any]:
    return await scripts_routes.list_scripts(limit=_OPTION_SOURCE_SCRIPT_LIMIT)


async def _option_source_devices() -> dict[str, Any]:
    return await devices_routes.list_devices()


async def _option_source_mcp_servers() -> dict[str, Any]:
    # Delegates like the other two — names and origins only, never the
    # entries (see the handler's docstring in routes/system_settings.py).
    from . import system_settings as system_settings_routes

    return await system_settings_routes.list_detected_mcp_servers()


# Keyed by the names ``code_bridge_core/workflow_step_schema.py`` publishes in
# ``OPTION_SOURCES``. A test asserts these two sets match, so a new option
# source added to the schema fails loudly here instead of leaving the canvas
# with an empty dropdown and no explanation.
_OPTION_SOURCE_RESOLVERS: dict[str, Callable[[], Coroutine[Any, Any, dict[str, Any]]]] = {
    "scripts": _option_source_scripts,
    "devices": _option_source_devices,
    "mcp-servers": _option_source_mcp_servers,
}


@router.get("/option-sources/{name}", response_model=None)
async def canvas_option_source(
    name: str,
    session: CanvasSession = Depends(verify_canvas_token),
) -> dict[str, Any]:
    """Options for a ``select`` field whose choices are not inline.

    The canvas cannot call ``/api/agent/scripts`` or ``/api/devices`` — it has
    no API key, and giving it one is the thing this whole design refuses. So
    the two read-only catalogs the step schema points at are re-exposed here,
    behind the canvas token, by name rather than by URL.
    """
    resolver = _OPTION_SOURCE_RESOLVERS.get(name)
    if resolver is None:
        known = sorted(_OPTION_SOURCE_RESOLVERS)
        if name in OPTION_SOURCES:
            detail = (
                f"Option source '{name}' is published by the workflow step"
                " schema but is not wired into the canvas API. Add a resolver"
                " in routes/canvas_api.py."
            )
            raise HTTPException(status_code=501, detail=detail)
        raise HTTPException(
            status_code=404,
            detail=f"Unknown option source '{name}'. Known: {', '.join(known)}.",
        )
    payload = await resolver()
    return {"name": name, **payload}
