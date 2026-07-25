"""Agent builder endpoints for the PC dashboard.

The phone talks to ``/api/agent/*`` with a paired API key. The dashboard has
no key — it is trusted because it is served on the localhost-only listener —
so these thin wrappers expose the same agent operations behind
:func:`require_local_access` instead.

They deliberately delegate to the handlers in :mod:`routes.agents` rather than
re-implementing anything: validation, pseudo-agent protection, workflow
normalisation and the builder session machinery stay in one place. Registering
them only in ``_DASHBOARD_ONLY_ROUTERS`` keeps them off the tunnel-exposed API
app entirely, so this adds no external surface.
"""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, Response

from agent.agent_models import (
    AgentCreate,
    AgentUpdate,
    BuilderCommitRequest,
    BuilderTurn,
    DryRunRequest,
)

from . import agents as agents_routes
from .deps import require_local_access

router = APIRouter(
    prefix="/api/dashboard/agent",
    tags=["dashboard-agent"],
    dependencies=[Depends(require_local_access)],
)


@router.get("/agents", response_model=None)
async def list_agents(
    include_archived: bool = False,
    include_pseudo: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return await agents_routes.list_agents(
        include_archived=include_archived,
        include_pseudo=include_pseudo,
        limit=limit,
    )


@router.post("/agents", response_model=None)
async def create_agent(body: AgentCreate) -> dict[str, Any] | JSONResponse:
    return await agents_routes.create_agent(body)


@router.get("/agents/{agent_id}", response_model=None)
async def get_agent(agent_id: str) -> dict[str, Any]:
    return await agents_routes.get_agent(agent_id)


@router.patch("/agents/{agent_id}", response_model=None)
async def update_agent(
    agent_id: str,
    body: AgentUpdate,
) -> dict[str, Any] | JSONResponse:
    return await agents_routes.update_agent(agent_id, body)


@router.delete("/agents/{agent_id}", response_model=None)
async def delete_agent(
    agent_id: str,
    archive: bool = True,
) -> dict[str, Any] | JSONResponse:
    return await agents_routes.delete_agent(agent_id, archive=archive)


@router.post("/builder/converse", response_model=None)
async def builder_converse(body: BuilderTurn) -> Any:
    """One turn of the conversational builder — same session store the app uses."""
    return await agents_routes.builder_converse(body)


@router.post("/builder/commit", response_model=None)
async def builder_commit(body: BuilderCommitRequest) -> Any:
    """Persist the agent the builder conversation has been assembling."""
    return await agents_routes.builder_commit(body)


@router.post("/agents/{agent_id}/dry-run", response_model=None)
async def start_dry_run(
    agent_id: str,
    body: DryRunRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    return await agents_routes.start_dry_run(agent_id, body, background_tasks)


__all__ = ["router"]
