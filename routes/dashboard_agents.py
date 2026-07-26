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
    AgentTaskCreate,
    AgentUpdate,
    BuilderCommitRequest,
    BuilderTurn,
    DryRunRequest,
)
from approvals.approval_models import ApprovalDecisionCreate
from agent.script_models import (
    ScriptDraftRequest,
    ScriptDraftSave,
    ScriptRegister,
    ScriptUpdate,
)
from policy.policy_models import PolicyRuleCreate

from . import agents as agents_routes
from . import approvals as approvals_routes
from . import policies as policies_routes
from . import scripts as scripts_routes
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


# --- Tasks and schedules -------------------------------------------------
#
# An agent on its own never fires. It runs because a task is assigned to it and
# a schedule fires that task, which is why the PC needs these too: without them
# the dashboard could only create agents that sit idle until someone opens the
# app to schedule them.


@router.get("/runs", response_model=None)
async def list_runs(
    agent_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Recent runs — what the dashboard's status view is built from."""
    return await agents_routes.list_runs(agent_id=agent_id, status=status, limit=limit)


@router.get("/tasks/{task_id}/steps", response_model=None)
async def list_task_steps(task_id: str) -> dict[str, Any]:
    """Steps with their outputs — where a shell step's exit code and captured
    stdout/stderr live. This is what turns "the run failed" into something you
    can act on without opening a terminal."""
    return await agents_routes.list_task_steps(task_id)


@router.get("/tasks", response_model=None)
async def list_tasks(
    workspace_id: str | None = None,
    project_name: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=200),
) -> dict[str, Any]:
    return await agents_routes.list_tasks(
        workspace_id=workspace_id,
        project_name=project_name,
        kind=kind,
        status=status,
        limit=limit,
    )


@router.post("/tasks", response_model=None)
async def create_task(body: AgentTaskCreate) -> dict[str, Any]:
    return await agents_routes.create_task(body)


@router.get("/tasks/{task_id}/schedules", response_model=None)
async def list_task_schedules(task_id: str) -> dict[str, Any]:
    return await agents_routes.list_task_schedules(task_id)


@router.post("/tasks/{task_id}/schedules", response_model=None)
async def create_task_schedule(task_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return await agents_routes.create_task_schedule(task_id, body)


@router.get("/schedules", response_model=None)
async def list_all_schedules(enabled_only: bool = False) -> dict[str, Any]:
    return await agents_routes.list_all_schedules(enabled_only=enabled_only)


@router.patch("/schedules/{schedule_id}", response_model=None)
async def patch_schedule(schedule_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return await agents_routes.patch_schedule(schedule_id, body)


@router.delete("/schedules/{schedule_id}", response_model=None)
async def delete_schedule(schedule_id: str) -> dict[str, Any]:
    return await agents_routes.delete_schedule(schedule_id)


@router.post("/schedules/{schedule_id}/trigger", response_model=None)
async def trigger_schedule_now(schedule_id: str) -> dict[str, Any]:
    return await agents_routes.trigger_schedule_now(schedule_id)


# --- Unattended permissions ----------------------------------------------
#
# A scheduled run has no client attached, so a tool permission prompt has
# nobody to answer it and the task parks in waiting_for_user (and, because
# schedules skip while a task is active, every later fire is skipped too).
# Standing "allow" policy rules are what make unattended runs possible, so
# the PC — where you set schedules up — has to be able to see and write them.


@router.get("/policies/rules", response_model=None)
async def list_policy_rules(
    scope: str | None = None,
    operation: str | None = None,
    include_expired: bool = False,
) -> dict[str, Any]:
    return await policies_routes.list_policy_rules(
        scope=scope,
        operation=operation,
        include_expired=include_expired,
    )


@router.post("/policies/rules", response_model=None)
async def create_policy_rule(body: PolicyRuleCreate) -> dict[str, Any]:
    return await policies_routes.create_policy_rule(body)


@router.delete("/policies/rules/{rule_id}", response_model=None)
async def delete_policy_rule(rule_id: str) -> dict[str, Any]:
    return await policies_routes.delete_policy_rule(rule_id)


# Registering a script is dashboard-only: pairing a phone must not hand out
# the ability to point a workflow at any executable on the machine. Listing is
# on the shared router so the phone can still read what a run is doing.


@router.get("/scripts", response_model=None)
async def list_scripts(limit: int = Query(default=100, ge=1, le=200)) -> dict[str, Any]:
    return await scripts_routes.list_scripts(limit=limit)


@router.post("/scripts", response_model=None)
async def register_script(body: ScriptRegister) -> dict[str, Any]:
    return await scripts_routes.register_script(body)


@router.post("/scripts/draft", response_model=None)
async def draft_script(body: ScriptDraftRequest) -> dict[str, Any]:
    """Ask the LLM for a script. Returns text only — nothing is saved or run."""
    return await scripts_routes.draft_script(body)


@router.post("/scripts/save-draft", response_model=None)
async def save_drafted_script(body: ScriptDraftSave) -> dict[str, Any]:
    """Write the reviewed draft into the managed dir and register it."""
    return await scripts_routes.save_drafted_script(body)


@router.patch("/scripts/{script_id}", response_model=None)
async def update_script(script_id: str, body: ScriptUpdate) -> dict[str, Any]:
    return await scripts_routes.update_script(script_id, body)


@router.delete("/scripts/{script_id}", response_model=None)
async def delete_script(script_id: str) -> dict[str, Any]:
    return await scripts_routes.delete_script(script_id)


@router.get("/approvals/pending", response_model=None)
async def list_pending_approvals(run_id: str | None = None) -> dict[str, Any]:
    return await approvals_routes.list_pending_approvals(run_id=run_id)


@router.post("/approvals/{approval_id}/decision", response_model=None)
async def decide_approval(approval_id: str, body: ApprovalDecisionCreate) -> Any:
    return await approvals_routes.create_approval_decision(approval_id, body)


__all__ = ["router"]
