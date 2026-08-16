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
    AgentRunOnceRequest,
    AgentTaskCreate,
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
# The mirrors below must declare the *route-level* body models, not the shared
# draft models. `agents_routes` reads `body.commit_incomplete`, which only the
# Body subclasses carry; typing a mirror with the plain model made every
# dashboard create/update/commit raise AttributeError at runtime while the
# phone's identical call succeeded, because only the phone reached the real
# route. Tests did not catch it — they exercised `routes/agents.py` directly.
from .agents import AgentCreateBody, AgentUpdateBody, BuilderCommitBody
from . import approvals as approvals_routes
from . import policies as policies_routes
from . import script_proposals as script_proposals_routes
from . import scripts as scripts_routes
from . import cli_agents as cli_agents_routes
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
async def create_agent(body: AgentCreateBody) -> dict[str, Any] | JSONResponse:
    return await agents_routes.create_agent(body)


@router.get("/agents/{agent_id}", response_model=None)
async def get_agent(agent_id: str) -> dict[str, Any]:
    return await agents_routes.get_agent(agent_id)


@router.patch("/agents/{agent_id}", response_model=None)
async def update_agent(
    agent_id: str,
    body: AgentUpdateBody,
) -> dict[str, Any] | JSONResponse:
    return await agents_routes.update_agent(agent_id, body)


@router.delete("/agents/{agent_id}", response_model=None)
async def delete_agent(
    agent_id: str,
    archive: bool = True,
) -> dict[str, Any] | JSONResponse:
    return await agents_routes.delete_agent(agent_id, archive=archive)


@router.post("/builder/converse", response_model=None)
async def builder_converse(body: BuilderTurn, background_tasks: BackgroundTasks) -> Any:
    """One turn of the conversational builder — same session store the app uses."""
    return await agents_routes.builder_converse(body, background_tasks)


@router.post("/builder/converse/jobs", response_model=None)
async def create_builder_converse_job(
    body: BuilderTurn,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Start a turn that may outlast the synchronous window.

    The Configurator regularly takes longer than the fast path allows, and
    the synchronous route answers a timeout with a failure the user reads as
    "it is broken". The app already polls this job; the dashboard had no
    mirror for it, so the PC was the only place the builder appeared to fail.
    """
    return await agents_routes.create_builder_converse_job(body, background_tasks)


@router.get("/builder/converse/jobs/{job_id}", response_model=None)
async def get_builder_converse_job(job_id: str) -> dict[str, Any]:
    return await agents_routes.get_builder_converse_job(job_id)


@router.post("/builder/commit", response_model=None)
async def builder_commit(body: BuilderCommitBody) -> Any:
    """Persist the agent the builder conversation has been assembling."""
    return await agents_routes.builder_commit(body)


@router.get("/workflow/step-schema", response_model=None)
async def get_workflow_step_schema() -> dict[str, Any]:
    """Same schema the phone gets from `/api/agent/workflow/step-schema`.

    That route is api-key-gated (agents_router is shared with the
    tunnel-exposed app); the dashboard has no key, so it needs this mirror
    like every other agent-builder read below.
    """
    return await agents_routes.get_workflow_step_schema()


@router.get("/browser-runtime/readiness", response_model=None)
async def browser_runtime_readiness() -> dict[str, Any]:
    """Same probe the phone gets from `/api/agent/browser-runtime/readiness`.

    The agents page draws a readiness chip from this. Without the mirror the
    dashboard would get a 401 (the shared route is api-key-gated) and the
    page would have to guess — and guessing "ready" is exactly the claim a
    browser step cannot honour.
    """
    return await agents_routes.browser_runtime_readiness()


@router.post("/agents/{agent_id}/dry-run", response_model=None)
async def start_dry_run(
    agent_id: str,
    body: DryRunRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    return await agents_routes.start_dry_run(agent_id, body, background_tasks)


@router.post("/agents/{agent_id}/run-once", response_model=None)
async def run_agent_once(agent_id: str, body: AgentRunOnceRequest) -> dict[str, Any]:
    """Same real execution the phone gets from ``/api/agent/agents/{id}/run-once``.

    See ``routes.agents.run_agent_once`` for why this exists and what it
    actually does — this is a thin mirror, not a second implementation.
    """
    return await agents_routes.run_agent_once(agent_id, body)


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


@router.get("/policies/operations", response_model=None)
async def list_policy_operations() -> dict[str, Any]:
    """Operations the permissions form may offer, derived from runtime reality.

    Also reachable at ``GET /api/policies/operations`` for the phone, which
    can write standing rules and therefore needs to be able to look up what
    one means. This mirror stays so the dashboard keeps its single
    ``/api/dashboard/agent`` prefix.
    """
    return policies_routes.list_policy_operations()


@router.get("/policies/step-operations", response_model=None)
async def list_step_operations() -> dict[str, Any]:
    """Which policy operations each workflow step type will need, from the
    runtime's own gating logic (see ``policy/required_operations.py``).

    The readiness rail fetches this once at bootstrap, the same way it
    fetches ``/policies/operations`` and ``/workflow/step-schema``, and
    applies it locally to each agent's ``flow_json`` instead of keeping a
    second, hand-written copy of the mapping in the template.
    """
    return policies_routes.list_step_operations()


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


@router.get("/builder/scripts/proposals/{proposal_id}", response_model=None)
async def get_script_proposal(proposal_id: str) -> dict[str, Any]:
    """Poll a script the builder conversation asked for. Reading is not approving."""
    return await script_proposals_routes.get_script_proposal(proposal_id)


@router.post("/builder/scripts/proposals/{proposal_id}/approve", response_model=None)
async def approve_script_proposal(proposal_id: str) -> dict[str, Any]:
    """Register a proposal the user has read, and return its new script_id.

    Dashboard-only, for the same reason ``POST /scripts`` and
    ``/scripts/save-draft`` are: this is the act that puts a new executable
    where a workflow step can point at it. The phone can hold the
    conversation, see the proposal and read the drafted script; a person on
    the PC decides it may exist. There is deliberately no request body — the
    bytes registered are the bytes the proposal showed."""
    return await script_proposals_routes.approve_script_proposal(proposal_id)


@router.patch("/scripts/{script_id}", response_model=None)
async def update_script(script_id: str, body: ScriptUpdate) -> dict[str, Any]:
    return await scripts_routes.update_script(script_id, body)


@router.delete("/scripts/{script_id}", response_model=None)
async def delete_script(script_id: str) -> dict[str, Any]:
    return await scripts_routes.delete_script(script_id)


# Discovering and importing CLI agent definitions. Same shared-router
# reasoning as routes/cli_agents.py's module docstring (importing is the same
# class of authoring POST /agents already allows an api-key holder to do
# directly) — these are mirrors for the no-key dashboard, not a stricter
# gate. Kept here for the same reason every other shared agent-builder read
# in this file is mirrored: IP-login/local-network trust is a separate
# mechanism from pairing, and the dashboard should not depend on it being on.
#
# Mirrored under *both* sub-prefixes, generated from the shared route table so
# the two cannot drift: `/cli-agents/...` is canonical, `/subagents/...` is the
# pre-rename path a dashboard page served by an older build still requests. A
# missing mirror does not surface as an error here — it renders as an empty
# result, which is how the sweep mirror went unnoticed the first time, so
# test_dashboard_cli_agent_mirrors asserts the pairing directly.

for _sub_prefix in ("/cli-agents", "/subagents"):
    for _method, _path, _handler in cli_agents_routes._ROUTES:
        router.add_api_route(
            f"{_sub_prefix}{_path}",
            _handler,
            methods=[_method],
            response_model=None,
        )


@router.get("/approvals/pending", response_model=None)
async def list_pending_approvals(run_id: str | None = None) -> dict[str, Any]:
    return await approvals_routes.list_pending_approvals(run_id=run_id)


@router.post("/approvals/{approval_id}/decision", response_model=None)
async def decide_approval(approval_id: str, body: ApprovalDecisionCreate) -> Any:
    # Delegating to the api-key route rather than calling `decide_approval`
    # directly is what makes the dashboard resume a parked run too: the
    # `agent.approval_resume` hand-off lives in that handler, and adding a
    # second call here would spawn the same resume twice.
    return await approvals_routes.create_approval_decision(approval_id, body)


__all__ = ["router"]
