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

from datetime import date
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, Response

from agent.agent_models import (
    AgentRunOnceRequest,
    AgentTaskCreate,
    BuilderTurn,
    DryRunRequest,
)
from approvals.approval_models import ApprovalDecisionCreate, ApprovalRequestCreate
from approvals.approver_channel import CHANNEL_DESKTOP
from agent.script_models import (
    ScriptDraftRequest,
    ScriptDraftSave,
    ScriptRegister,
    ScriptUpdate,
)
from policy.policy_models import PolicyRuleCreate
from agent.browser_action_adapter import (
    get_browser_runtime_readiness,
    reset_browser_readiness_cache,
)
from system.browser_preferences import (
    get_browser_preferences,
    preference_options,
    resolve_browser_launch_plan,
    set_browser_preferences,
)
from system.browser_runtime_install_jobs import (
    BROWSER_INSTALL_DISABLED,
    BrowserRuntimeInstallDisabled,
    get_browser_runtime_install_job_manager,
)

from . import agents as agents_routes
from .agent_browser_rtc import active_dashboard_handoff
# The mirrors below must declare the *route-level* body models, not the shared
# draft models. `agents_routes` reads `body.commit_incomplete`, which only the
# Body subclasses carry; typing a mirror with the plain model made every
# dashboard create/update/commit raise AttributeError at runtime while the
# phone's identical call succeeded, because only the phone reached the real
# route. Tests did not catch it — they exercised `routes/agents.py` directly.
from .agents import AgentCreateBody, AgentUpdateBody, BuilderCommitBody
from . import approvals as approvals_routes
from . import experience as experience_routes
from . import projects as projects_routes
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


class DashboardBrowserHandoffComplete(agents_routes.AgentTaskStepRespondCreate):
    expected_browser_session_id: str = Field(min_length=1)
    expected_run_id: str = Field(min_length=1)


@router.get("/tasks/{task_id}/browser-handoff", response_model=None)
async def get_task_browser_handoff(task_id: str) -> dict[str, Any]:
    payload = active_dashboard_handoff(task_id)
    if payload is None:
        raise HTTPException(status_code=409, detail="Browser handoff is unavailable.")
    return await agents_routes.get_task_browser_handoff(task_id)


@router.post("/tasks/{task_id}/browser-handoff/complete", response_model=None)
async def complete_task_browser_handoff(
    task_id: str,
    body: DashboardBrowserHandoffComplete,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    payload = active_dashboard_handoff(task_id)
    if payload is None or str(payload["browser_session"]["id"]) != body.expected_browser_session_id or str(payload["browser_session"]["run_id"]) != body.expected_run_id:
        raise HTTPException(status_code=409, detail="Browser handoff is unavailable or stale.")
    return await agents_routes.complete_task_browser_handoff(task_id, body, background_tasks)


@router.get("/overview")
def overview() -> dict[str, Any]:
    return experience_routes.overview()


@router.get("/action-items")
def action_items(
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
) -> dict[str, Any]:
    return experience_routes.action_items(limit=limit, cursor=cursor)


@router.get("/history")
def history(
    project_name: str | None = None,
    status: str | None = None,
    since: date | None = None,
    until: date | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
) -> dict[str, Any]:
    return experience_routes.history(project_name=project_name, status=status,
                                     since=since, until=until, limit=limit, cursor=cursor)


@router.get("/runs/{run_id}/summary")
def run_summary(run_id: str) -> dict[str, Any]:
    return experience_routes.summary(run_id)


@router.put("/runs/{run_id}/review")
def review_run(run_id: str, body: experience_routes.ReviewUpdate) -> dict[str, Any]:
    from agent.agent_store import get_agent_store
    from agent import experience_service

    if get_agent_store().get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"review": experience_service.set_review(run_id, body.reviewed, "desktop_owner")}


@router.post("/approvals/{approval_id}/resume")
async def resume_decided_approval(approval_id: str) -> dict[str, Any]:
    return await approvals_routes.resume_decided_approval(approval_id)


@router.get("/runs/{run_id}/checkpoint")
async def get_run_checkpoint(run_id: str) -> dict[str, Any]:
    return await agents_routes.get_run_checkpoint(run_id)


@router.post("/tasks/{task_id}/steps/{step_id}/respond", response_model=None)
async def respond_to_task_step(
    task_id: str,
    step_id: str,
    body: agents_routes.AgentTaskStepRespondCreate,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    return await agents_routes.respond_to_task_step(task_id, step_id, body, background_tasks)


@router.get("/runs/{run_id}/artifacts")
async def list_run_artifacts(run_id: str) -> dict[str, Any]:
    return await agents_routes.list_run_artifacts(run_id)


@router.get("/runs/{run_id}/artifacts/{artifact_id}/content")
async def get_run_artifact_content(
    run_id: str,
    artifact_id: str,
    max_chars: int = Query(default=60000, ge=1, le=200000),
) -> dict[str, Any]:
    return await agents_routes.get_run_artifact_content(run_id, artifact_id, max_chars=max_chars)


@router.get("/projects", response_model=None)
async def list_projects() -> dict[str, Any]:
    return await projects_routes.list_projects()


@router.get("/projects/{name}", response_model=None)
async def get_project(name: str) -> dict[str, Any] | JSONResponse:
    return await projects_routes.get_project(name)


@router.post("/tasks/{task_id}/start", response_model=None)
async def start_task(
    task_id: str,
    body: agents_routes.AgentTaskStartCreate,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    return await agents_routes.start_task(task_id, body, background_tasks)


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


@router.post("/builder/converse/jobs/{job_id}/permission", response_model=None)
async def answer_builder_permission(
    job_id: str, body: agents_routes.BuilderPermissionDecision
) -> dict[str, Any]:
    """Answer a tool call the Configurator asked about.

    Mirrored here for the same reason the rest of the builder is: the
    dashboard talks from localhost with no API key, and this is the surface
    where a person is sitting in front of the conversation that asked."""
    return await agents_routes.answer_builder_permission(job_id, body)


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


@router.post("/agents/{agent_id}/graph/suggest", response_model=None)
async def suggest_agent_graph(agent_id: str, body: "agents_routes.GraphSuggestBody") -> Any:
    """Same proposals the phone gets, behind localhost instead of a key.

    The T-I1-17 door for the desktop canvas: the dashboard listener already
    holds LLM power (`builder/converse` above), so offering suggestions here
    adds no surface a canvas token could leak.
    """
    return await agents_routes.suggest_agent_graph(agent_id, body)


@router.get("/mcp-servers", response_model=None)
async def list_detected_mcp_servers() -> dict[str, Any]:
    """Same catalog the phone gets from `/api/system/mcp-servers/detected`.

    The `mcp-servers` option source's keyless mirror (see
    `workflow_step_schema.OPTION_SOURCES`), so the dashboard's step editor can
    fill the `mcp_tool` server picker without holding an API key.
    """
    from . import system_settings as system_settings_routes

    return await system_settings_routes.list_detected_mcp_servers()


@router.get("/browser-runtime/readiness", response_model=None)
async def browser_runtime_readiness() -> dict[str, Any]:
    """Same probe the phone gets from `/api/agent/browser-runtime/readiness`.

    The agents page draws a readiness chip from this. Without the mirror the
    dashboard would get a 401 (the shared route is api-key-gated) and the
    page would have to guess — and guessing "ready" is exactly the claim a
    browser step cannot honour.
    """
    return await agents_routes.browser_runtime_readiness()


# The three routes below are NOT mirrors of anything in `routes/agents.py`, and
# that is on purpose. Every other endpoint in this file exists because the
# dashboard has no API key; these exist because the *phone* should not have one
# either for this operation. Starting a 200MB download on the host is a local,
# physical-machine decision, and `agents_router` is shared with the
# tunnel-exposed app — a leaked pairing key would otherwise be enough to make a
# stranger's server fetch 450MB on demand. The phone still gets the readiness
# probe and the exact command; only the trigger is localhost-only.
@router.post("/browser-runtime/install", response_model=None)
async def install_browser_runtime() -> JSONResponse:
    """Install the Chromium build `browser_action` steps run on.

    Answers 202 with a job to poll. The job's own re-probe decides whether the
    runtime is usable afterwards — this endpoint never reports readiness on the
    strength of the install having started.
    """
    manager = get_browser_runtime_install_job_manager()
    try:
        job = await manager.start_install()
    except BrowserRuntimeInstallDisabled as exc:
        return JSONResponse(
            status_code=409,
            content={"error": str(exc), "error_code": BROWSER_INSTALL_DISABLED},
        )
    return JSONResponse(status_code=202, content=manager.serialize(job))


@router.get("/browser-runtime/install/jobs/{job_id}", response_model=None)
async def get_browser_runtime_install_job(job_id: str) -> dict[str, Any] | JSONResponse:
    manager = get_browser_runtime_install_job_manager()
    job = manager.get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Install job not found"})
    return manager.serialize(job)


@router.get("/browser-runtime/preferences", response_model=None)
async def get_browser_runtime_preferences() -> dict[str, Any]:
    """The three answers that decide how a browser step runs, and their costs.

    Localhost-only for the same reason the install trigger is: one of the
    options hands the agent every login the person's own Chrome holds. That is
    a decision for someone sitting at the machine, not for whoever holds a
    pairing key. The phone still sees the *result* — the readiness probe
    reports which browser will be used and under which profile.
    """
    preferences = get_browser_preferences()
    plan = resolve_browser_launch_plan(preferences)
    return {
        "preferences": preferences.as_dict(),
        "options": preference_options(),
        "plan": plan.as_dict(),
    }


@router.put("/browser-runtime/preferences", response_model=None)
async def update_browser_runtime_preferences(
    body: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any] | JSONResponse:
    """Store the operator's choice. Unknown values are refused, not rounded off."""
    try:
        preferences = set_browser_preferences(
            browser=body.get("browser"),
            headless_mode=body.get("headless_mode"),
            profile=body.get("profile"),
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # The stored choice changes what the readiness answer *is* (a machine that
    # was "needs a 200MB download" becomes "will use your Chrome" the moment
    # the browser setting changes), so the cached snapshot has to go.
    reset_browser_readiness_cache()
    readiness = await get_browser_runtime_readiness(force_refresh=True)
    return {
        "preferences": preferences.as_dict(),
        "options": preference_options(),
        "plan": resolve_browser_launch_plan(preferences).as_dict(),
        "readiness": readiness,
    }


@router.post("/browser-runtime/install/jobs/{job_id}/cancel", response_model=None)
async def cancel_browser_runtime_install_job(job_id: str) -> dict[str, Any] | JSONResponse:
    manager = get_browser_runtime_install_job_manager()
    job = await manager.cancel_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Install job not found"})
    return manager.serialize(job)


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


@router.post("/runs/{run_id}/resume", response_model=None)
async def resume_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    max_steps: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    """Same resume the phone gets from ``/api/agent/runs/{id}/resume``.

    Mirrored here because the canvas needs it: walking a workflow one step at
    a time is `max_steps=1` on the run, then the same on each continuation,
    and the canvas talks to the dashboard router. A thin mirror, not a second
    implementation.
    """
    return await agents_routes.resume_run(run_id, background_tasks, max_steps=max_steps)


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


@router.post("/devices/permissions/request", response_model=None)
async def request_device_permission_route(body: agents_routes.DevicePermissionRequest) -> dict[str, Any] | JSONResponse:
    return await agents_routes.request_device_permission_route(body)


@router.get("/devices/permissions/kinds", response_model=None)
async def list_device_permission_kinds() -> dict[str, Any]:
    return await agents_routes.list_device_permission_kinds()


@router.get("/agents/{agent_id}/repair-proposals", response_model=None)
async def list_repair_proposals(agent_id: str, limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    return await agents_routes.list_repair_proposals(agent_id, limit=limit)


@router.post("/agents/{agent_id}/repair-proposals/{proposal_id}/accept", response_model=None)
async def accept_repair_proposal(agent_id: str, proposal_id: str, body: agents_routes.RepairProposalDecision | None = None) -> dict[str, Any] | JSONResponse:
    return await agents_routes.accept_repair_proposal(agent_id, proposal_id, body)


@router.post("/agents/{agent_id}/repair-proposals/{proposal_id}/reject", response_model=None)
async def reject_repair_proposal(agent_id: str, proposal_id: str, body: agents_routes.RepairProposalDecision | None = None) -> dict[str, Any]:
    return await agents_routes.reject_repair_proposal(agent_id, proposal_id, body)


@router.get("/runs/{run_id}/steps", response_model=None)
async def list_run_steps(run_id: str) -> dict[str, Any]:
    """The steps of one run — what the run detail pane is built from.

    The pane used to read ``/tasks/{task_id}/steps`` instead. Every run of an
    agent shares that agent's single task, so each run opened the whole
    history: one click on a two-step run rendered 386 steps from 193 runs.
    """
    return await agents_routes.list_run_steps(run_id)


@router.get("/tasks/{task_id}/steps", response_model=None)
async def list_task_steps(task_id: str) -> dict[str, Any]:
    """Steps with their outputs — where a shell step's exit code and captured
    stdout/stderr live. This is what turns "the run failed" into something you
    can act on without opening a terminal."""
    return await agents_routes.list_task_steps(task_id)


@router.get("/runs/{run_id}/events", response_model=None)
async def list_run_events(
    run_id: str,
    after_sequence: int | None = Query(default=None, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    """Events for a run — where ``task.step.mcp_tools`` lives: the record of
    which declared MCP servers a step's turn was actually given, and which
    were missing on this machine. That report is an event, not a step output,
    so the steps mirror above cannot surface it; without this read path the
    dashboard would restore exactly the silent drop the event exists to
    prevent (see ``_apply_declared_mcp_servers`` in agent/task_orchestrator.py).
    """
    return await agents_routes.list_run_events(
        run_id,
        after_sequence=after_sequence,
        limit=limit,
    )


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


# Local agents that are not LLM runs — the feedback agent is the first — file
# their approval requests here rather than on the api-key listener: they run
# on this machine with no paired-device key, and the dashboard listener is the
# one bound to localhost. Going through `request_approval_for_operation` is
# what puts them under the policy engine: a standing rule for their operation
# can now actually match, and the audit log sees the request.
@router.post("/approvals/request", response_model=None)
async def request_approval(body: ApprovalRequestCreate) -> Any:
    return await approvals_routes.request_approval(body)


@router.get("/approvals/pending", response_model=None)
async def list_pending_approvals(run_id: str | None = None) -> dict[str, Any]:
    return await approvals_routes.list_pending_approvals(run_id=run_id)


# After `/approvals/pending`: a path parameter declared earlier would swallow it.
@router.get("/approvals/{approval_id}", response_model=None)
async def get_approval(approval_id: str) -> dict[str, Any]:
    return approvals_routes.get_approval_with_decision(approval_id)


@router.post("/approvals/{approval_id}/decision", response_model=None)
async def decide_approval(approval_id: str, body: ApprovalDecisionCreate) -> Any:
    # Delegating to the shared body in routes/approvals.py rather than calling
    # `approvals.approval_service.decide_approval` directly is what makes the
    # dashboard resume a parked run too: the `agent.approval_resume` hand-off
    # lives there, and adding a second call here would spawn the same resume
    # twice.
    #
    # This is the *only* route allowed to answer a `desktop_only` approval, and
    # the claim rests entirely on where this router lives: `require_local_access`
    # on the router above, plus registration in `_DASHBOARD_ONLY_ROUTERS`
    # (routes/__init__.py) so it is never mounted on the tunnel-exposed API app.
    # Reaching this handler therefore means the request arrived on the
    # localhost-bound dashboard listener — someone at the machine. Nothing in
    # `body` is consulted for that; `body.approver` is audit metadata only.
    return await approvals_routes.apply_approval_decision(
        approval_id, body, channel=CHANNEL_DESKTOP
    )


__all__ = ["router"]
