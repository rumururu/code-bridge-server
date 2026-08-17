"""Pydantic models for Agent Cockpit APIs."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonContainer = dict[str, Any] | list[Any]


MCPTransport = Literal["stdio", "http", "sse"]
MCPSource = Literal["codex_mcp_list", "gemini_mcp_list", "user_declared"]


class MCPCategory(str, Enum):
    """User-facing grouping for MCP capabilities."""

    BROWSER = "browser"
    FILESYSTEM = "filesystem"
    COMMUNICATION = "communication"
    VCS = "vcs"
    DATA = "data"
    OTHER = "other"


class MCPRiskTier(str, Enum):
    """Risk level used by Agent Builder approval hints."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MCPServer(BaseModel):
    """Normalized MCP server entry exposed to Agent Builder."""

    mcp_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    transport: MCPTransport
    tool_names: list[str] = Field(default_factory=list)
    source: MCPSource
    user_capability: str = ""
    user_examples: list[str] = Field(default_factory=list)
    category: MCPCategory = MCPCategory.OTHER
    risk_tier: MCPRiskTier = MCPRiskTier.MEDIUM


class AgentToolDraft(BaseModel):
    """One MCP tool candidate selected for an agent draft."""

    mcp_id: str = Field(min_length=1, max_length=160)
    tool_names: list[str] = Field(default_factory=list)
    user_capability: str = ""
    user_examples: list[str] = Field(default_factory=list)
    category: MCPCategory = MCPCategory.OTHER
    risk_tier: MCPRiskTier = MCPRiskTier.MEDIUM


class WorkflowStep(BaseModel):
    """Draft workflow step for an agent run.

    ``type`` must list every type in ``workflow_v2.ALLOWED_STEP_TYPES``. The
    two the runtime gained last — ``shell`` and ``notify`` — were missing here
    for as long as they existed, so a Configurator draft block containing the
    shell step the prompt asks it to write failed validation and the whole
    turn's draft was dropped as "invalid draft block". The conversation could
    describe a script step and never produce one. ``test_configurator_script_
    proposal.py`` asserts the two sets still match.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str = Field(min_length=1, max_length=120)
    type: Literal[
        "llm",
        "shell",
        "notify",
        "mcp_tool",
        "browser_action",
        "app_action",
        "android_action",
        "mobile_action",
        "device_action",
        "manual_handoff",
        "approval_gate",
        "condition",
    ] = "llm"
    description: str = Field(default="", max_length=2000)
    tool_hint: str | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)
    success_criteria: str = Field(default="", max_length=2000)
    on_failure: str | dict[str, Any] = Field(default="ask_user")
    # Formal twin of ``on_failure``. The runtime always consulted it
    # (``workflow_v2.normalize_success_policy``, COMMON_STEP_FIELDS) but this
    # model only kept it alive through ``extra="allow"`` — an accident, not a
    # contract. "continue" is the runtime's own default: run the next step.
    on_success: str | dict[str, Any] = Field(default="continue")


class ScriptRequestDraft(BaseModel):
    """A script the Configurator needs but cannot name, because it does not exist.

    A ``shell`` step names a registered script by id, so until someone
    registers one the conversation had nowhere to go: the Configurator could
    only describe the script in prose and the user had to leave, open the
    dashboard, have one written, register it and come back to re-explain. This
    is that description in a shape the server can act on.

    It is a *request*, never an installation: it carries no script body, no
    path and no ``script_id``. What the model writes here is only ever used as
    the intent handed to the script writer, and the result of that is shown to
    the user before it exists anywhere on disk.
    """

    name: str = Field(min_length=1, max_length=80)
    purpose: str = Field(min_length=1, max_length=400)
    expected_output: str = Field(default="", max_length=400)
    # Which step will run it, so approval can fill that step's script_id
    # instead of making the user match them up by hand.
    step_id: str | None = None


class AgentDraft(BaseModel):
    """Incremental Agent Builder draft carried between Configurator turns."""

    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = None
    system_prompt: str = ""
    provider_id: Literal["anthropic", "openai", "google"] | None = None
    model: str | None = None
    tools: list[AgentToolDraft] = Field(default_factory=list)
    flow: list[WorkflowStep] = Field(default_factory=list)
    memory_seeds: list[str] = Field(default_factory=list)
    script_requests: list[ScriptRequestDraft] = Field(default_factory=list)

    @field_validator("memory_seeds", mode="before")
    @classmethod
    def _coerce_memory_seeds(cls, value: Any) -> Any:
        # The Configurator LLM tends to emit memory seeds either as bare
        # strings or as ``{"title": ..., "body": ...}`` dicts. Both shapes
        # carry the same semantic payload; flatten dicts so we keep the
        # storage contract (list[str]) without forcing the prompt to
        # constrain the LLM output shape.
        if not isinstance(value, list):
            return value
        out: list[Any] = []
        for item in value:
            if isinstance(item, dict):
                title = str(item.get("title", "")).strip()
                body = str(item.get("body", "")).strip()
                if title and body:
                    out.append(f"{title}: {body}")
                elif body:
                    out.append(body)
                elif title:
                    out.append(title)
            else:
                out.append(item)
        return out


class TaskDraft(BaseModel):
    """Optional task draft produced after the agent draft is ready."""

    goal: str | None = None
    schedule: str | None = None
    cwd: str | None = None
    workspace_id: str | None = None


class BuilderTurn(BaseModel):
    """Request body for an Agent Builder Configurator turn."""

    session_id: str | None = None
    user_message: str = Field(min_length=1)
    draft: AgentDraft | None = None


class ScriptProposalView(BaseModel):
    """A proposed script as the conversation shows it.

    ``status`` is the whole safety story of this object:

    - ``drafting`` — the script writer is running; there is no body yet.
    - ``ready`` — a body exists **in memory only**. Nothing has been written to
      disk and nothing has been registered, so no workflow step can run it.
    - ``failed`` — drafting failed and ``error`` says why. ``body`` stays null;
      nothing is invented to fill the gap.
    - ``registered`` — a person approved it through the approval endpoint and
      ``script_id`` is the registered script a shell step may now name.

    Nothing moves from ``ready`` to ``registered`` on its own. That transition
    exists only in :func:`routes.script_proposals.approve_script_proposal`,
    which is reachable only from the dashboard — "write me a script" must never
    become "run whatever you just wrote".
    """

    proposal_id: str
    name: str
    purpose: str
    expected_output: str = ""
    step_id: str | None = None
    status: Literal["drafting", "ready", "failed", "registered"]
    interpreter: str = "bash"
    body: str | None = None
    error: str | None = None
    script_id: str | None = None


class BuilderTurnResponse(BaseModel):
    """Response body for an Agent Builder Configurator turn."""

    session_id: str
    assistant_message: str
    updated_draft: AgentDraft
    is_ready_to_commit: bool
    should_offer_task: bool = Field(default=False, deprecated=True)
    task_draft: TaskDraft | None = None
    script_proposals: list[ScriptProposalView] = Field(default_factory=list)
    status: str = "completed"
    job_id: str | None = None
    fallback: bool = False
    error: str | None = None

    @field_validator("should_offer_task", mode="before")
    @classmethod
    def _force_deprecated_task_offer_false(cls, _value: object) -> bool:
        return False


class BuilderCommitRequest(BaseModel):
    """Request body for committing an Agent Builder draft."""

    session_id: str = Field(min_length=1)
    draft: AgentDraft
    also_create_task: bool | TaskDraft = Field(default=False, deprecated=True)
    task_draft: TaskDraft | None = None


class BuilderStepDebugRequest(BaseModel):
    """Request body for running one draft workflow step from Agent Builder."""

    session_id: str | None = None
    draft: AgentDraft
    step_index: int = Field(ge=0)
    execute_llm: bool = True


class AgentCreate(BaseModel):
    """Request body for creating a persistent agent definition."""

    id: str | None = None
    name: str = Field(min_length=1, max_length=80)
    description: str | None = None
    system_prompt: str = ""
    provider_id: Literal["anthropic", "openai", "google"] | None = None
    model: str | None = None
    tools_json: JsonContainer | None = None
    flow_json: JsonContainer | None = None
    policy_overrides_json: JsonContainer | None = None


class AgentUpdate(BaseModel):
    """Request body for partially updating a persistent agent."""

    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = None
    system_prompt: str | None = None
    provider_id: Literal["anthropic", "openai", "google"] | None = None
    model: str | None = None
    tools_json: JsonContainer | None = None
    flow_json: JsonContainer | None = None
    policy_overrides_json: JsonContainer | None = None


class MemoryCreate(BaseModel):
    """Request body for adding one agent memory item."""

    content: str = Field(min_length=1)
    source_run_id: str | None = None
    pinned: bool = False


class MemoryUpdate(BaseModel):
    """Request body for partially updating one agent memory item."""

    content: str | None = Field(default=None, min_length=1)
    pinned: bool | None = None


class DryRunRequest(BaseModel):
    """Request body for simulating an agent's next run."""

    task_id: str | None = None


class AgentRunOnceRequest(BaseModel):
    """Request body for executing an agent's task for real, once, right now.

    Unlike :class:`DryRunRequest` this triggers ``POST
    /agents/{agent_id}/run-once`` — real shell steps, real LLM calls, real
    notifications. ``task_id`` is optional for the same reason it is on the
    dry-run request: most agents have exactly one assigned task, so the route
    resolves it automatically; it only needs to be named when an agent has
    more than one.
    """

    task_id: str | None = None
    provider_id: str | None = None
    model: str | None = None
    cwd: str | None = None
    prompt: str | None = None
    capabilities: list[str] = Field(default_factory=list)


class AgentRunCreate(BaseModel):
    """Request body for creating a durable agent run."""

    project_name: str | None = None
    workspace_id: str | None = None
    agent_id: str | None = None
    provider_id: str | None = None
    model: str | None = None
    title: str | None = None
    goal: str | None = None
    cwd: str | None = None
    parent_run_id: str | None = None
    native_session_id: str | None = None
    task_id: str | None = None


class AgentRunMessageCreate(BaseModel):
    """Request body for adding a message to an agent run."""

    role: str = Field(pattern="^(user|assistant|system|tool)$")
    content: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class AgentEventCreate(BaseModel):
    """Request body for appending a normalized event to an agent run."""

    event_type: str
    provider_id: str | None = None
    provider_event: dict[str, Any] | None = None
    app_event: dict[str, Any] | None = None
    call_id: str | None = None
    parent_event_id: str | None = None


class AgentArtifactCreate(BaseModel):
    """Request body for registering an artifact produced by an agent run."""

    kind: str
    path: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTaskCreate(BaseModel):
    """Request body for creating a tracked agent task."""

    title: str
    description: str | None = None
    project_name: str | None = None
    workspace_id: str | None = None
    run_id: str | None = None
    assigned_agent_id: str | None = None
    kind: str = "general"
    source: str = "manual"
    goal: str | None = None
    priority: int = 0
    due_at: str | None = None
    labels: list[str] = Field(default_factory=list)
    assignee: str | None = None
    requester: str | None = None
    acceptance: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTaskUpdate(BaseModel):
    """Request body for updating a tracked work task."""

    title: str | None = None
    description: str | None = None
    project_name: str | None = None
    workspace_id: str | None = None
    run_id: str | None = None
    assigned_agent_id: str | None = None
    kind: str | None = None
    source: str | None = None
    goal: str | None = None
    status: str | None = None
    priority: int | None = None
    due_at: str | None = None
    labels: list[str] | None = None
    assignee: str | None = None
    requester: str | None = None
    acceptance: list[str] | None = None
    metadata: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class AgentTaskRunLinkCreate(BaseModel):
    """Request body for linking a run to a work task."""

    run_id: str
    role: str = "primary"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTaskStepUpdate(BaseModel):
    """Request body for updating an orchestrated task step."""

    title: str | None = None
    status: str | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    approval_id: str | None = None
    artifact_id: str | None = None


class AgentTaskStepRunCreate(BaseModel):
    """Request body for running one orchestrated task step."""

    input: dict[str, Any] | None = None
    approval_id: str | None = None
    require_approval: bool = False


class AgentTaskStepRespondCreate(BaseModel):
    """Request body for responding to a waiting task step."""

    message: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    remember: bool = False
    resume: bool = False


class BrowserHandoffActionCreate(BaseModel):
    """Request body for applying user-guided browser handoff actions."""

    actions: list[dict[str, Any]] = Field(min_length=1, max_length=10)


class AgentConnectorRequestUpdate(BaseModel):
    """Request body for reviewing or completing connector requests."""

    status: str | None = None
    parameters: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class AgentTaskStartCreate(BaseModel):
    """Request body for starting an orchestrated work task."""

    provider_id: str | None = None
    model: str | None = None
    cwd: str | None = None
    auto_start: bool = True
    dry_run: bool = False
    prompt: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    approval_id: str | None = None


class AgentRunPreflightCreate(BaseModel):
    """Request body for running verification commands for an agent run."""

    project_name: str | None = None
    commands: list[str] = Field(min_length=1, max_length=10)
    timeout: int = Field(default=300, ge=1, le=1800)
    approval_id: str | None = None
