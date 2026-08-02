"""Stateful Agent Builder Configurator helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from agent.agent_models import (
    AgentDraft,
    AgentToolDraft,
    MCPCategory,
    MCPRiskTier,
    TaskDraft,
    WorkflowStep,
)

logger = logging.getLogger(__name__)

BUILDER_SESSION_IDLE_TTL = timedelta(minutes=30)

# Placeholder inside _SYSTEM_PROMPT_TEMPLATE that _workflow_step_schema_block()
# fills in. A plain .replace(), not str.format(): the template is full of
# literal `{` / `}` (every JSON-shaped example in the prompt), so .format()
# would need every one of those escaped and would break the moment someone
# adds another example without remembering to double the braces.
_WORKFLOW_STEP_SCHEMA_MARKER = "{{WORKFLOW_STEP_SCHEMA_BLOCK}}"

_DRAFT_BLOCK_RE = re.compile(
    r"```[ \t]*(draft|task_draft)[^\n]*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_READY_LINE_RE = re.compile(r"^[ \t]*READY_TO_COMMIT[ \t]*$", re.MULTILINE)

_SYSTEM_PROMPT_TEMPLATE = """You are the Agent Builder Configurator for Code Bridge.

Your job: have a conversation with a user who wants to create a new
"Agent" — a persistent autonomous worker — and produce a structured
AgentDraft JSON they can commit.

You DO NOT execute anything. You only ask clarifying questions and
maintain a draft.

Schema you must fill out:

AgentDraft {
  name              // short human label (1~80 chars)
  description       // 1~2 sentence summary
  system_prompt     // the role/instructions the agent will live by
  provider_id       // "anthropic" | "openai" | "google" (which CLI runs it)
  model             // optional model id
  tools             // list of { mcp_id, tool_names } from the catalog
  flow              // list of WorkflowStep
  memory_seeds      // optional initial memory items
  script_requests   // list of ScriptRequest — scripts you need that are
                    // not registered yet (see rule 2d)
}

ScriptRequest {
  name              // short script name, e.g. "Check disk usage"
  purpose           // one line: what the script does, in the imperative
  expected_output   // what it prints on success, e.g. "USED_PCT=87"
  step_id           // id of the shell step that will run it once approved
}

{{WORKFLOW_STEP_SCHEMA_BLOCK}}

Rules:

1. Always respond in two parts:
   - A natural-language reply for the user (Korean unless user writes in
     English).
   - An updated AgentDraft JSON inside a fenced ```draft``` block. The
     server consumes that block and the user does not see it.

2. Drive the conversation field by field. Ask the smallest useful
   question per turn. Do not dump a giant questionnaire.

2a. Prefer a saveable draft over a perfect questionnaire. When the user's
    intent is reasonably clear, fill all required UI fields immediately:
    `name`, `description`, `system_prompt`, `provider_id`, `tools`, and
    at least one concrete workflow step. Never leave `system_prompt`
    empty. If details are missing, choose a conservative placeholder and
    ask a follow-up in the natural-language reply while still returning a
    saveable draft.

2b. Do not require the user to know schema names, MCP ids, or tool action
    names. Translate plain-language requests into the draft JSON yourself.
    For example, "매시간 품앗이 요청 확인 후 참여하고 앱 설치/실행 확인" should
    become a scheduled Android app automation workflow, not a request for
    more schema details.

2c. Pick the cheapest step type that can do the work.

    - A fixed command with a knowable answer — disk usage, a health check,
      a build, an rsync — is a `shell` step running a REGISTERED script.
      It costs no tokens, needs no approval, and gives the same answer
      every time. An `llm` step that only runs `df -h` wakes a model every
      time the schedule fires to read one number.
    - Judgement about that answer is a SECOND `llm` step after it. The
      shell step's exit code and output are handed to it as evidence, so
      "check the disk, and tell me if it is nearly full" is two steps:
      shell reads, llm decides.
    - `notify` is how the agent tells the user something. It is the
      default way to report: a scheduled run finishes while nobody is
      watching, and anything only printed is lost.

2d. Shell steps name a registered script by `script_id`. The list of
    registered scripts is given to you below; use one whose description
    matches.

    If none does, add an entry to `script_requests` describing the script
    you need: its name, a one-line purpose, what it should print on
    success, and the `step_id` of the shell step that will run it. The
    server has that script written and shows the text to the user; nothing
    is saved or installed until they approve it. Say so in your reply —
    e.g. "필요한 스크립트 초안을 만들고 있어요. 확인하고 승인하시면 이 단계에
    연결됩니다." — and carry on with the rest of the draft.

    Until a script is approved it has no id, so do NOT add its shell step
    to `flow` yet, and never invent a `script_id`. Keep the step out until
    the registered-scripts list below contains it. Never put a command line
    in the draft: a workflow step names a script, it never carries one.

    Once the server tells you a script was registered, use that id on the
    step and drop the matching entry from `script_requests`.

3. When the user mentions tools or external systems, capture them in
   the agent draft's `tools` array using a stable mcp_id (e.g.
   `playwright`, `slack`, `github`). The user installs the MCP in
   their provider CLI themselves (e.g. `codex mcp add playwright -- npx
   @playwright/mcp@latest`); Code Bridge does not manage installation.

3a. Any agent that is supposed to tell the user something ends with a
    `notify` step. Do not write "sends an alert" into system_prompt and
    leave it there — that is a sentence, not a delivery. The app inbox is
    the default and needs no setup; only reach for an external channel
    (Slack, email) if the user asks for one by name, and say plainly that
    it needs an MCP they install themselves.

4. When the user mentions repetition or schedule ("every 2 hours",
   "nightly"), do NOT put it in the AgentDraft. Keep timing in the
   `task_draft` stream described in rule 6.

5. When all required fields (name, system_prompt, provider_id) are
   filled AND the workflow has at least one step AND the user
   indicates readiness ("좋아", "시작해줘", "OK", "go" 등), set
   `is_ready_to_commit: true` in your response signal (server-side
   field — you express this by writing "READY_TO_COMMIT" on its
   own line above the ```draft``` block).

6. During the conversation, AFTER you've gathered the agent's role and
   recommended tools, ask the user about timing in plain language —
   e.g. "이걸 일정에 따라 자동으로 돌릴까요, 아니면 제가 부를 때만 돌릴까요?
   (예: '매 2시간', '매일 아침', '필요할 때만')"

   If the user names a schedule ("매 2시간", "every morning", "nightly",
   "매일 09:30" 등):
     - populate `task_draft.goal` from the agent's role summary
     - populate `task_draft.schedule` with the canonical expression
       ("every 2h", "daily 09:30" etc.)
     - acknowledge: "매 2시간마다 자동으로 돌리도록 일정 잡았어요."
     - NEVER use the word "Task" in your user-facing reply. Use "일정"
       or "자동 실행" instead.

   If the user says one-off / "필요할 때만" / "수동":
     - leave `task_draft` absent or empty
     - acknowledge: "그럼 부르실 때 한 번씩 돌리는 걸로 해둘게요."

   Either way, continue toward READY_TO_COMMIT once workflow and
   required fields are filled.

   Do NOT ask a separate follow-up about registering an internal work item.
   The schedule question above is the single decision point.

8. Keep system_prompt suggestion concise (under 800 chars). The
   Agent will receive memory items at run time, so do not preload
   it with every possible instruction.

9. If the user contradicts an earlier choice, update the draft
   silently and acknowledge the change in your reply.

10. When recommending tools, refer to them by their MCP package name
    (e.g. "Playwright", "Slack", "Filesystem"). Briefly describe what
    each tool does so the user can recognize whether it fits.

11. Before suggesting any tool, first confirm the user's intended
    actions in plain language. Ask: "이 봇이 해야 할 일에 (A)웹사이트 자동
    조작 (B)파일 읽기/수정 (C)외부 알림 (D)데이터 정리 중 어떤 게 들어가나요?"
    Only after the user picks 1-2 categories should you suggest matching
    MCPs from that category.

12. For tools that perform destructive or sensitive actions (filesystem
    writes, external messaging, billing actions, account management),
    ALWAYS note that the user can require per-run approval. Example:
    "Filesystem 쓰기 권한은 매번 실행 전에 확인 받도록 해두겠습니다."

13. For browser automation that may hit login, captcha, account challenge,
    or bot detection, do NOT imply bypassing it. Make that workflow step a
    `browser_action` and set `on_failure` to ask the user or manual handoff,
    resuming the same step after the user completes the action.

14. For external messaging workflows such as email, chat, DM, or Naver
    Note, always require a concrete recipient, message body, and an
    explicit approval step before the final send/submit action. If any of
    recipient/body/approval is missing, ask for the missing item instead
    of marking the draft ready.

15. When you see one or more lines like
    `[User edited <field> directly: <value>]`
    at the start of the user's message, ALWAYS acknowledge them on the
    first line of your reply in plain Korean. Examples:
    - "방금 수정하신 name 반영했어요."
    - "직접 바꾸신 system_prompt 기준으로 이어갈게요."
    Then continue with whatever follow-up question you would have asked
    anyway. This signal must NEVER appear inside the ```draft``` block —
    it is for the user-facing reply only.

"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _registered_scripts_block() -> str:
    """List the scripts a shell step may name.

    Without this the Configurator is told shell steps exist and given no way
    to reference one, so it either invents a script_id — which fails at
    execution, far from the conversation that produced it — or falls back to
    an llm step that re-derives a fixed command on every scheduled run.
    """
    try:
        from agent.script_store import get_script_store

        scripts = get_script_store().list_scripts(limit=50)
    except (ImportError, RuntimeError, OSError, sqlite3.Error):
        # Building a prompt must not depend on the script table existing. A
        # migration mid-flight, or a test database that never ran one, would
        # otherwise take down the whole conversation.
        return ""

    if not scripts:
        return (
            "\nRegistered scripts: none yet.\n"
            "No shell step can be written until one is registered. If the work "
            "is a fixed command, put it in `script_requests` (rule 2d) so the "
            "server drafts it for the user to approve, and use an llm step "
            "only as a stopgap.\n"
        )

    lines = ["\nRegistered scripts (use these ids for shell steps):"]
    for script in scripts:
        description = (script.get("description") or "").strip()
        suffix = f" — {description}" if description else ""
        lines.append(f"  - {script['id']}: {script['name']}{suffix}")
    lines.append("")
    return "\n".join(lines)


def _workflow_step_schema_block() -> str:
    """Generate the ``WorkflowStep { ... }`` listing from the published schema.

    Reuses :mod:`agent.workflow_step_schema` — built from the same
    ``WORKFLOW_STEP_SCHEMA`` ``normalize_workflow_step`` enforces — instead of
    the hand-written field list this replaced (AGENT_COMPOSITION_SPEC.md
    Phase 2.3). That list used to offer every field on every type regardless
    of whether the normalizer would accept it there (e.g. ``device_id`` on an
    ``llm`` step), so a draft the Configurator wrote in good faith could be
    silently stripped at commit. Generating it means a field the server
    rejects for a type can never appear in that type's listing here, and a
    field the server *does* accept can never be missing from it.
    """

    from agent.workflow_step_schema import base_field_types, field_help_en
    from agent.workflow_v2 import ALLOWED_STEP_TYPES, NOTIFY_LEVELS

    field_types = base_field_types()
    type_options = " | ".join(f'"{t}"' for t in sorted(ALLOWED_STEP_TYPES))

    def scope_note(types_for_field: list[str]) -> str:
        if set(types_for_field) == set(ALLOWED_STEP_TYPES):
            return "all step types"
        return ", ".join(types_for_field) + " steps only"

    lines = [
        "WorkflowStep {",
        '  id                // stable machine id, e.g. "login_check"',
        f"  type              // {type_options}",
    ]
    for field_key in sorted(field_types):
        types_for_field = field_types[field_key]
        label = field_key.ljust(17)
        if field_key == "notify":
            level_options = " | ".join(f'"{level}"' for level in NOTIFY_LEVELS)
            lines.append(
                f"  {label} // {scope_note(types_for_field)}: "
                f"{{ title, body, level }} where level is {level_options}"
            )
            continue
        lines.append(f"  {label} // {scope_note(types_for_field)}: {field_help_en(field_key)}")
    lines.append('  name              // short label (e.g. "Login")')
    lines.append("  description       // legacy summary; keep for compatibility")
    lines.append(
        "  on_failure        // prefer structured object:\n"
        '                    // {type:"retry", max_attempts:1, then:{type:"ask_user", resume:"same_step"}}\n'
        '                    // {type:"ask_user", resume:"same_step", prompt:"..."}\n'
        '                    // {type:"manual_handoff", resume:"same_step", prompt:"..."}\n'
        '                    // {type:"goto_step", target_step_id:"..."} or {type:"abort"}'
    )
    lines.append("}")
    return "\n".join(lines)


def build_configurator_system_prompt(_unused: list[Any] | None = None) -> str:
    """Return the Configurator system prompt.

    The legacy MCP catalog injection was removed: the Configurator now
    accepts whatever MCP name the user supplies and writes it into the
    agent draft. Provider CLIs handle the actual MCP runtime; the user
    installs MCPs via their own CLI (e.g. ``codex mcp add ...``).

    Registered scripts are injected because they are the one thing the
    Configurator cannot invent: a shell step names a script by id, and an id
    it made up fails at execution rather than in the conversation.
    """

    template = _SYSTEM_PROMPT_TEMPLATE.replace(
        _WORKFLOW_STEP_SCHEMA_MARKER, _workflow_step_schema_block()
    )
    return template + _registered_scripts_block()


@dataclass(frozen=True)
class ParsedConfiguratorResponse:
    """Parsed Configurator response content."""

    assistant_message: str
    draft: AgentDraft | None
    task_draft: TaskDraft | None
    is_ready_to_commit: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class BuilderSession:
    """In-memory Agent Builder conversation session."""

    system_prompt: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=_utcnow)
    last_seen_at: datetime = field(default_factory=_utcnow)
    messages: list[dict[str, str]] = field(default_factory=list)
    current_draft: AgentDraft = field(default_factory=AgentDraft)
    task_draft: TaskDraft | None = None
    is_ready_to_commit: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def is_expired(self, now: datetime | None = None) -> bool:
        reference = now or _utcnow()
        return reference - self.last_seen_at > BUILDER_SESSION_IDLE_TTL

    def touch(self) -> None:
        self.last_seen_at = _utcnow()

    def set_client_draft(self, draft: AgentDraft | None) -> None:
        if draft is not None:
            self.current_draft = draft

    def append_user_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    def apply_registered_script(
        self,
        *,
        script: dict[str, Any],
        step_id: str | None,
        request_name: str,
    ) -> AgentDraft:
        """Tell a live conversation about a script that now exists.

        The registered-scripts list is baked into ``system_prompt`` when the
        session is created (see :func:`build_configurator_system_prompt`), so a
        script approved mid-conversation would otherwise be invisible to every
        later turn: the model would keep saying it needs the script it just
        got, and keep being unable to name it. Appending the id here is what
        makes approval actually close the loop rather than merely file a row.

        The step's ``script_id`` is filled in the same act, because the whole
        point of ``step_id`` on the request is that the user should not have to
        copy an id from one panel into another.
        """
        script_id = str(script.get("id") or "").strip()
        if not script_id:
            return self.current_draft

        flow = self.current_draft.flow
        if step_id:
            flow = [
                step.model_copy(update={"script_id": script_id})
                if (step.id or "").strip() == step_id.strip()
                else step
                for step in flow
            ]
        remaining = [
            request
            for request in self.current_draft.script_requests
            if request.name.strip().casefold() != request_name.strip().casefold()
        ]
        self.current_draft = self.current_draft.model_copy(
            update={"flow": flow, "script_requests": remaining}
        )

        name = str(script.get("name") or request_name).strip()
        description = str(script.get("description") or "").strip()
        suffix = f" — {description}" if description else ""
        line = f"  - {script_id}: {name}{suffix}"
        step_note = f" Use it as the script_id of step '{step_id}'." if step_id else ""
        self.system_prompt = (
            f"{self.system_prompt}\n"
            f"\nThe user approved a new script, now registered:\n{line}\n"
            f"It is available for shell steps.{step_note}\n"
        )
        self.touch()
        return self.current_draft

    def apply_llm_response(
        self,
        raw_response: str,
        *,
        user_message: str,
    ) -> ParsedConfiguratorResponse:
        before = self.current_draft
        parsed = parse_configurator_response(raw_response)
        if parsed.draft is not None:
            self.current_draft = _preserve_flow_for_additive_request(
                before,
                parsed.draft,
                user_message=user_message,
            )
        if parsed.task_draft is not None:
            self.task_draft = parsed.task_draft
            if not _task_draft_has_content(parsed.task_draft):
                self.task_draft = None
        elif _looks_like_manual_timing(user_message):
            self.task_draft = None
        self.current_draft, self.task_draft = enrich_draft_from_user_intent(
            self.current_draft,
            previous_draft=before,
            task_draft=self.task_draft,
            user_message=user_message,
        )
        if parsed.is_ready_to_commit:
            self.is_ready_to_commit = True
        self.messages.append({"role": "assistant", "content": parsed.assistant_message})
        logger.warning(
            "builder_configurator_turn session_id=%s user_msg_hash=%s draft_delta=%s",
            self.session_id,
            hash_user_message(user_message),
            summarize_draft_delta(before, self.current_draft),
        )
        return parsed


BUILDER_SESSIONS: dict[str, BuilderSession] = {}


def create_builder_session(
    *,
    system_prompt: str,
    draft: AgentDraft | None = None,
) -> BuilderSession:
    session = BuilderSession(
        system_prompt=system_prompt,
        current_draft=draft or AgentDraft(),
    )
    BUILDER_SESSIONS[session.session_id] = session
    return session


def get_builder_session(session_id: str | None) -> BuilderSession | None:
    if not session_id:
        return None
    clear_expired_builder_sessions()
    session = BUILDER_SESSIONS.get(session_id)
    if session is not None and session.is_expired():
        BUILDER_SESSIONS.pop(session_id, None)
        return None
    return session


def delete_builder_session(session_id: str) -> None:
    BUILDER_SESSIONS.pop(session_id, None)


def clear_expired_builder_sessions(now: datetime | None = None) -> None:
    reference = now or _utcnow()
    expired = [
        session_id
        for session_id, session in BUILDER_SESSIONS.items()
        if session.is_expired(reference)
    ]
    for session_id in expired:
        BUILDER_SESSIONS.pop(session_id, None)


def build_configurator_turn_prompt(session: BuilderSession) -> str:
    """Build a provider message from the persistent Configurator state."""

    history_lines = []
    for message in session.messages:
        role = message.get("role", "user")
        if role == "system":
            continue
        content = message.get("content", "")
        history_lines.append(f"{role}: {content}")
    draft_json = json.dumps(
        session.current_draft.model_dump(),
        ensure_ascii=False,
        indent=2,
    )
    history = "\n".join(history_lines).strip() or "(no prior turns)"
    return (
        f"{session.system_prompt}\n\n"
        "Current server-side AgentDraft:\n"
        f"```json\n{draft_json}\n```\n\n"
        "Conversation history:\n"
        f"{history}\n\n"
        "Continue the conversation according to the rules above."
    )


def parse_configurator_response(response_text: str) -> ParsedConfiguratorResponse:
    """Extract natural-language text, draft JSON, and response signals."""

    blocks: list[tuple[str, str]] = []

    def _collect_block(match: re.Match[str]) -> str:
        blocks.append((match.group(1).lower(), match.group(2)))
        return "\n"

    without_blocks = _DRAFT_BLOCK_RE.sub(_collect_block, response_text)
    is_ready = bool(_READY_LINE_RE.search(without_blocks)) or bool(
        _READY_LINE_RE.search(response_text)
    )
    assistant_message = _READY_LINE_RE.sub("", without_blocks)
    assistant_message = _collapse_blank_lines(assistant_message).strip()

    draft: AgentDraft | None = None
    task_draft: TaskDraft | None = None
    warnings: list[str] = []
    saw_draft_block = False

    for block_type, block_body in blocks:
        parsed_json = _parse_json_block(block_type, block_body, warnings)
        if parsed_json is None:
            continue
        if block_type == "draft":
            saw_draft_block = True
            try:
                draft = AgentDraft.model_validate(parsed_json)
            except ValidationError as exc:
                warning = f"invalid draft block: {exc}"
                warnings.append(warning)
                logger.warning("builder_configurator_invalid_draft_block error=%s", exc)
        elif block_type == "task_draft":
            try:
                task_draft = TaskDraft.model_validate(parsed_json)
            except ValidationError as exc:
                warning = f"invalid task_draft block: {exc}"
                warnings.append(warning)
                logger.warning("builder_configurator_invalid_task_draft_block error=%s", exc)

    if not saw_draft_block:
        warnings.append("missing draft block")
        logger.warning("builder_configurator_missing_draft_block")

    return ParsedConfiguratorResponse(
        assistant_message=assistant_message,
        draft=draft,
        task_draft=task_draft,
        is_ready_to_commit=is_ready,
        warnings=warnings,
    )


def _parse_json_block(
    block_type: str,
    block_body: str,
    warnings: list[str],
) -> Any | None:
    try:
        return json.loads(block_body.strip())
    except json.JSONDecodeError as exc:
        warning = f"invalid {block_type} JSON: {exc}"
        warnings.append(warning)
        logger.warning(
            "builder_configurator_invalid_json block=%s error=%s",
            block_type,
            exc,
        )
        return None


def _task_draft_has_content(task_draft: TaskDraft) -> bool:
    return any(
        isinstance(value, str) and value.strip()
        for value in (
            task_draft.goal,
            task_draft.schedule,
            task_draft.cwd,
            task_draft.workspace_id,
        )
    )


def enrich_draft_from_user_intent(
    draft: AgentDraft,
    *,
    previous_draft: AgentDraft | None = None,
    task_draft: TaskDraft | None,
    user_message: str,
) -> tuple[AgentDraft, TaskDraft | None]:
    """Patch common Configurator omissions without pretending MCP is installed.

    The Configurator prompt tells the LLM to declare required MCPs, workflow
    tool hints, and schedules. In practice the model can still return a
    plausible-looking agent with empty ``tools``/``flow``/``task_draft``.
    This deterministic pass keeps the builder useful for obvious intents while
    preserving the user's responsibility to install MCPs in their CLI.
    """

    raw_text = _raw_intent_text(user_message, draft)
    text = raw_text.casefold()
    is_naver_note = _looks_like_naver_note(text)
    prefers_app_automation = _prefers_app_automation(text)
    is_web = _looks_like_web_automation(text) and not prefers_app_automation
    is_naver_cafe = _looks_like_naver_cafe(text)
    schedule = _extract_schedule(text)
    next_draft = draft
    next_task_draft = task_draft

    next_draft = _ensure_basic_save_fields(
        next_draft,
        previous_draft=previous_draft,
        user_message=user_message,
    )

    if is_web:
        next_draft = _ensure_playwright_tool(next_draft)
        plan_kind = (
            "naver_note"
            if is_naver_note
            else "naver_cafe"
            if is_naver_cafe
            else "generic_web"
        )
        if _flow_needs_operational_plan(next_draft.flow, plan_kind=plan_kind):
            next_draft = next_draft.model_copy(
                update={
                    "flow": _naver_note_flow(text)
                    if is_naver_note
                    else _naver_cafe_flow()
                    if is_naver_cafe
                    else _generic_web_automation_flow(
                        target_url=_extract_url(raw_text),
                        required_text=_extract_required_visible_text(raw_text),
                    )
                }
            )
        else:
            next_draft = next_draft.model_copy(
                update={"flow": _add_playwright_hints(next_draft.flow)}
            )

    if not is_web:
        next_draft = _ensure_requested_generic_capabilities(next_draft, user_message)
    next_draft = _ensure_android_review_exchange_template(next_draft, text)
    next_draft = _normalize_app_automation_steps(next_draft, user_message)
    next_draft = _repair_empty_workflow_actions(next_draft, text)

    if schedule is not None and not _looks_like_manual_timing(text):
        goal = _task_goal_from_draft(next_draft)
        if next_task_draft is None or not _task_draft_has_content(next_task_draft):
            next_task_draft = TaskDraft(goal=goal, schedule=schedule)
        else:
            existing_goal = next_task_draft.goal
            previous_auto_goal = (
                _task_goal_from_draft(previous_draft) if previous_draft is not None else None
            )
            should_replace_goal = (
                not isinstance(existing_goal, str)
                or not existing_goal.strip()
                or (
                    isinstance(previous_auto_goal, str)
                    and existing_goal.strip() == previous_auto_goal.strip()
                )
            )
            next_task_draft = next_task_draft.model_copy(
                update={
                    "goal": goal if should_replace_goal else next_task_draft.goal,
                    "schedule": next_task_draft.schedule or schedule,
                }
            )

    return next_draft, next_task_draft


def _ensure_basic_save_fields(
    draft: AgentDraft,
    *,
    previous_draft: AgentDraft | None,
    user_message: str,
) -> AgentDraft:
    updates: dict[str, Any] = {}
    text = _intent_text(user_message, draft)

    if not (draft.name or "").strip():
        updates["name"] = _fallback_agent_name(text)
    if not (draft.description or "").strip():
        updates["description"] = _fallback_agent_description(text)
    if not draft.system_prompt.strip():
        updates["system_prompt"] = _fallback_system_prompt(text)
    if draft.provider_id is None:
        provider_id = _infer_provider_id(user_message)
        if provider_id is None and previous_draft is not None:
            provider_id = previous_draft.provider_id
        updates["provider_id"] = provider_id or "openai"

    if not updates:
        return draft
    return draft.model_copy(update=updates)


def _fallback_agent_name(text: str) -> str:
    if _looks_like_android_review_exchange(text):
        return "Android Review Exchange Agent"
    if _looks_like_naver_note(text):
        return "Naver Note Agent"
    if _looks_like_naver_cafe(text):
        return "Naver Cafe Agent"
    if _prefers_app_automation(text):
        return "Android App Agent"
    if _looks_like_web_automation(text):
        return "Web Automation Agent"
    return "Custom Agent"


def _fallback_agent_description(text: str) -> str:
    if _looks_like_android_review_exchange(text):
        return "Handles Android review exchange requests with duplicate checks, approval gates, app install, launch verification, and result recording."
    if _looks_like_naver_note(text):
        return "Prepares and sends Naver Note messages with login handoff, send approval, and result recording."
    if _looks_like_naver_cafe(text):
        return "Automates safe Naver Cafe participation workflows with duplicate checks and user handoff for login or captcha."
    if _prefers_app_automation(text):
        return "Runs Android app automation with user approval for sensitive actions and result reporting."
    if _looks_like_web_automation(text):
        return "Runs a browser workflow with target confirmation, safe interaction, and result reporting."
    return "Runs the requested workflow, asks for missing details, and reports the result."


def _fallback_system_prompt(text: str) -> str:
    if _looks_like_android_review_exchange(text):
        return (
            "Run Android review exchange requests using app/device automation only. "
            "Confirm the target request, avoid duplicate submissions using memory, "
            "require approval before joining requests or installing apps, stop for "
            "login/captcha/account challenges, verify app launch, and record results."
        )
    if _looks_like_naver_note(text):
        return (
            "Prepare Naver Note sends safely. Confirm recipient and body, do not bypass "
            "login/captcha/account challenges, require explicit approval before sending, "
            "and record send results and duplicate-prevention memory."
        )
    if _looks_like_naver_cafe(text):
        return (
            "Run Naver Cafe participation safely. Check login and access state, avoid "
            "captcha or policy bypass, prevent duplicate applications using memory, "
            "ask before uncertain external actions, and report each result."
        )
    if _prefers_app_automation(text):
        return (
            "Run the requested Android app workflow safely. Confirm targets, require "
            "approval before installs or submissions, stop for account/device prompts, "
            "verify visible results, and record outcomes."
        )
    if _looks_like_web_automation(text):
        return (
            "Run the requested browser workflow safely. Confirm target URL and inputs, "
            "stop for login/captcha/account challenges, ask before sensitive submissions, "
            "verify visible results, and summarize outcomes."
        )
    return (
        "Run the requested workflow safely. Ask for missing required details, require "
        "approval before sensitive external actions, stop for manual handoff when needed, "
        "and report concise results."
    )


def _infer_provider_id(user_message: str) -> str | None:
    text = user_message.casefold()
    if _has_any(text, ("anthropic", "claude")):
        return "anthropic"
    if _has_any(text, ("google", "gemini")):
        return "google"
    if _has_any(text, ("openai", "codex", "gpt")):
        return "openai"
    return None


def _preserve_flow_for_additive_request(
    previous: AgentDraft,
    parsed: AgentDraft,
    *,
    user_message: str,
) -> AgentDraft:
    """Keep established workflow steps when the user asks to add capability."""

    if (
        not previous.flow
        or not parsed.flow
        or not _looks_like_additive_workflow_request(user_message)
    ):
        return parsed
    merged = _append_unique_steps(previous.flow, parsed.flow)
    return parsed.model_copy(update={"flow": merged})


def _looks_like_additive_workflow_request(user_message: str) -> bool:
    text = user_message.casefold()
    if any(
        marker in text
        for marker in (
            "replace",
            "rewrite",
            "regenerate",
            "start over",
            "remove",
            "delete",
            "instead",
            "교체",
            "다시 만들어",
            "새로 만들어",
            "삭제",
            "제거",
            "대신",
        )
    ):
        return False
    return any(
        marker in text
        for marker in (
            "add",
            "also",
            "include",
            "append",
            "plus",
            "step",
            "capability",
            "추가",
            "넣어",
            "포함",
            "단계",
            "기능",
        )
    )


def _append_unique_steps(
    existing: list[WorkflowStep],
    additions: list[WorkflowStep],
) -> list[WorkflowStep]:
    out = list(existing)
    seen = {_step_identity(step) for step in out}
    for step in additions:
        identity = _step_identity(step)
        if identity in seen:
            continue
        out.append(step)
        seen.add(identity)
    return out


def _step_identity(step: WorkflowStep) -> str:
    if step.id and step.id.strip():
        return f"id:{step.id.strip().casefold()}"
    return f"text:{step.type}:{step.name.strip().casefold()}:{step.description.strip().casefold()}"


def _intent_text(user_message: str, draft: AgentDraft) -> str:
    return _raw_intent_text(user_message, draft).casefold()


def _raw_intent_text(user_message: str, draft: AgentDraft) -> str:
    parts = [
        user_message,
        draft.name or "",
        draft.description or "",
        draft.system_prompt or "",
    ]
    for step in draft.flow:
        parts.extend([step.name, step.description, step.tool_hint or ""])
    for tool in draft.tools:
        parts.extend([tool.mcp_id, tool.user_capability, " ".join(tool.user_examples)])
    return "\n".join(part for part in parts if part)


def _looks_like_web_automation(text: str) -> bool:
    if _extract_url(text):
        return True
    return any(
        marker in text
        for marker in (
            "playwright",
            "browser",
            "website",
            "web site",
            "webpage",
            "web page",
            "url",
            "click",
            "form",
            "login",
            "captcha",
            "play store",
            "google play",
            "app store",
            "install app",
            "install the app",
            "app launch",
            "launch app",
            "웹",
            "브라우저",
            "사이트",
            "페이지",
            "클릭",
            "로그인",
            "캡차",
            "플레이스토어",
            "플레이 스토어",
            "앱 설치",
            "앱 실행",
            "게시판",
            "댓글",
            "쪽지",
            "메시지",
            "message",
        )
    ) or _looks_like_naver_cafe(text) or _looks_like_naver_note(text)


def _looks_like_naver_cafe(text: str) -> bool:
    return (
        ("naver" in text and "cafe" in text)
        or "네이버 카페" in text
        or "품앗이" in text
        or "답방" in text
    )


def _looks_like_naver_note(text: str) -> bool:
    return (
        ("naver" in text and "note" in text)
        or "네이버 쪽지" in text
        or "네이버쪽지" in text
        or "note.naver.com" in text
        or ("쪽지" in text and ("naver" in text or "네이버" in text or "@naver.com" in text))
    )


def _prefers_app_automation(text: str) -> bool:
    return _has_any(
        text.casefold(),
        (
            "app_action",
            "android_action",
            "mobile_action",
            "device_action",
            "android-device",
            "android_device",
            "not browser",
            "no browser",
            "never a browser",
            "android review",
            "mobile qa",
            "install app",
            "install the app",
            "verify launch",
            "verify app launch",
            "launch app",
            "open android",
            "open the android",
            "open settings",
            "open settings app",
            "open android settings",
            "play store",
            "google play",
            "앱 설치",
            "앱 실행",
            "실행 검증",
            "플레이스토어",
            "플레이 스토어",
        ),
    )


def _looks_like_android_review_exchange(text: str) -> bool:
    normalized = text.casefold()
    return (
        _has_any(
            normalized,
            (
                "android review exchange",
                "app review exchange",
                "mobile review exchange",
                "review exchange agent",
                "review exchange bot",
                "review swap",
                "review request exchange",
                "리뷰 품앗이",
                "리뷰 교환",
                "리뷰 맞교환",
                "리뷰 답방",
            ),
        )
        or (
            _has_any(normalized, ("android", "mobile", "app", "앱", "안드로이드"))
            and _has_any(normalized, ("review", "리뷰"))
            and _has_any(normalized, ("exchange", "swap", "품앗이", "교환", "답방"))
        )
    )


_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+", re.IGNORECASE)
_ANDROID_PACKAGE_RE = re.compile(
    r"\b(?:[a-zA-Z][a-zA-Z0-9_]*\.){2,}[a-zA-Z][a-zA-Z0-9_]*\b"
)


def _extract_naver_note_recipient(text: str) -> str | None:
    for match in _EMAIL_RE.finditer(text):
        email = match.group(0)
        if email.casefold().endswith("@naver.com"):
            return email
    return None


def _extract_url(text: str) -> str | None:
    match = _URL_RE.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,;:!?)]}")


def _extract_android_package_name(text: str) -> str | None:
    for match in _ANDROID_PACKAGE_RE.finditer(text):
        package_name = match.group(0).rstrip(".,;:!?)]}")
        if package_name.startswith(("http.", "https.")):
            continue
        return package_name
    return None


def _extract_required_visible_text(text: str) -> str | None:
    patterns = (
        r"본문에\s+(.{2,120}?)\s*(?:문구|텍스트|글자)",
        r"(.{2,120}?)\s*(?:문구|텍스트|글자)(?:가|이)?\s*(?:보이는지|있는지|포함되는지|포함되어)",
        r"(?:contains|include|includes|including)\s+['\"]?(.{2,120}?)['\"]?(?:\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = _clean_visible_text_candidate(match.group(1))
        if value:
            return value
    quoted = re.search(r"['\"]([^'\"]{2,120})['\"]", text)
    if quoted:
        return _clean_visible_text_candidate(quoted.group(1))
    return None


def _clean_visible_text_candidate(value: str) -> str | None:
    text = value.strip()
    text = re.sub(r"^(?:에|에서|the)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" \t\r\n'\"`.,;:!?()[]{}")
    if not text or text.startswith("http://") or text.startswith("https://"):
        return None
    return text[:120]


def _ensure_requested_generic_capabilities(
    draft: AgentDraft,
    user_message: str,
) -> AgentDraft:
    capabilities = _requested_generic_capabilities(user_message)
    if not capabilities:
        return draft

    next_draft = draft
    if capabilities & {"join_request", "open_app_store", "install_app", "verify_launch"}:
        next_draft = _ensure_app_action_tool(next_draft)

    package_name = _extract_android_package_name(user_message)
    additions: list[WorkflowStep] = []
    if "remember_request_id" in capabilities:
        additions.append(_remember_request_id_step())
    if "join_request" in capabilities:
        additions.extend(_join_request_steps())
    if "open_app_store" in capabilities:
        additions.append(_open_app_store_step())
    if "install_app" in capabilities:
        additions.extend(_install_app_steps())
    if "verify_launch" in capabilities:
        additions.append(_verify_app_launch_step(package_name=package_name))
    if "save_result" in capabilities:
        additions.append(_save_result_step())

    if not additions:
        return next_draft
    return next_draft.model_copy(
        update={"flow": _append_missing_capability_steps(next_draft.flow, additions)}
    )


def _ensure_android_review_exchange_template(
    draft: AgentDraft,
    text: str,
) -> AgentDraft:
    if not _looks_like_android_review_exchange(text):
        return draft

    next_draft = _ensure_app_action_tool(draft)
    additions = [
        _collect_review_exchange_requests_step(),
        _select_review_exchange_candidate_step(),
        *_join_request_steps(),
        _open_app_store_step(),
        *_install_app_steps(),
        _verify_app_launch_step(),
        _save_result_step(),
    ]
    return next_draft.model_copy(
        update={"flow": _append_missing_capability_steps(next_draft.flow, additions)}
    )


def _requested_generic_capabilities(user_message: str) -> set[str]:
    text = user_message.casefold()
    capabilities: set[str] = set()

    if _has_any(
        text,
        (
            "request id",
            "request_id",
            "request-id",
            "remember id",
            "요청 id",
            "요청 아이디",
            "요청번호",
            "요청 번호",
        ),
    ):
        capabilities.add("remember_request_id")
    if _has_any(
        text,
        (
            "join request",
            "request to join",
            "submit request",
            "participation request",
            "참여 신청",
            "참여신청",
            "참여하고",
            "참여하기",
            "가입 신청",
            "가입신청",
            "신청 요청",
        ),
    ):
        capabilities.add("join_request")
    if _has_any(
        text,
        (
            "play store",
            "google play",
            "app store",
            "store listing",
            "플레이스토어",
            "플레이 스토어",
            "앱 스토어",
            "스토어 페이지",
        ),
    ):
        capabilities.add("open_app_store")
    if _has_any(
        text,
        (
            "install app",
            "install the app",
            "installs app",
            "installs the app",
            "installing app",
            "installing the app",
            "app install",
            "install from store",
            "앱 설치",
            "설치하기",
        ),
    ):
        capabilities.update({"open_app_store", "install_app"})
    if _has_any(
        text,
        (
            "verify launch",
            "verifies launch",
            "verify app launch",
            "launch verification",
            "launch app",
            "open android",
            "open the android",
            "open settings",
            "open settings app",
            "open android settings",
            "open app",
            "app launches",
            "앱 실행",
            "앱 열기",
            "앱 열고",
            "앱을 열",
            "을 열",
            "를 열",
            "실행 확인",
            "실행 검증",
        ),
    ):
        capabilities.add("verify_launch")
    if _has_any(
        text,
        (
            "save result",
            "saves result",
            "saves the result",
            "record result",
            "records result",
            "records the result",
            "result recording",
            "store result",
            "persist result",
            "결과 저장",
            "결과 기록",
            "결과를 저장",
            "결과를 기록",
        ),
    ):
        capabilities.add("save_result")

    return capabilities


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _append_missing_capability_steps(
    flow: list[WorkflowStep],
    additions: list[WorkflowStep],
) -> list[WorkflowStep]:
    out = list(flow)
    for step in additions:
        if _flow_has_equivalent_capability(out, step):
            continue
        out.append(step)
    return out


def _normalize_app_automation_steps(draft: AgentDraft, user_message: str) -> AgentDraft:
    if not _prefers_app_automation(_intent_text(user_message, draft)):
        return draft

    changed = False
    next_flow: list[WorkflowStep] = []
    split_launch_actions = _has_dedicated_launch_followup_steps(draft.flow)
    for step in draft.flow:
        normalized = _normalize_app_automation_step(
            step,
            split_launch_actions=split_launch_actions,
        )
        changed = changed or normalized is not step
        next_flow.append(normalized)
    if not changed:
        return draft
    return _ensure_app_action_tool(draft.model_copy(update={"flow": next_flow}))


def _normalize_app_automation_step(
    step: WorkflowStep,
    *,
    split_launch_actions: bool = False,
) -> WorkflowStep:
    haystack = _step_search_text(step)
    tool_hint = (step.tool_hint or "").strip().casefold()
    is_app_tool = tool_hint in {
        "app_action",
        "android_adb",
        "android",
        "android-device",
        "android_device",
        "mobile_action",
        "device_action",
    }
    is_app_step = step.type in {"app_action", "android_action", "mobile_action", "device_action"}
    is_browser_step = step.type == "browser_action"
    is_llm_wait_step = step.type == "llm" and not step.actions and _looks_like_wait_step(haystack)
    if not (is_app_tool or is_app_step or step.type == "mcp_tool" or is_browser_step or is_llm_wait_step):
        return step
    if is_llm_wait_step:
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[{"type": "wait", "seconds": _wait_seconds_from_text(haystack)}],
        )
    if is_browser_step and _action_list_has_type(step.actions, "wait"):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[{"type": "wait", "seconds": 1}],
        )
    if is_browser_step and (
        _action_list_has_type(step.actions, "read_screen")
        or _action_list_has_type(step.actions, "extract")
    ):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[{"type": "read_screen", "target": "current_screen"}],
        )
    if is_browser_step and _action_list_has_type(step.actions, "screenshot"):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[{"type": "screenshot", "label": _safe_action_label(step, "screen")}],
        )
    if step.actions and not is_browser_step:
        if _has_any(
            haystack,
            ("install app", "app install", "install the app", "installs the app", "앱 설치"),
        ) and not _action_list_has_type(step.actions, "install_app"):
            actions = [
                *step.actions,
                {"type": "install_app", "source": "user_provided_store_or_package"},
            ]
        else:
            actions = step.actions
        return _with_app_actions(
            step,
            step_type="app_action" if step.type == "mcp_tool" and is_app_tool else step.type,
            tool_hint="android_adb" if is_app_tool else (step.tool_hint or "android_adb"),
            actions=actions,
        )
    if _has_any(haystack, ("open settings", "settings app", "android settings", "device settings", "com.android.settings")):
        package_name = _extract_android_package_name(haystack) or "com.android.settings"
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=_verify_launch_actions(
                package_name,
                include_evidence=not split_launch_actions,
            ),
        )
    if is_browser_step and _has_any(haystack, ("wait", "대기")):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[{"type": "wait", "seconds": 1}],
        )
    if is_browser_step and _has_any(
        haystack,
        (
            "read_screen",
            "read screen",
            "dump screen",
            "screen state",
            "visible screen",
            "ui hierarchy",
            "화면 상태",
            "화면 읽",
        ),
    ):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[{"type": "read_screen", "target": "current_screen"}],
        )
    if is_browser_step and _has_any(haystack, ("screenshot", "screen shot", "스크린샷", "화면 캡처")):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[{"type": "screenshot", "label": _safe_action_label(step, "screen")}],
        )
    if is_browser_step and _has_any(
        haystack,
        (
            "verify_launch",
            "verify launch",
            "launch app",
            "app launches",
            "앱 실행",
            "실행 확인",
            "실행 검증",
        ),
    ):
        package_name = _extract_android_package_name(haystack)
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=_verify_launch_actions(
                package_name,
                include_evidence=not split_launch_actions,
            ),
        )
    if step.actions and _has_any(
        haystack,
        ("install app", "app install", "install the app", "installs the app", "앱 설치"),
    ) and not _action_list_has_type(step.actions, "install_app"):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[
                *step.actions,
                {"type": "install_app", "source": "user_provided_store_or_package"},
            ],
        )
    if step.actions:
        return _with_app_actions(
            step,
            step_type="app_action" if step.type == "mcp_tool" and is_app_tool else step.type,
            tool_hint="android_adb" if is_app_tool else (step.tool_hint or "android_adb"),
            actions=_repair_app_action_payloads(step),
        )

    if _has_any(haystack, ("install app", "app install", "install the app", "installs the app", "앱 설치")):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=step.actions or [
                {"type": "install_app", "source": "user_provided_store_or_package"},
            ],
        )
    if _has_any(haystack, ("open play store", "play store", "google play", "앱 스토어", "플레이스토어", "플레이 스토어")):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=step.actions or [
                {"type": "open_play_store", "source": "user_provided_store_or_package"},
                {"type": "wait", "target": "app_listing_visible"},
            ],
        )
    if _has_any(haystack, ("verify launch", "launch app", "app launches", "앱 실행", "실행 확인", "실행 검증")):
        package_name = _extract_android_package_name(haystack)
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=step.actions
            or _verify_launch_actions(
                package_name,
                include_evidence=not split_launch_actions,
            ),
        )
    if _has_any(haystack, ("join request", "join exchange", "request to join", "participation request", "참여 신청", "가입 신청")):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=step.actions or [
                {"type": "tap_text", "text": "join_or_apply_control_from_current_screen"},
                {"type": "wait", "target": "join_request_result"},
            ],
        )
    if _has_any(
        haystack,
        (
            "collect request",
            "fetch request",
            "list request",
            "read request",
            "review exchange request",
            "review request",
            "품앗이 요청",
            "리뷰 교환 요청",
            "리뷰 품앗이 요청",
            "요청 수집",
            "요청 목록",
            "신규 요청",
        ),
    ):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=step.actions or [
                {"type": "read_screen", "target": "current_review_exchange_request_list"},
            ],
        )
    if _has_any(
        haystack,
        (
            "read screen",
            "dump screen",
            "screen state",
            "visible screen",
            "ui hierarchy",
            "화면 상태",
            "화면 읽",
            "화면 덤프",
        ),
    ):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=step.actions or [
                {"type": "read_screen", "target": "current_screen"},
            ],
        )
    if _has_any(haystack, ("screenshot", "screen shot", "capture png", "스크린샷", "화면 캡처")):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=step.actions or [
                {"type": "screenshot", "label": _safe_action_label(step, "screen")},
            ],
        )
    if step.type == "mcp_tool" and not step.actions and _has_any(
        haystack,
        (
            "remember",
            "memory",
            "save result",
            "record result",
            "persist result",
            "결과 저장",
            "결과 기록",
            "기억",
            "메모리",
        ),
    ):
        return step.model_copy(update={"type": "llm", "tool_hint": None, "actions": []})
    if step.type == "mcp_tool" and not step.actions:
        return step.model_copy(update={"type": "llm", "tool_hint": None, "actions": []})
    return step


def _repair_empty_workflow_actions(draft: AgentDraft, text: str) -> AgentDraft:
    repaired: list[WorkflowStep] = []
    changed = False
    for step in draft.flow:
        next_step = step
        if not step.actions and _step_requires_runtime_actions(step):
            next_step = _repair_empty_runtime_action_step(step, text)
        repaired.append(next_step)
        changed = changed or next_step != step
    if not changed:
        return draft
    next_draft = draft.model_copy(update={"flow": repaired})
    if any(
        step.type in {"app_action", "android_action", "mobile_action", "device_action"}
        for step in repaired
    ):
        next_draft = _ensure_app_action_tool(next_draft)
    if any(step.type == "browser_action" for step in repaired):
        next_draft = _ensure_playwright_tool(next_draft)
    return next_draft


def _step_requires_runtime_actions(step: WorkflowStep) -> bool:
    return step.type in {
        "browser_action",
        "app_action",
        "android_action",
        "mobile_action",
        "device_action",
    }


def _repair_empty_runtime_action_step(step: WorkflowStep, text: str) -> WorkflowStep:
    haystack = f"{_step_search_text(step)}\n{text.casefold()}"
    if _has_any(
        haystack,
        (
            "collect request",
            "fetch request",
            "list request",
            "read request",
            "review exchange request",
            "review request",
            "품앗이 요청",
            "리뷰 교환 요청",
            "리뷰 품앗이 요청",
            "요청 수집",
            "요청 목록",
            "신규 요청",
        ),
    ):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[
                {"type": "read_screen", "target": "current_review_exchange_request_list"},
                {"type": "screenshot", "label": _safe_action_label(step, "request_list")},
            ],
        )
    if _has_any(haystack, ("install app", "app install", "install the app", "installs the app", "앱 설치")):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[{"type": "install_app", "source": "user_provided_store_or_package"}],
        )
    if _has_any(haystack, ("play store", "google play", "app store", "store listing", "플레이스토어", "플레이 스토어", "앱 스토어")):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[
                {"type": "open_play_store", "source": "user_provided_store_or_package"},
                {"type": "wait", "target": "app_listing_visible"},
            ],
        )
    if _has_any(haystack, ("verify launch", "launch app", "app launches", "앱 실행", "실행 확인", "실행 검증")):
        package_name = _extract_android_package_name(haystack)
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=_verify_launch_actions(
                package_name,
                include_evidence=not split_launch_actions,
            ),
        )
    if _has_any(haystack, ("join request", "request to join", "participation request", "참여 신청", "가입 신청", "참여하고")):
        return _with_app_actions(
            step,
            step_type="app_action",
            tool_hint="android_adb",
            actions=[
                {"type": "tap_text", "text": "join_or_apply_control_from_current_screen"},
                {"type": "wait", "target": "join_request_result"},
                {"type": "screenshot", "label": _safe_action_label(step, "join_request_result")},
            ],
        )
    if step.type == "browser_action":
        return step.model_copy(
            update={
                "actions": [
                    {"type": "navigate", "url": "configured_target_url"},
                    {"type": "screenshot", "label": _safe_action_label(step, "browser_state")},
                ]
            }
        )
    return step.model_copy(
        update={
            "actions": [
                {"type": "read_screen", "target": "current_screen"},
                {"type": "screenshot", "label": _safe_action_label(step, "screen")},
            ]
        }
    )


def _safe_action_label(step: WorkflowStep, suffix: str) -> str:
    raw = (step.id or step.name or "workflow_step").strip().casefold()
    label = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    return f"{label or 'workflow_step'}_{suffix}"


def _looks_like_wait_step(text: str) -> bool:
    return _has_any(
        text,
        (
            "wait",
            "sleep",
            "elapsed",
            "settle",
            "render",
            "load before",
            "time to draw",
            "대기",
            "기다",
        ),
    )


def _has_dedicated_launch_followup_steps(flow: list[WorkflowStep]) -> bool:
    followup_step_ids: set[str] = set()
    for step in flow:
        text = _step_search_text(step)
        step_key = step.id or step.name or text
        if _action_list_has_type(step.actions, "wait") or _looks_like_wait_step(text):
            followup_step_ids.add(step_key)
        if _action_list_has_type(step.actions, "read_screen") or _has_any(
            text,
            (
                "read_screen",
                "read screen",
                "dump screen",
                "screen state",
                "visible screen",
                "ui hierarchy",
                "화면 상태",
                "화면 읽",
            ),
        ):
            followup_step_ids.add(step_key)
        if _action_list_has_type(step.actions, "screenshot") or _has_any(
            text,
            ("screenshot", "screen shot", "스크린샷", "화면 캡처"),
        ):
            followup_step_ids.add(step_key)
    return len(followup_step_ids) >= 2


def _wait_seconds_from_text(text: str) -> int:
    match = re.search(r"(\d+)\s*(?:seconds?|secs?|s|초)", text)
    if match:
        return max(1, min(int(match.group(1)), 30))
    match = re.search(r"(\d+)\s*(?:minutes?|mins?|m|분)", text)
    if match:
        return max(1, min(int(match.group(1)) * 60, 300))
    return 1


def _repair_app_action_payloads(step: WorkflowStep) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    fallback_label = (step.id or step.name or "app_action").strip() or "app_action"
    for index, action in enumerate(step.actions):
        if not isinstance(action, dict):
            repaired.append(action)
            continue
        next_action = dict(action)
        if next_action.get("type") == "screenshot" and not (
            next_action.get("label") or next_action.get("target")
        ):
            next_action["label"] = f"{fallback_label}_screenshot_{index + 1}"
        repaired.append(next_action)
    return repaired


def _action_list_has_type(actions: list[dict[str, Any]], action_type: str) -> bool:
    return any(isinstance(action, dict) and action.get("type") == action_type for action in actions)


def _with_app_actions(
    step: WorkflowStep,
    *,
    step_type: str,
    tool_hint: str,
    actions: list[dict[str, Any]],
) -> WorkflowStep:
    repaired_actions = _repair_app_action_payloads(step.model_copy(update={"actions": actions}))
    if step.type == step_type and step.tool_hint == tool_hint and step.actions == repaired_actions:
        return step
    return step.model_copy(
        update={
            "type": step_type,
            "tool_hint": tool_hint,
            "actions": repaired_actions,
        }
    )


def _flow_has_equivalent_capability(
    flow: list[WorkflowStep],
    requested_step: WorkflowStep,
) -> bool:
    requested_id = (requested_step.id or "").strip().casefold()
    for step in flow:
        if requested_id and (step.id or "").strip().casefold() == requested_id:
            return True
        haystack = _step_search_text(step)
        if requested_id == "collect_review_exchange_requests" and _has_any(
            haystack,
            (
                "collect request",
                "collect and select a request",
                "current review exchange request",
                "review exchange request list",
                "read request",
                "요청 수집",
                "요청 목록",
                "리뷰 교환 요청",
                "리뷰 품앗이 요청",
            ),
        ):
            return True
        if requested_id == "select_review_exchange_candidate" and _has_any(
            haystack,
            (
                "select candidate",
                "select request",
                "collect and select a request",
                "duplicate",
                "dedupe",
                "중복",
                "요청 선택",
                "후보 선택",
            ),
        ):
            return True
        if requested_id == "remember_request_id" and _has_any(
            haystack,
            ("request id", "request_id", "request-id", "요청 id", "요청번호"),
        ):
            return True
        if requested_id == "prepare_join_request_context" and _has_any(
            haystack,
            ("join request", "join exchange", "request to join", "participation request", "참여 신청", "가입 신청"),
        ):
            return True
        if requested_id == "approve_join_request_submission" and step.type == "approval_gate" and _has_any(
            haystack,
            ("join request", "join exchange", "request to join", "participation request", "참여 신청", "가입 신청"),
        ):
            return True
        if requested_id == "submit_join_request" and _has_concrete_app_action(step, "tap_text") and _has_any(
            haystack,
            ("join request", "join exchange", "request to join", "participation request", "참여 신청", "가입 신청"),
        ):
            return True
        if requested_id == "open_app_store_listing" and _has_concrete_app_action(step, "open_play_store") and _has_any(
            haystack,
            (
                "play store",
                "google play",
                "app store",
                "store listing",
                "open_play_store",
                "open_app_store",
                "플레이스토어",
                "스토어 페이지",
            ),
        ):
            return True
        if requested_id == "approve_app_install" and step.type == "approval_gate" and _has_any(
            haystack,
            ("install app", "app install", "install the app", "installs the app", "앱 설치"),
        ):
            return True
        if requested_id == "install_app" and _has_concrete_app_action(step, "install_app") and _has_any(
            haystack,
            ("install app", "app install", "install the app", "installs the app", "앱 설치"),
        ):
            return True
        if requested_id == "verify_app_launch" and _has_concrete_app_action(step, "verify_launch") and _has_any(
            haystack,
            ("verify launch", "verify_launch", "launch app", "app launches", "앱 실행", "실행 확인"),
        ):
            return True
        if requested_id == "record_execution_result":
            step_id = (step.id or "").strip().casefold()
            if step_id in {"remember_request_id", "verify_and_report_result"}:
                continue
            if _has_any(
                haystack,
                (
                    "save result",
                    "record result",
                    "result recording",
                    "store result",
                    "persist result",
                    "결과 저장",
                    "결과 기록",
                ),
            ):
                return True
    return False


def _has_concrete_app_action(step: WorkflowStep, action_type: str) -> bool:
    if step.type not in {"app_action", "android_action", "mobile_action", "device_action"}:
        return False
    return any(isinstance(action, dict) and action.get("type") == action_type for action in step.actions)


def _step_search_text(step: WorkflowStep) -> str:
    parts = [
        step.id or "",
        step.name,
        step.description,
        step.success_criteria,
        step.tool_hint or "",
        json.dumps(step.actions, ensure_ascii=False, sort_keys=True),
    ]
    for attr in ("instruction", "observation", "memory_read", "memory_write"):
        value = getattr(step, attr, "")
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts).casefold()


def _collect_review_exchange_requests_step() -> WorkflowStep:
    return WorkflowStep(
        id="collect_review_exchange_requests",
        type="app_action",
        name="리뷰 교환 요청 수집",
        description="연결된 Android 앱 화면에서 현재 처리 가능한 리뷰 교환 요청 목록과 대상 앱 정보를 읽는다.",
        instruction="Read the visible review exchange request list. Do not join, install, or submit anything in this step.",
        tool_hint="android_adb",
        actions=[
            {"type": "read_screen", "target": "current_review_exchange_request_list"},
            {"type": "screenshot", "label": "review_exchange_request_list"},
        ],
        success_criteria="처리 가능한 요청, 대상 앱, 요청 ID 또는 식별자가 확인됨",
        on_failure={
            "type": "manual_handoff",
            "resume": "same_step",
            "prompt": "리뷰 교환 요청 목록 화면을 열어 둔 뒤 재개하세요.",
        },
    )


def _select_review_exchange_candidate_step() -> WorkflowStep:
    return WorkflowStep(
        id="select_review_exchange_candidate",
        type="llm",
        name="중복 없는 요청 선택",
        description="수집한 요청 중 이미 처리한 요청, 동일 앱/동일 요청 ID, 정책상 위험한 요청을 제외하고 하나를 선택한다.",
        instruction="Select one safe review exchange request. Skip duplicates and anything requiring policy, captcha, or account bypass.",
        memory_read="최근 리뷰 교환 요청 ID, 대상 앱, 설치/실행/신청 결과 이력을 확인한다.",
        success_criteria="중복이 아니고 처리 가능한 리뷰 교환 요청 하나가 선택됨",
        on_failure={
            "type": "ask_user",
            "resume": "same_step",
            "prompt": "처리할 리뷰 교환 요청을 선택할 수 없습니다. 기준이나 대상 요청을 알려주세요.",
        },
    )


def _remember_request_id_step() -> WorkflowStep:
    return WorkflowStep(
        id="remember_request_id",
        type="llm",
        name="요청 ID 기억",
        description="사용자가 제공한 요청 ID를 이번 실행의 추적 키로 확인하고 이후 단계와 결과 기록에 연결한다.",
        instruction="Extract the request identifier from the user request or ask for it before continuing.",
        memory_write="요청 ID, 관련 대상, 실행 결과를 다음 실행에서 조회할 수 있게 저장한다.",
        success_criteria="요청 ID가 확인되고 결과 기록에 사용할 추적 키가 준비됨",
        on_failure={
            "type": "ask_user",
            "resume": "same_step",
            "prompt": "이 실행에 연결할 요청 ID를 알려주세요.",
        },
    )


def _join_request_steps() -> list[WorkflowStep]:
    return [
        WorkflowStep(
            id="prepare_join_request_context",
            type="llm",
            name="신청 요청 정보 확인",
            description="신청 대상, 신청 조건, 요청 ID, 제출할 내용, 중복 신청 여부를 확인한다.",
            instruction="Confirm the target, payload, constraints, and duplicate-prevention context for the join/request action.",
            memory_read="동일 대상 또는 동일 요청 ID로 이미 처리한 신청 이력이 있는지 확인한다.",
            success_criteria="신청 대상과 제출 조건이 명확하고 중복 신청이 아님",
            on_failure={"type": "ask_user", "resume": "same_step"},
        ),
        WorkflowStep(
            id="approve_join_request_submission",
            type="approval_gate",
            name="신청 제출 전 승인",
            description="외부 서비스에 신청 요청을 제출하기 전에 대상과 내용을 사용자에게 확인받는다.",
            instruction="Ask for explicit approval before submitting the join/request action.",
            success_criteria="사용자가 신청 대상과 제출 내용을 승인함",
            on_failure={"type": "abort", "reason": "join_request_not_approved"},
        ),
        WorkflowStep(
            id="submit_join_request",
            type="app_action",
            name="신청 요청 제출",
            description="승인된 대상에 한해 연결된 앱 화면에서 신청 요청을 제출하고 결과 화면을 확인한다.",
            instruction="Only submit after the approval gate succeeds. Stop for handoff if login, captcha, account challenge, or native permission prompt appears.",
            tool_hint="android_adb",
            actions=[
                {"type": "tap_text", "text": "join_or_apply_control_from_current_screen"},
                {"type": "wait", "target": "join_request_result"},
                {"type": "screenshot", "label": "join_request_result"},
            ],
            success_criteria="신청 요청의 성공 또는 명확한 실패 사유가 기록됨",
            on_failure={
                "type": "manual_handoff",
                "resume": "same_step",
                "prompt": "로그인, 인증, 캡차, 권한 확인 또는 예기치 않은 제출 화면을 직접 처리한 뒤 재개하세요.",
            },
        ),
    ]


def _open_app_store_step() -> WorkflowStep:
    return WorkflowStep(
        id="open_app_store_listing",
        type="app_action",
        name="앱 스토어 페이지 열기",
        description="연결된 Android 환경에서 사용자가 지정한 Play Store 페이지를 열고 설치 가능 상태를 확인한다.",
        instruction="Open the configured Play Store listing. Do not install or submit account actions in this step.",
        tool_hint="android_adb",
        actions=[
            {"type": "open_play_store", "source": "user_provided_store_or_package"},
            {"type": "wait", "target": "app_listing_visible"},
            {"type": "screenshot", "label": "app_store_listing"},
        ],
        success_criteria="앱 스토어 페이지가 열리고 대상 앱을 식별할 수 있음",
        on_failure={
            "type": "manual_handoff",
            "resume": "same_step",
            "prompt": "스토어 로그인, 지역 제한, 인증 또는 접근 권한 확인을 직접 처리한 뒤 재개하세요.",
        },
    )


def _install_app_steps() -> list[WorkflowStep]:
    return [
        WorkflowStep(
            id="approve_app_install",
            type="approval_gate",
            name="앱 설치 전 승인",
            description="기기 또는 계정에 앱을 설치하기 전에 대상 앱과 설치 범위를 사용자에게 확인받는다.",
            instruction="Require explicit approval before initiating or asking the user to complete an app install.",
            success_criteria="사용자가 대상 앱 설치를 승인함",
            on_failure={"type": "abort", "reason": "app_install_not_approved"},
        ),
        WorkflowStep(
            id="install_app",
            type="app_action",
            name="앱 설치",
            description="승인 후 대상 앱 설치를 진행하고 설치 완료 또는 실패 사유를 확인한다.",
            instruction=(
                "Install the approved target app when a concrete package, store URL, "
                "or user-provided source is available. Do not bypass account, billing, "
                "device confirmation, or OS permission prompts."
            ),
            tool_hint="android_adb",
            actions=[
                {"type": "install_app", "source": "user_provided_store_or_package"},
            ],
            success_criteria="앱 설치 완료 또는 설치 불가 사유가 확인됨",
            on_failure={
                "type": "manual_handoff",
                "resume": "same_step",
                "prompt": "스토어 로그인, 기기 확인, 결제/권한 프롬프트 또는 OS 설치 확인을 직접 처리한 뒤 재개하세요.",
            },
        ),
    ]


def _verify_launch_actions(
    package_name: str | None = None,
    *,
    include_evidence: bool = True,
) -> list[dict[str, Any]]:
    target = (
        {"type": "verify_launch", "package": package_name}
        if package_name
        else {"type": "verify_launch", "app": "installed_app_from_previous_step"}
    )
    if not include_evidence:
        return [target]
    return [
        target,
        {"type": "wait", "seconds": 1},
        {"type": "read_screen", "target": "launched_app_screen"},
        {"type": "screenshot", "label": "app_launch_result"},
    ]


def _verify_app_launch_step(package_name: str | None = None) -> WorkflowStep:
    return WorkflowStep(
        id="verify_app_launch",
        type="app_action",
        name="앱 실행 확인",
        description="설치된 앱을 실행해 첫 화면, 로그인/권한 요청, 오류 여부를 확인하고 증거를 남긴다.",
        instruction="Launch or ask the connected device workflow to launch the installed app and report the visible result.",
        tool_hint="android_adb",
        actions=_verify_launch_actions(package_name),
        success_criteria="앱이 실행되었는지 또는 실행 실패 사유가 확인됨",
        on_failure={
            "type": "manual_handoff",
            "resume": "same_step",
            "prompt": "앱 실행, 로그인, 초기 권한 또는 첫 화면 확인을 직접 처리한 뒤 재개하세요.",
        },
    )


def _save_result_step() -> WorkflowStep:
    return WorkflowStep(
        id="record_execution_result",
        type="llm",
        name="결과 저장",
        description="요청 ID, 수행 단계, 성공/실패 상태, 수동 개입 내용, 다음 조치를 요약해 저장한다.",
        instruction="Persist the run result in memory and summarize it for the user.",
        memory_write="요청 ID, 대상 앱/서비스, 설치/실행/신청 결과, 오류 사유, 후속 조치를 저장한다.",
        success_criteria="결과가 사용자에게 보고되고 다음 실행에서 참고할 수 있게 저장됨",
        on_failure={"type": "ask_user", "resume": "same_step"},
    )


def _extract_schedule(text: str) -> str | None:
    if any(marker in text for marker in ("every hour", "hourly", "매시간", "매 시간", "1시간마다", "한 시간마다")):
        return "every 1h"
    match = re.search(r"every\s+(\d+)\s*(m|min|minute|minutes)\b", text)
    if match:
        return f"every {match.group(1)}m"
    match = re.search(r"every\s+(\d+)\s*(h|hr|hour|hours)\b", text)
    if match:
        return f"every {match.group(1)}h"
    match = re.search(r"(\d+)\s*분마다", text)
    if match:
        return f"every {match.group(1)}m"
    match = re.search(r"(\d+)\s*시간마다", text)
    if match:
        return f"every {match.group(1)}h"
    if any(marker in text for marker in ("nightly", "매일 밤")):
        return "daily 21:00"
    match = re.search(r"daily\s+([0-2]?\d:[0-5]\d)", text)
    if match:
        return f"daily {match.group(1)}"
    match = re.search(r"매일\s*([0-2]?\d)[:시]\s*([0-5]\d)?", text)
    if match:
        minute = match.group(2) or "00"
        return f"daily {int(match.group(1)):02d}:{minute}"
    return None


def _ensure_playwright_tool(draft: AgentDraft) -> AgentDraft:
    if any(tool.mcp_id == "playwright" for tool in draft.tools):
        return draft
    return draft.model_copy(
        update={
            "tools": [
                *draft.tools,
                AgentToolDraft(
                    mcp_id="playwright",
                    tool_names=[
                        "browser_navigate",
                        "browser_snapshot",
                        "browser_click",
                        "browser_type",
                    ],
                    user_capability="웹사이트를 열고 상태를 읽고 필요한 버튼/입력 필드를 조작한다.",
                    user_examples=["네이버 카페 게시판 열기", "신청 버튼 클릭", "댓글/폼 입력"],
                    category=MCPCategory.BROWSER,
                    risk_tier=MCPRiskTier.MEDIUM,
                ),
            ]
        }
    )


def _ensure_app_action_tool(draft: AgentDraft) -> AgentDraft:
    if any(tool.mcp_id == "app_action" for tool in draft.tools):
        return draft
    return draft.model_copy(
        update={
            "tools": [
                *draft.tools,
                AgentToolDraft(
                    mcp_id="app_action",
                    tool_names=[
                        "list_packages",
                        "open_play_store",
                        "install_package",
                        "install_app",
                        "launch_app",
                        "verify_launch",
                        "tap_text",
                        "read_screen",
                        "wait",
                    ],
                    user_capability="연결된 Android 기기에서 앱 설치, 실행, 화면 확인, 간단한 탭을 수행한다.",
                    user_examples=["Play Store 열기", "앱 설치", "설치된 앱 실행 확인"],
                    category=MCPCategory.OTHER,
                    risk_tier=MCPRiskTier.MEDIUM,
                ),
            ]
        }
    )


def _flow_needs_operational_plan(
    flow: list[WorkflowStep],
    *,
    plan_kind: str,
) -> bool:
    if not flow:
        return True
    if plan_kind in {"naver_cafe", "naver_note"} and len(flow) < 3:
        return True
    return all(_is_placeholder_step(step) for step in flow)


def _is_placeholder_step(step: WorkflowStep) -> bool:
    return (
        bool(re.fullmatch(r"step\s*\d+", step.name.strip(), flags=re.IGNORECASE))
        and not step.description.strip()
        and not (step.tool_hint or "").strip()
        and not step.success_criteria.strip()
    )


def _naver_cafe_flow() -> list[WorkflowStep]:
    return [
        WorkflowStep(
            id="check_run_context",
            type="llm",
            name="실행 조건과 처리 이력 확인",
            description="대상 카페 URL, 게시판, 검색어, 신청 문구 템플릿, 시간당 처리 한도, 최근 처리 이력을 확인한다.",
            success_criteria="대상과 제한 조건이 확인되고 중복 방지 기준이 준비됨",
            on_failure={"type": "ask_user", "resume": "same_step"},
        ),
        WorkflowStep(
            id="open_cafe_and_check_login",
            type="browser_action",
            name="네이버 카페 열기와 로그인 상태 확인",
            description="Playwright로 대상 네이버 카페/게시판을 열고 로그인 세션, 접근 권한, 캡차/봇 차단 여부를 확인한다.",
            tool_hint="playwright",
            actions=[
                {"type": "navigate", "target": "configured_cafe_url"},
                {"type": "assert", "kind": "login_or_access_state_visible"},
                {"type": "screenshot", "label": "login_state"},
            ],
            success_criteria="게시판이 열리고 캡차 없이 자동화 가능한 상태임",
            on_failure={
                "type": "manual_handoff",
                "resume": "same_step",
                "prompt": "네이버 로그인, 접근 권한 확인, 캡차 같은 수동 처리를 완료한 뒤 재개하세요.",
            },
        ),
        WorkflowStep(
            id="discover_pumasi_posts",
            type="browser_action",
            name="품앗이 후보 글 탐색",
            description="게시판의 최신 글을 읽고 사용자가 지정한 조건에 맞는 품앗이/출석/답방 후보를 선별한다.",
            tool_hint="playwright",
            actions=[
                {"type": "extract", "target": "post_list"},
                {"type": "assert", "kind": "candidate_posts_evaluated"},
            ],
            success_criteria="후보 글 URL, 제목, 작성자, 조건 목록이 추려짐",
            on_failure={
                "type": "retry",
                "max_attempts": 1,
                "then": {"type": "ask_user", "resume": "same_step"},
            },
        ),
        WorkflowStep(
            id="dedupe_and_safety_check",
            type="llm",
            name="중복과 안전 조건 검사",
            description="이미 신청한 글, 24시간 내 중복 대상, 캡차/개인정보/금전 요구, 카페 규정 위반 가능성을 걸러낸다.",
            tool_hint="playwright",
            success_criteria="신청 가능한 대상만 남고 위험/중복 대상은 스킵 사유와 함께 제외됨",
            on_failure={"type": "abort", "reason": "safety_check_failed"},
        ),
        WorkflowStep(
            id="submit_participation_requests",
            type="browser_action",
            name="참여신청 수행",
            description="허용된 대상에 한해 사용자의 문구 템플릿 또는 짧고 정중한 기본 문구로 참여신청을 입력/제출한다.",
            tool_hint="playwright",
            actions=[
                {"type": "click", "target": "eligible_post_or_reply_control"},
                {"type": "type", "target": "reply_or_application_field", "source": "configured_message_template"},
                {"type": "click", "target": "submit_button"},
                {"type": "assert", "kind": "submission_recorded_or_failure_reason_visible"},
            ],
            success_criteria="각 신청 대상에 대해 제출 성공 또는 명확한 실패 사유가 기록됨",
            on_failure={
                "type": "ask_user",
                "resume": "same_step",
                "prompt": "신청 제출 중 자동으로 판단하기 어려운 화면이 나오면 처리 방법을 알려주세요.",
            },
        ),
        WorkflowStep(
            id="record_results_and_report",
            type="llm",
            name="결과 기록과 요약 보고",
            description="확인한 글 수, 신청한 URL, 스킵 사유, 오류, 다음 실행에서 참고할 중복 방지 이력을 요약한다.",
            success_criteria="사용자에게 실행 요약이 전달되고 처리 이력이 메모리에 남을 준비가 됨",
            on_failure={"type": "ask_user", "resume": "same_step"},
        ),
    ]


def _naver_note_flow(text: str) -> list[WorkflowStep]:
    recipient = _extract_naver_note_recipient(text) or "{{recipient}}"
    return [
        WorkflowStep(
            id="prepare_note_context",
            type="llm",
            name="쪽지 대상과 본문 확인",
            description="받는 사람, 쪽지 본문, 발송 목적, 중복 발송 여부를 확인한다. 본문이 없으면 발송 단계로 진행하지 않고 사용자에게 요청한다.",
            instruction=(
                "Confirm the Naver Note recipient, message body, and safety constraints. "
                "Do not proceed to sending if the message body is missing."
            ),
            observation="Recipient and body must be explicit. Duplicate sends must be avoided.",
            memory_read="최근 네이버 쪽지 발송 이력과 동일 수신자/동일 본문 여부를 확인한다.",
            memory_write="발송 성공/실패, 수신자, 본문 요약, 발송 시각을 다음 실행의 중복 방지 이력으로 저장한다.",
            success_criteria="수신자와 본문이 확인되고 중복/위험 조건이 없음",
            on_failure={
                "type": "ask_user",
                "resume": "same_step",
                "prompt": "받는 사람과 보낼 쪽지 본문을 알려주세요.",
            },
        ),
        WorkflowStep(
            id="open_naver_note_and_login",
            type="browser_action",
            name="네이버 쪽지 열기와 로그인 확인",
            description="서버 브라우저에서 네이버 쪽지를 열고 로그인/캡차/추가 인증 상태를 확인한다.",
            instruction="Open Naver Note and stop for user handoff if login, captcha, or account challenge appears.",
            tool_hint="playwright",
            actions=[
                {"type": "navigate", "url": "https://note.naver.com/"},
                {"type": "assert", "kind": "page_state_readable"},
                {"type": "screenshot", "label": "naver_note_login_state"},
            ],
            success_criteria="네이버 쪽지 화면이 열리고 로그인 세션 또는 수동 개입 필요 상태가 확인됨",
            on_failure={
                "type": "manual_handoff",
                "resume": "same_step",
                "prompt": "네이버 로그인, 2FA, 캡차 또는 접근 권한 확인을 직접 완료한 뒤 재개하세요.",
            },
        ),
        WorkflowStep(
            id="compose_note_draft",
            type="browser_action",
            name="쪽지 초안 작성",
            description="쪽지쓰기 화면을 열고 수신자와 본문을 입력하되 아직 발송하지 않는다.",
            instruction="Fill the recipient and approved message body. Do not click the final send button in this step.",
            tool_hint="playwright",
            actions=[
                {"type": "click", "selector": "text=쪽지쓰기"},
                {"type": "fill", "selector": "input[name='to'], input[title*='받는'], textarea[name='to']", "value": recipient},
                {"type": "fill", "selector": "textarea[name='content'], textarea[title*='내용'], div[contenteditable='true']", "value": "{{approved_note_body}}"},
                {"type": "screenshot", "label": "naver_note_draft_before_send"},
            ],
            success_criteria="수신자와 본문이 입력된 발송 직전 초안 화면이 캡처됨",
            on_failure={
                "type": "manual_handoff",
                "resume": "same_step",
                "prompt": "쪽지쓰기 화면 구조가 자동화와 다릅니다. 수신자와 본문을 직접 입력한 뒤 발송하지 말고 재개하세요.",
            },
        ),
        WorkflowStep(
            id="approve_note_send",
            type="approval_gate",
            name="발송 전 사용자 승인",
            description="수신자와 본문을 사용자에게 확인받고 발송을 승인받는다.",
            instruction="Ask for explicit approval before clicking the final send button.",
            success_criteria="사용자가 수신자/본문을 확인하고 발송을 승인함",
            on_failure={"type": "abort", "reason": "send_not_approved"},
        ),
        WorkflowStep(
            id="submit_note_and_capture_result",
            type="browser_action",
            name="쪽지 발송과 결과 확인",
            description="사용자 승인 후 발송 버튼을 클릭하고 성공/실패 상태를 캡처한다.",
            instruction="Only run after the approval gate has completed. Click send and capture the result state.",
            tool_hint="playwright",
            actions=[
                {"type": "click", "selector": "text=보내기"},
                {"type": "wait", "timeout_ms": 1200},
                {"type": "screenshot", "label": "naver_note_send_result"},
                {"type": "extract", "selector": "body"},
            ],
            success_criteria="쪽지 발송 성공 메시지 또는 명확한 실패 사유가 기록됨",
            on_failure={
                "type": "ask_user",
                "resume": "same_step",
                "prompt": "발송 결과를 자동으로 판단하지 못했습니다. 화면 상태를 확인하고 계속할지 알려주세요.",
            },
        ),
        WorkflowStep(
            id="record_note_result",
            type="llm",
            name="결과 기록과 보고",
            description="발송 여부, 수신자, 본문 요약, 실패 사유, 다음 실행에서 피해야 할 중복 조건을 정리한다.",
            instruction="Summarize the Naver Note send result and record duplicate-prevention memory.",
            memory_write="수신자, 본문 요약, 발송 결과, 오류 사유를 저장한다.",
            success_criteria="사용자가 결과와 남은 조치를 이해할 수 있음",
            on_failure={"type": "ask_user", "resume": "same_step"},
        ),
    ]


def _generic_web_automation_flow(
    *,
    target_url: str | None = None,
    required_text: str | None = None,
) -> list[WorkflowStep]:
    if target_url:
        check_actions: list[dict[str, Any]] = [
            {"type": "navigate", "url": target_url},
        ]
        if required_text:
            check_actions.append(
                {
                    "type": "assert",
                    "kind": "text_visible",
                    "value": required_text,
                    "timeout_ms": 5000,
                }
            )
        else:
            check_actions.append({"type": "assert", "kind": "page_state_readable"})
        check_actions.append({"type": "screenshot", "label": "web_check_result"})
        return [
            WorkflowStep(
                id="prepare_web_check",
                type="llm",
                name="점검 조건 확인",
                description="대상 URL과 성공 조건을 확인하고 로그인이나 민감한 제출이 필요 없는 점검인지 판단한다.",
                success_criteria="대상 URL과 성공 조건이 명확함",
                on_failure={"type": "ask_user", "resume": "same_step"},
            ),
            WorkflowStep(
                id="open_page_and_verify_content",
                type="browser_action",
                name="웹페이지 열기와 문구 확인",
                description="Playwright로 대상 페이지를 열고 요청된 문구 또는 읽기 가능한 페이지 상태를 확인한다.",
                tool_hint="playwright",
                actions=check_actions,
                success_criteria=(
                    f"'{required_text}' 문구가 확인됨"
                    if required_text
                    else "페이지가 열리고 읽기 가능한 상태임"
                ),
                on_failure={
                    "type": "retry",
                    "max_attempts": 1,
                    "then": {"type": "ask_user", "resume": "same_step"},
                },
            ),
            WorkflowStep(
                id="report_web_check_result",
                type="llm",
                name="점검 결과 보고",
                description="페이지 열기, 문구 확인, 스크린샷 위치, 실패 사유를 요약한다.",
                tool_hint="playwright",
                success_criteria="사용자가 점검 결과와 증거를 이해할 수 있음",
                on_failure={"type": "ask_user", "resume": "same_step"},
            ),
        ]
    return [
        WorkflowStep(
            id="confirm_target_and_constraints",
            type="llm",
            name="대상과 조건 확인",
            description="자동화할 URL, 입력값, 제한 조건, 사용자 승인 필요 여부를 확인한다.",
            success_criteria="실행 대상과 제한 조건이 명확함",
            on_failure={"type": "ask_user", "resume": "same_step"},
        ),
        WorkflowStep(
            id="open_page_and_check_state",
            type="browser_action",
            name="웹페이지 열기와 상태 확인",
            description="Playwright로 대상 웹페이지를 열고 로그인/권한/차단 상태를 확인한다.",
            tool_hint="playwright",
            actions=[
                {"type": "navigate", "target": "configured_url"},
                {"type": "assert", "kind": "page_state_readable"},
                {"type": "screenshot", "label": "initial_state"},
            ],
            success_criteria="페이지 상태를 읽고 다음 동작 가능 여부를 판단함",
            on_failure={
                "type": "manual_handoff",
                "resume": "same_step",
                "prompt": "로그인, 캡차, 권한 확인 등 수동 처리를 완료한 뒤 재개하세요.",
            },
        ),
        WorkflowStep(
            id="perform_allowed_browser_actions",
            type="browser_action",
            name="조건에 맞는 작업 수행",
            description="허용된 범위 안에서 클릭, 입력, 제출 등 필요한 웹 조작을 수행한다.",
            tool_hint="playwright",
            actions=[
                {"type": "click", "target": "configured_control"},
                {"type": "type", "target": "configured_input", "source": "configured_value"},
                {"type": "assert", "kind": "requested_action_completed"},
            ],
            success_criteria="요청된 웹 작업이 완료되거나 실패 사유가 확인됨",
            on_failure={
                "type": "retry",
                "max_attempts": 1,
                "then": {"type": "ask_user", "resume": "same_step"},
            },
        ),
        WorkflowStep(
            id="verify_and_report_result",
            type="llm",
            name="결과 검증과 보고",
            description="화면 상태를 다시 확인하고 완료/스킵/오류를 사용자에게 요약한다.",
            tool_hint="playwright",
            success_criteria="사용자가 결과와 남은 조치를 이해할 수 있음",
            on_failure={"type": "ask_user", "resume": "same_step"},
        ),
    ]


def _add_playwright_hints(flow: list[WorkflowStep]) -> list[WorkflowStep]:
    next_flow: list[WorkflowStep] = []
    for step in flow:
        if step.tool_hint:
            next_flow.append(step)
            continue
        text = f"{step.name}\n{step.description}".casefold()
        if _looks_like_web_automation(text):
            next_flow.append(step.model_copy(update={"tool_hint": "playwright"}))
        else:
            next_flow.append(step)
    return next_flow


def _task_goal_from_draft(draft: AgentDraft) -> str:
    if draft.description and draft.description.strip():
        return draft.description.strip()
    if draft.name and draft.name.strip():
        return f"{draft.name.strip()} 실행"
    return "Agent scheduled run"


def _looks_like_manual_timing(user_message: str) -> bool:
    text = user_message.casefold()
    return any(
        marker in text
        for marker in (
            "필요할 때만",
            "필요할때만",
            "부를 때만",
            "부를때만",
            "수동",
            "manual",
            "one-off",
            "one off",
        )
    )


def _collapse_blank_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    collapsed: list[str] = []
    blank_seen = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and blank_seen:
            continue
        collapsed.append(line)
        blank_seen = is_blank
    return "\n".join(collapsed)


def hash_user_message(user_message: str) -> str:
    sample = user_message[:80].encode("utf-8", errors="replace")
    return hashlib.sha256(sample).hexdigest()[:16]


def summarize_draft_delta(before: AgentDraft, after: AgentDraft) -> str:
    before_data = before.model_dump()
    after_data = after.model_dump()
    changed = [
        key
        for key, value in after_data.items()
        if before_data.get(key) != value
    ]
    if not changed:
        return "no_change"
    summaries: list[str] = []
    for key in changed:
        before_value = before_data.get(key)
        after_value = after_data.get(key)
        if isinstance(before_value, list) or isinstance(after_value, list):
            summaries.append(f"{key}:{len(before_value or [])}->{len(after_value or [])}")
        else:
            summaries.append(f"{key}:updated")
    return ",".join(summaries)
