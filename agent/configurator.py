"""Stateful Agent Builder Configurator helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, replace
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
from agent.capability_registry import ToolVerification, verify_declared_mcp_ids

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

7. Adding an `approval_gate` or `manual_handoff` step is correct whenever
   rule 12, 13, or 14 calls for one — keep adding them there. But if the
   flow you are building ALSO carries a schedule (rule 6 populated
   `task_draft.schedule`), that gate has a cost the user has not been told:
   the runtime parks the run and waits for a human on every single fire,
   whether or not anyone is there to answer, and because a schedule skips a
   fire while the previous run is still active, one unanswered wait then
   silently skips every fire after it too. A user who asked for something
   unattended and got this gate believes they built an unattended agent;
   they actually built a daily chore that stalls the first time nobody is
   watching.

   Say this plainly, in the user's language, in the SAME turn you add the
   gate to a scheduled flow — not as a footnote later. For example:
   "이 단계 때문에 이 에이전트는 완전히 무인으로 돌지 않고, 실행마다 멈춰서
   사용자의 확인을 기다립니다. 응답이 없으면 다음 일정도 계속 건너뜁니다."

   Then offer only the choices that actually exist:
     - keep the gate and accept the agent is not unattended (a human must
       answer it every run), or
     - drop the gate.

   Do NOT say a permission rule, standing approval, or policy setting can
   make an `approval_gate` or `manual_handoff` step skip itself — nothing
   does. That class of step always stops and waits for a human, on every
   run, until it is removed from the flow. Telling the user a permission
   grant "fixes" this is a false statement, which is worse than saying
   nothing at all.

   Do not raise this warning when there is no schedule: a gate in a
   manually-run agent is expected and normal, and repeating the warning on
   every turn is noise, not safety.

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


def _step_id_for_script(script_name: str) -> str:
    """A step id for a script proposal that never named one.

    ``ScriptRequest.step_id`` is optional in the draft schema, so the model can
    ask for a script without saying which step runs it. That must not decide
    whether approval wires the script in — the id is bookkeeping, the wiring is
    the promise the user read.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", script_name.strip().casefold()).strip("_")
    return f"run_{slug}"[:64] if slug else "run_script"


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

        When no step carries that id yet, the step is *created* here rather
        than skipped. Prompt rule 2d tells the Configurator to keep the shell
        step out of ``flow`` until the script exists ("never invent a
        script_id"), so "no matching step" is the ordinary case, not the odd
        one — and the reply the user read said the script would be wired into
        the workflow on approval. Filling in nothing left that promise unkept:
        the conversation went on referring to a step that was in no draft, and
        the agent committed without the work it was built to do.
        """
        script_id = str(script.get("id") or "").strip()
        if not script_id:
            return self.current_draft

        script_name = str(script.get("name") or request_name).strip()
        target = (step_id or "").strip() or _step_id_for_script(script_name)
        flow = self.current_draft.flow
        matched = any((step.id or "").strip() == target for step in flow)
        if matched:
            flow = [
                step.model_copy(update={"script_id": script_id})
                if (step.id or "").strip() == target
                else step
                for step in flow
            ]
        else:
            # Head of the flow: a shell step reads, and the llm steps that
            # judge what it read are already written to run after it.
            flow = [
                WorkflowStep(
                    id=target,
                    type="shell",
                    name=(script_name or target)[:120],
                    description=str(script.get("description") or "").strip()[:2000],
                    script_id=script_id,
                ),
                *flow,
            ]
        remaining = [
            request
            for request in self.current_draft.script_requests
            if request.name.strip().casefold() != request_name.strip().casefold()
        ]
        self.current_draft = self.current_draft.model_copy(
            update={"flow": flow, "script_requests": remaining}
        )

        description = str(script.get("description") or "").strip()
        suffix = f" — {description}" if description else ""
        line = f"  - {script_id}: {script_name}{suffix}"
        step_note = f" It is already the script_id of step '{target}'."
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
        flow_before_enrichment = self.current_draft.flow
        self.current_draft, self.task_draft = enrich_draft_from_user_intent(
            self.current_draft,
            previous_draft=before,
            task_draft=self.task_draft,
            user_message=user_message,
        )
        gate_disclosure = _deterministic_gate_disclosure(
            before_flow=flow_before_enrichment,
            after_flow=self.current_draft.flow,
            task_draft=self.task_draft,
        )
        if gate_disclosure is not None:
            parsed = replace(
                parsed,
                assistant_message=_with_disclosure(parsed.assistant_message, gate_disclosure),
            )
        # After enrichment, so both what the model declared and what
        # `_ensure_playwright_tool` / `_ensure_app_action_tool` added are
        # checked by the same rule. Runs every turn, not only turns that
        # carried a draft block: a tool the model declared two turns ago is
        # still in `current_draft` and still on its way to `tools_json`.
        # `dropped` is empty on the turns after the first, so the user is told
        # once rather than in every reply.
        self.current_draft, dropped_tools = _drop_unverifiable_tools(self.current_draft)
        if dropped_tools:
            parsed = replace(
                parsed,
                assistant_message=_with_disclosure(
                    parsed.assistant_message, _unverified_tool_disclosure(dropped_tools)
                ),
            )
        if parsed.draft is not None:
            # Only when this turn actually put a draft on the screen. Saying
            # "there are no workflow steps" during a turn that proposed no
            # draft at all (a greeting, a question) would be noise.
            flow_disclosure = _empty_flow_disclosure(self.current_draft.flow)
            if flow_disclosure is not None:
                parsed = replace(
                    parsed,
                    assistant_message=_with_disclosure(
                        parsed.assistant_message, flow_disclosure
                    ),
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


# Step types that `_wait_for_user_step` (task_orchestrator.py) dispatches
# unconditionally: it parks the run and waits for a human on every single
# occurrence, and nothing in the policy engine consults or bypasses that
# dispatch. Any code that inserts one of these into a flow silently creates a
# run that cannot finish unattended.
_GATE_STEP_TYPES = {"approval_gate", "manual_handoff"}


def _deterministic_gate_disclosure(
    *,
    before_flow: list[WorkflowStep],
    after_flow: list[WorkflowStep],
    task_draft: TaskDraft | None,
) -> str | None:
    """Warn the user the turn a park-for-human step lands in a scheduled flow.

    `_join_request_steps` and `_install_app_steps` add an `approval_gate` step
    ahead of a sensitive external action -- correctly,
    per rules 13/14 below. But that step type is dispatched unconditionally
    every run (see `_GATE_STEP_TYPES` above), and it was inserted here rather
    than written by the model, so the model never saw it and never had a
    chance to mention its cost in its own reply. Without this check the
    disclosure in rule 7 only ever fires for gates the model wrote itself;
    a gate this file adds behind the scenes would stay silent and the exact
    defect this initiative exists to close -- a user who asked for something
    unattended getting a daily chore with no warning -- would reappear.

    Returns ``None`` (raises nothing, warns nothing) when there is no
    schedule: a gate in a manually-run agent is expected, and warning about
    it anyway would be noise, not safety.
    """

    schedule = (task_draft.schedule or "").strip() if task_draft is not None else ""
    if not schedule:
        return None

    before_ids = {_step_identity(step) for step in before_flow}
    new_gate_steps = [
        step
        for step in after_flow
        if step.type in _GATE_STEP_TYPES and _step_identity(step) not in before_ids
    ]
    if not new_gate_steps:
        return None

    names = ", ".join(step.name or step.id or step.type for step in new_gate_steps)
    return (
        f"참고: 방금 추가된 '{names}' 단계는 실행마다 멈춰서 사용자의 확인/승인을 "
        f"기다리는 단계입니다. 일정('{schedule}')대로 자동 실행되더라도 이 단계에서는 "
        "매번 사람이 응답할 때까지 멈추고, 응답이 없으면 다음 일정도 계속 건너뜁니다 — "
        "완전한 무인 실행은 아니게 됩니다. 이 승인 단계를 유지하고 무인 실행은 포기하시거나, "
        "이 단계를 없애 주세요. 어떤 권한(permission) 설정도 이 승인 단계 자체를 "
        "건너뛰게 하지는 못합니다."
    )


def _empty_flow_disclosure(flow: list[WorkflowStep]) -> str | None:
    """Say plainly that the draft has no runnable workflow yet.

    This file used to answer an empty flow by writing one: a keyword classifier
    decided the agent "looked like" Naver cafe / Naver note / generic web
    automation and `enrich_draft_from_user_intent` dropped a six-step template
    into the draft. The user then read a workflow they had never asked for,
    described in confident detail, and had no way to tell it apart from the
    model's own work. The global rule this project runs on is that when the
    model does not produce something the server does not manufacture a
    rule-based substitute and present it as the model's -- so the flow now
    stays empty, and this says so.

    Commit is already refused for an empty flow by `_validate_commit_draft`
    (`routes/agents.py`), which is a 422 the user meets *after* they press
    save. This is the same fact told during the conversation, where it can
    still be acted on.

    A flow of nothing but bare "Step 1" placeholders gets its own wording: it
    is not empty, so it clears the commit check, but nothing in it says what to
    do or what success means, and that is exactly the shape the deleted
    templates used to overwrite.
    """

    if flow and not all(_is_placeholder_step(step) for step in flow):
        return None

    if not flow:
        return (
            "참고: 이 초안에는 아직 워크플로 단계가 하나도 없습니다. 단계가 없으면 "
            "에이전트를 저장할 수 없고, 서버가 대신 단계를 만들어 넣지도 않습니다. "
            "어떤 순서로 무엇을 하면 되는지 알려주시면 그대로 단계로 옮기겠습니다."
        )
    return (
        "참고: 이 초안의 워크플로 단계에는 아직 이름만 있고 실제 내용(무엇을 하는지, "
        "무엇이 성공인지)이 없습니다. 서버가 그 내용을 임의로 채우지 않으므로, "
        "각 단계에서 할 일과 성공 기준을 알려주시면 그대로 반영하겠습니다."
    )


def _drop_unverifiable_tools(
    draft: AgentDraft,
) -> tuple[AgentDraft, list[ToolVerification]]:
    """Remove declared tools that name nothing existing on this machine.

    `draft.tools` is a promise, not a wish list. It is written into
    `agents.tools_json` at commit (`routes/agents.py` builder_commit), read
    back by `agent_store.simulate_run_from_workflow` to tell the user
    "this step would call X", shown in the app as what the agent will need
    permission for, and — since C6-2 — used to decide which MCP servers a
    Claude run is actually given. An entry naming a server that is not
    configured anywhere makes all four of those lie in the same direction:
    the agent looks more capable than it is, and the user only finds out at
    run time, when the step does nothing.

    Two options were on the table: drop the entry, or keep it flagged
    ``verified: false``. Dropping is what happens here, because

    1. `AgentToolDraft` (`agent/agent_models.py`) has no such field and
       pydantic ignores unknown keys, so a flag would need a schema change
       plus matching work in every consumer above. Until each one rendered
       the flag, an unverified tool would be displayed exactly like a real
       one — the fabricated promise, with extra steps.
    2. The flag would still have to be enforced at run time to be worth
       anything, and the run-time answer is already "there is no such server";
       carrying the declaration forward only postpones the same sentence.

    Dropping is not the same as hiding: every removal is named back to the
    user in the same turn by `_unverified_tool_disclosure`, while they are
    still in the conversation that can fix it — install the server, or plan
    without it.

    Returns the draft (unchanged object when nothing was dropped) and the
    verdicts for what was removed.
    """
    if not draft.tools:
        return draft, []

    verdicts = {
        verdict.mcp_id: verdict
        for verdict in verify_declared_mcp_ids(tool.mcp_id for tool in draft.tools)
    }
    kept: list[AgentToolDraft] = []
    dropped: list[ToolVerification] = []
    dropped_ids: set[str] = set()
    for tool in draft.tools:
        verdict = verdicts.get(tool.mcp_id.strip())
        if verdict is not None and not verdict.verified:
            if verdict.mcp_id not in dropped_ids:
                dropped_ids.add(verdict.mcp_id)
                dropped.append(verdict)
            continue
        kept.append(tool)

    if not dropped:
        return draft, []
    logger.warning(
        "builder_dropped_unverified_tools mcp_ids=%s",
        ",".join(verdict.mcp_id for verdict in dropped),
    )
    return draft.model_copy(update={"tools": kept}), dropped


def _unverified_tool_disclosure(dropped: list[ToolVerification]) -> str:
    """Name every tool that was removed from the draft, and why."""
    names = ", ".join(f"'{verdict.mcp_id}'" for verdict in dropped)
    return (
        f"참고: {names} 도구는 이 초안에서 제외했습니다. 이 서버(그리고 이 Mac의 CLI 설정)에 "
        "해당 이름의 MCP 서버가 등록되어 있지 않아서, 선언만으로는 실행되지 않기 때문입니다. "
        "초안에 남겨 두면 저장 후 앱에는 '이 에이전트가 쓸 수 있는 도구'로 보이지만 실행 시에는 "
        "아무 일도 일어나지 않습니다. 이 도구가 꼭 필요하면 먼저 CLI에 MCP 서버를 설치·등록한 뒤 "
        "다시 말씀해 주시고, 아니면 이 도구 없이 가능한 방법으로 워크플로를 다시 잡겠습니다."
    )


def _with_disclosure(message: str, disclosure: str) -> str:
    message = (message or "").strip()
    if not message:
        return disclosure
    if disclosure in message:
        return message
    return f"{message}\n\n{disclosure}"


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

    What it does **not** do any more is write the workflow. An empty ``flow``
    used to be answered with a keyword-selected six-step template
    (`_naver_cafe_flow` / `_naver_note_flow` / `_generic_web_automation_flow`,
    all deleted): the user was handed a detailed workflow they never asked for
    and could not distinguish from the model's own output. An empty flow now
    stays empty and `_empty_flow_disclosure` tells the user so in the same
    turn. `_add_playwright_hints` remains because it only tags steps the model
    actually wrote -- that is normalisation, not authorship.
    """

    raw_text = _raw_intent_text(user_message, draft)
    text = raw_text.casefold()
    prefers_app_automation = _prefers_app_automation(text)
    is_web = _looks_like_web_automation(text) and not prefers_app_automation
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
    """Give back a field the model dropped. Do not invent one it never wrote.

    Three keyword-driven writers used to live here -- `_fallback_agent_name`,
    `_fallback_agent_description` and `_fallback_system_prompt` -- and they
    were the flow-template defect one field further down. A draft with no
    name became "Naver Cafe Agent"; a draft with no description got a
    paragraph promising "duplicate checks and user handoff for login or
    captcha"; a draft with no `system_prompt` was given a whole set of
    operating instructions, written by a keyword match, that then governed the
    agent at runtime. All three are text the user reads as the model's work,
    and none of them were.

    What is left is preservation, not authorship: if the model returned a
    draft with a field blank and the *previous* draft -- the one the user has
    already read on screen -- had a value, that value is carried forward. The
    model dropping a field mid-conversation must not silently erase what the
    user already approved (that is what `_preserve_flow_for_additive_request`
    does for the flow). When there is nothing to carry forward the field stays
    empty and `_validate_commit_draft` (`routes/agents.py`) refuses the commit
    with 422 naming the missing field, which is the honest outcome.

    `provider_id` is deliberately still defaulted. It selects which configured
    backend runs the agent -- a setting, not content the user reads as having
    been written for them -- and the chain here is the user's own words
    (`_infer_provider_id`), then the previous draft, then the app-wide
    default. Changing that default to "refuse the commit" is a product
    decision about the builder's entry experience, not a fabrication fix.
    """

    updates: dict[str, Any] = {}

    if previous_draft is not None:
        if not (draft.name or "").strip() and (previous_draft.name or "").strip():
            updates["name"] = previous_draft.name
        if (
            not (draft.description or "").strip()
            and (previous_draft.description or "").strip()
        ):
            updates["description"] = previous_draft.description
        if not draft.system_prompt.strip() and previous_draft.system_prompt.strip():
            updates["system_prompt"] = previous_draft.system_prompt
    if draft.provider_id is None:
        provider_id = _infer_provider_id(user_message)
        if provider_id is None and previous_draft is not None:
            provider_id = previous_draft.provider_id
        updates["provider_id"] = provider_id or "openai"

    if not updates:
        return draft
    return draft.model_copy(update=updates)


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
    """What the user asked for — never what we already concluded from it.

    `draft.tools` is deliberately absent. Tools are an *output* of this
    classification, so feeding their text back in makes the pass self-
    confirming: one injected Playwright tool whose examples said "네이버 카페
    게시판 열기" was enough to make the next call classify a completely
    unrelated agent as Naver-cafe automation and overwrite its workflow.
    """
    parts = [
        user_message,
        draft.name or "",
        draft.description or "",
        draft.system_prompt or "",
    ]
    for step in draft.flow:
        parts.extend([step.name, step.description, step.tool_hint or ""])
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
            # "메시지"/"message" are deliberately NOT markers. They appear in
            # ordinary system-prompt prose ("에러 메시지를 요약해줘") and were
            # measured misclassifying non-web agents as browser automation.
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
    # "매일 오전 9시" is how people actually write it, and the period word sat
    # between "매일" and the hour, so the digit was never reached and the turn
    # came back with no schedule at all. `routes/agents.py` already reads the
    # same phrasing; the two disagreeing is what left `session.task_draft`
    # empty during a turn whose commit went on to create the schedule anyway.
    match = re.search(r"매일\s*(오전|오후|아침|저녁|밤|새벽)?\s*([0-2]?\d)[:시]\s*([0-5]\d)?", text)
    if match:
        period = match.group(1)
        hour = int(match.group(2))
        minute = match.group(3) or "00"
        # 오후 12시 is midday and 오전 12시 is midnight: shifting both by 12
        # would put one of them half a day out.
        if period in {"오후", "저녁", "밤"} and hour < 12:
            hour += 12
        elif period in {"오전", "아침", "새벽"} and hour == 12:
            hour = 0
        return f"daily {hour:02d}:{minute}"
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
                    # Neutral by design: these examples are shown to the user and
                    # must not name a site the agent has nothing to do with.
                    user_examples=["웹페이지 열기", "버튼 클릭", "입력 필드 채우기"],
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


def _is_placeholder_step(step: WorkflowStep) -> bool:
    """A step with a bare "Step 1" name and nothing else in it.

    This used to decide which flows were eligible to be overwritten by a
    template. Nothing is overwritten any more; the only caller left is
    `_empty_flow_disclosure`, which uses it to tell the user their workflow
    has names but no content.
    """
    return (
        bool(re.fullmatch(r"step\s*\d+", step.name.strip(), flags=re.IGNORECASE))
        and not step.description.strip()
        and not (step.tool_hint or "").strip()
        and not step.success_criteria.strip()
    )


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


# Public names for the two timing judgements the commit path has to make as
# well. Commit reads the conversation directly (see
# `_conversation_schedule_phrase` in routes/agents.py) and must reach the same
# verdict as the turn that produced it — a second, slightly different copy of
# "did the user ask for this to run by itself?" is how the builder started
# promising schedules it never created.
looks_like_manual_timing = _looks_like_manual_timing
task_goal_from_draft = _task_goal_from_draft


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
