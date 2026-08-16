"""Builder-runtime contract checks for a workflow, run at authoring time.

The runtime already refuses to guess: `browser_action_adapter._requires_user_target`
stops a run with a user checkpoint when an action still carries a placeholder
target. That honest stop is the last line of defence and stays. The problem it
cannot solve is timing — by then the workflow is already saved and scheduled,
and the first anyone hears of it is a run parked at 3am.

This module moves the same judgement to commit time. It is deliberately a
*reporting* layer:

* it never mutates the flow it is given;
* it never decides an HTTP status code or a user-visible policy;
* it returns findings, and the caller (the commit gate) decides what a
  ``blocking`` finding means for its own endpoint.

Keeping that boundary means the same analysis can back a 422 on commit, a
warning banner on the dashboard, and a diagnostic endpoint without any of them
having to agree on a response shape first.

Single source of truth: placeholder detection is imported from
``browser_action_adapter._is_placeholder`` rather than re-implemented. Two
copies would drift, and the copy that drifted would either block commits the
runtime is happy to run or wave through the exact workflow the runtime parks.

The same rule decides the device side. An app step is judged by
``app_action_executor.app_action_gap`` — the function the adapter itself calls
before running an action — and not by a field scan, because which field is a
target depends on the action: ``read_screen`` never reads its ``target`` and a
placeholder there is harmless, while ``verify_launch`` parks without a real
package. A second opinion here would refuse commits the device would have run.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .app_action_executor import (
    REASON_ACTIONS_MISSING,
    REASON_UNSUPPORTED_ACTION,
    app_action_gap,
    app_actions_missing_gap,
)
from .browser_action_adapter import (
    PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
    _is_placeholder,
)

SEVERITY_BLOCKING = "blocking"
SEVERITY_WARNING = "warning"

CODE_UNRESOLVED_BROWSER_TARGET = "unresolved_browser_target"
CODE_BROWSER_RUNTIME_UNAVAILABLE = "browser_runtime_unavailable"
CODE_UNKNOWN_STEP_REFERENCE = "unknown_step_reference"
CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE = "possible_unknown_step_reference"
CODE_UNRESOLVED_APP_TARGET = "unresolved_app_target"
CODE_UNSUPPORTED_APP_ACTION = "unsupported_app_action"
CODE_APP_ACTIONS_MISSING = "app_actions_missing"

BROWSER_STEP_TYPES: frozenset[str] = frozenset({"browser_action"})

# The four types `task_orchestrator._is_app_action_workflow_type` dispatches to
# the device adapter (`task_orchestrator.py:3064-3070`). All four park the run
# on an unresolved target exactly as a browser step does, so all four belong to
# a gate whose stated purpose is "do not schedule a guaranteed 3am stall".
APP_STEP_TYPES: frozenset[str] = frozenset(
    {"app_action", "android_action", "mobile_action", "device_action"}
)

# Which finding a runtime stop is reported as. Everything that is not "this
# action type does not exist here" or "there are no actions" is an unresolved
# target: a package, an APK path, a screen text or a key the author has to
# supply.
_APP_GAP_CODES: dict[str, str] = {
    REASON_UNSUPPORTED_ACTION: CODE_UNSUPPORTED_APP_ACTION,
    REASON_ACTIONS_MISSING: CODE_APP_ACTIONS_MISSING,
}

# The action fields the runtime reads as a target (see
# `browser_action_adapter._requires_user_target`). Checked in this order so the
# finding list is stable between runs.
_ACTION_TARGET_FIELDS: tuple[str, ...] = ("url", "selector", "target", "text", "value")

# Author-facing prose. `name`/`description` exist on every step type;
# `instruction`/`observation`/`success_criteria` are llm-step text that
# routinely names other steps.
_PROSE_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "instruction",
    "observation",
    "success_criteria",
)

# Confirmed references — an author writing one of these means a step id and
# nothing else, so an id that does not exist is a definite error.
_TEMPLATE_REFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\{\{\s*steps\.([A-Za-z0-9_-]+)"),
    re.compile(r"\{\{\s*([A-Za-z0-9_-]+)\.output\s*\}\}"),
    re.compile(r"\bgoto(?:_step)?\s*:\s*([A-Za-z0-9_-]+)"),
)

# Inferred references — snake_case tokens that *might* be a step id. Warning
# only: `pubspec_yaml` in a sentence is not an error, and a false positive that
# blocks a commit is worse than one that prints a note.
_SNAKE_TOKEN = re.compile(r"(?<![\w./-])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![\w./-])")
_QUOTED_SPAN = re.compile(r"`([^`\n]{1,120})`|\"([^\"\n]{1,120})\"|'([^'\n]{1,120})'")
_STEP_WORD = re.compile(r"스텝|단계|step", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"[.!?。\n]+")


@dataclass(frozen=True)
class ContractFinding:
    """One contract problem found in a workflow.

    ``severity`` is the module's judgement of certainty, not a policy:
    ``blocking`` means "this workflow cannot run as written", ``warning`` means
    "a human should look". What either one does to a request is the caller's
    decision.
    """

    severity: str
    code: str
    step_id: str | None
    detail: dict[str, Any] = field(default_factory=dict)
    ask: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContractReport:
    findings: list[ContractFinding] = field(default_factory=list)

    @property
    def blocking(self) -> list[ContractFinding]:
        return [f for f in self.findings if f.severity == SEVERITY_BLOCKING]

    @property
    def warnings(self) -> list[ContractFinding]:
        return [f for f in self.findings if f.severity == SEVERITY_WARNING]

    @property
    def has_blocking(self) -> bool:
        return any(f.severity == SEVERITY_BLOCKING for f in self.findings)

    def by_code(self, code: str) -> list[ContractFinding]:
        return [f for f in self.findings if f.code == code]

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "blocking": [f.to_dict() for f in self.blocking],
            "warnings": [f.to_dict() for f in self.warnings],
        }


def analyze_workflow(
    flow: Any,
    *,
    browser_readiness: dict[str, Any] | None = None,
) -> ContractReport:
    """Report what would stop this workflow at runtime, before it is saved.

    ``flow`` is a list of step dicts (normalized or raw — both ``type`` and
    ``step_type`` are read). ``browser_readiness`` is the payload from
    ``browser_action_adapter.get_browser_runtime_readiness``; pass ``None`` when
    it is unknown, and no readiness claim is made either way.

    The flow is never modified.
    """
    steps = _steps(flow)
    findings: list[ContractFinding] = []

    findings.extend(_check_placeholder_targets(steps))
    findings.extend(_check_app_actions(steps))
    findings.extend(_check_browser_runtime(steps, browser_readiness))
    findings.extend(_check_step_references(steps))

    return ContractReport(findings=findings)


def _steps(flow: Any) -> list[dict[str, Any]]:
    if not isinstance(flow, list):
        return []
    return [step for step in flow if isinstance(step, dict)]


def _step_type(step: dict[str, Any]) -> str:
    raw = step.get("type") or step.get("step_type") or ""
    return str(raw).strip().lower()


def _step_id(step: dict[str, Any]) -> str | None:
    raw = step.get("id")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _check_placeholder_targets(steps: list[dict[str, Any]]) -> list[ContractFinding]:
    """Check 1 — a browser action still pointing at `configured_*`.

    Blocking: the runtime adapter will park every such run waiting for a human,
    so saving it schedules a guaranteed stall.
    """
    findings: list[ContractFinding] = []
    for step in steps:
        if _step_type(step) not in BROWSER_STEP_TYPES:
            continue
        actions = step.get("actions")
        if not isinstance(actions, list):
            continue
        step_id = _step_id(step)
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            for field_name in _ACTION_TARGET_FIELDS:
                value = action.get(field_name)
                if not isinstance(value, str):
                    continue
                text = value.strip()
                if not text or not _is_placeholder(text):
                    continue
                findings.append(
                    ContractFinding(
                        severity=SEVERITY_BLOCKING,
                        code=CODE_UNRESOLVED_BROWSER_TARGET,
                        step_id=step_id,
                        detail={
                            "step_id": step_id,
                            "action_index": action_index,
                            "action_type": str(action.get("type") or "").strip().lower(),
                            "field": field_name,
                            "value": text,
                        },
                        ask=(
                            f"'{step_id or '브라우저 단계'}' 단계의 {action_index + 1}번째 action에 "
                            f"실제 {field_name} 값이 필요합니다. 지금은 '{text}' 자리표시자라서 "
                            "실행하면 사용자 입력 대기로 멈춥니다."
                        ),
                    )
                )
    return findings


def _check_app_actions(steps: list[dict[str, Any]]) -> list[ContractFinding]:
    """Check 1b — a device action the adapter would stop on.

    Blocking, for the same reason the browser check is: the adapter returns
    ``waiting_for_user`` for every one of these, so saving it schedules a run
    that parks. The judgement is the adapter's own ``app_action_gap``; this
    function only decides where the finding goes and how it is phrased.

    Only payload-shaped problems appear here. Whether an APK file exists or the
    text is on screen is not knowable at authoring time, and the adapter keeps
    those checks to itself.
    """
    findings: list[ContractFinding] = []
    for step in steps:
        if _step_type(step) not in APP_STEP_TYPES:
            continue
        step_id = _step_id(step)
        actions = step.get("actions")
        if not isinstance(actions, list) or not actions:
            findings.append(_app_finding(step_id, None, None, app_actions_missing_gap()))
            continue
        for action_index, action in enumerate(actions):
            gap = app_action_gap(action)
            if gap is None:
                continue
            findings.append(_app_finding(step_id, action_index, action, gap))
    return findings


def _app_finding(
    step_id: str | None,
    action_index: int | None,
    action: Any,
    gap: Any,
) -> ContractFinding:
    action_type = (
        str(action.get("type") or "").strip().lower()
        if isinstance(action, dict)
        else ""
    )
    where = (
        f"'{step_id or '앱 조작 단계'}' 단계"
        if action_index is None
        else f"'{step_id or '앱 조작 단계'}' 단계의 {action_index + 1}번째 action"
    )
    detail: dict[str, Any] = {
        "step_id": step_id,
        "action_type": action_type,
        "field": gap.field,
        "value": gap.value,
        # The `wait_reason` the run would have carried. A client that has seen
        # the parked run recognises the same string here.
        "reason": gap.reason,
    }
    if action_index is not None:
        detail["action_index"] = action_index
    return ContractFinding(
        severity=SEVERITY_BLOCKING,
        code=_APP_GAP_CODES.get(gap.reason, CODE_UNRESOLVED_APP_TARGET),
        step_id=step_id,
        detail=detail,
        ask=f"{where}: {gap.message}",
    )


def _check_browser_runtime(
    steps: list[dict[str, Any]],
    browser_readiness: dict[str, Any] | None,
) -> list[ContractFinding]:
    """Check 2 — the server has no browser runtime for the browser steps.

    Warning, not blocking: the workflow itself is fine, and the fix is a server
    install the author usually cannot perform. Saying so is honest; refusing
    the commit would hold their work hostage to someone else's machine.
    """
    if not isinstance(browser_readiness, dict):
        return []
    if browser_readiness.get("ready"):
        return []

    browser_step_ids = [
        _step_id(step) for step in steps if _step_type(step) in BROWSER_STEP_TYPES
    ]
    if not browser_step_ids:
        return []

    install_command = (
        str(browser_readiness.get("install_command") or "").strip()
        or PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND
    )
    message = str(browser_readiness.get("message") or "").strip()

    # Not every unready server is one download away. Since the browser can now
    # be an installed Chrome, "not ready" also covers a setting this machine
    # cannot honour (Chrome-only on a machine with no Chrome, a shared profile
    # that has moved). Telling that operator to run `playwright install
    # chromium` would be a fix for a problem they do not have, so the probe's
    # own message leads and the command is only offered when a download is
    # genuinely what is missing.
    needs_download = browser_readiness.get("install_required") is not False
    if needs_download:
        ask = (
            "이 서버에는 브라우저 런타임이 없습니다. 준비되기 전까지 브라우저 단계는 "
            f"실행 시 사용자 개입 대기로 멈춥니다. 설치 명령: {install_command}"
        )
    else:
        ask = (
            "이 서버의 브라우저 설정으로는 브라우저 단계를 실행할 수 없습니다. "
            f"{message} 대시보드의 브라우저 런타임 설정에서 바꾸세요."
        ).strip()

    return [
        ContractFinding(
            severity=SEVERITY_WARNING,
            code=CODE_BROWSER_RUNTIME_UNAVAILABLE,
            step_id=None,
            detail={
                "install_command": install_command,
                "message": message,
                "step_ids": [sid for sid in browser_step_ids if sid],
                "playwright_python": bool(browser_readiness.get("playwright_python")),
                "chromium_executable": bool(browser_readiness.get("chromium_executable")),
                "install_required": needs_download,
            },
            ask=ask,
        )
    ]


def _check_step_references(steps: list[dict[str, Any]]) -> list[ContractFinding]:
    """Check 3 — prose naming a step that does not exist.

    Two tiers on purpose. A template reference (`{{steps.x}}`, `{{x.output}}`,
    `goto:x`) can only mean a step id, so a missing one blocks. A bare
    snake_case token only *looks* like an id — it is just as likely a filename
    — so it warns and never blocks.
    """
    step_ids = {sid for sid in (_step_id(step) for step in steps) if sid}
    findings: list[ContractFinding] = []
    seen: set[tuple[str, str | None, str, str]] = set()

    for step in steps:
        step_id = _step_id(step)
        for field_name in _PROSE_FIELDS:
            text = step.get(field_name)
            if not isinstance(text, str) or not text.strip():
                continue

            confirmed = _template_references(text)
            for reference in sorted(confirmed):
                if reference in step_ids:
                    continue
                key = (CODE_UNKNOWN_STEP_REFERENCE, step_id, field_name, reference)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    ContractFinding(
                        severity=SEVERITY_BLOCKING,
                        code=CODE_UNKNOWN_STEP_REFERENCE,
                        step_id=step_id,
                        detail={
                            "step_id": step_id,
                            "field": field_name,
                            "reference": reference,
                            "known_step_ids": sorted(step_ids),
                        },
                        ask=(
                            f"'{step_id or '이 단계'}'의 {field_name}이(가) 존재하지 않는 단계 "
                            f"'{reference}'을(를) 가리킵니다. 단계 id를 고치거나 해당 단계를 추가하세요."
                        ),
                    )
                )

            for reference in sorted(_inferred_references(text)):
                if reference in step_ids or reference in confirmed:
                    continue
                key = (
                    CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE,
                    step_id,
                    field_name,
                    reference,
                )
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    ContractFinding(
                        severity=SEVERITY_WARNING,
                        code=CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE,
                        step_id=step_id,
                        detail={
                            "step_id": step_id,
                            "field": field_name,
                            "reference": reference,
                            "known_step_ids": sorted(step_ids),
                        },
                        ask=(
                            f"'{step_id or '이 단계'}'의 {field_name}에 나오는 '{reference}'이(가) "
                            "단계 id처럼 보이지만 이 워크플로에는 없습니다. 단계 이름이라면 고치고, "
                            "그냥 이름/파일명이라면 무시해도 됩니다."
                        ),
                    )
                )

    return findings


def _template_references(text: str) -> set[str]:
    references: set[str] = set()
    for pattern in _TEMPLATE_REFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            reference = match.group(1).strip()
            if reference:
                references.add(reference)
    return references


def _inferred_references(text: str) -> set[str]:
    """snake_case tokens that read like a step id.

    Two admission routes, both requiring the author to have singled the token
    out: it is the whole content of a backtick/quote span, or it sits in a
    sentence that talks about steps. A quoted span must match *entirely* so
    `pubspec.yaml` and `lib/foo_bar.dart` never contribute a token.
    """
    references: set[str] = set()

    for match in _QUOTED_SPAN.finditer(text):
        span = next((group for group in match.groups() if group is not None), "")
        candidate = span.strip()
        if _SNAKE_TOKEN.fullmatch(candidate):
            references.add(candidate)

    for sentence in _SENTENCE_SPLIT.split(text):
        if not _STEP_WORD.search(sentence):
            continue
        for match in _SNAKE_TOKEN.finditer(sentence):
            references.add(match.group(1))

    return references
