"""Workflow v2 normalization helpers.

This module is intentionally independent from the task orchestrator. It gives
the runtime and builder a shared, low-risk way to turn legacy workflow JSON into
the structured shape that a future step runner can execute.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


class WorkflowNormalizationError(ValueError):
    """Raised when workflow JSON cannot be normalized safely."""


ALLOWED_STEP_TYPES = {
    "llm",
    # Runs a registered script (see agent.script_store). Deterministic: no
    # model call, no tokens, no permission prompt — vetting happened at
    # registration. Its exit code and output become evidence for later steps,
    # so a failure can escalate to an LLM step via on_failure: goto_step.
    "shell",
    # Leaves a message for whoever comes back. Delivery is the server's own
    # capability rather than a script, so an agent that needs to say something
    # does not first need a notifier written, vetted and registered.
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
}

ALLOWED_ACTION_TYPES = {
    "navigate",
    "click",
    "type",
    "fill",
    "press",
    "wait",
    "assert",
    "extract",
    "screenshot",
    "select",
    "check",
    "uncheck",
    "evaluate",
    "install_app",
    "open_play_store",
    "verify_launch",
    "launch_app",
    "open_app",
    "close_app",
    "read_screen",
    "read_ui",
    "dump_ui",
    "tap_text",
    "wait_text",
    "tap",
    "input_text",
    "type_text",
    "press_key",
}

_STEP_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,79}$")


def normalize_workflow(raw_steps: Any) -> list[dict[str, Any]]:
    """Normalize a workflow list into a minimally executable v2 shape.

    The helper accepts the existing legacy step shape:
    ``name/description/tool_hint/success_criteria/on_failure``.

    It returns JSON-serializable dictionaries with:
    - stable ``id``
    - normalized ``type``
    - structured ``on_failure`` policy object
    - normalized ``actions`` list when provided
    """

    if raw_steps is None:
        return []
    if not isinstance(raw_steps, list):
        raise WorkflowNormalizationError("workflow must be a list")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, raw_step in enumerate(raw_steps, start=1):
        step = normalize_workflow_step(raw_step, index=index, seen_ids=seen_ids)
        seen_ids.add(step["id"])
        normalized.append(step)

    _validate_failure_targets(normalized)
    return normalized


def normalize_workflow_step(
    raw_step: Any,
    *,
    index: int,
    seen_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Normalize one workflow step dictionary."""

    if not isinstance(raw_step, dict):
        raise WorkflowNormalizationError(f"step {index} must be an object")

    step = deepcopy(raw_step)
    seen = seen_ids or set()
    step_id = _normalize_step_id(step.get("id"), index=index, seen_ids=seen)
    step_type = _normalize_step_type(step.get("type") or step.get("step_type"), index)

    step["id"] = step_id
    step["type"] = step_type
    step["name"] = _clean_text(step.get("name")) or f"Step {index}"
    step["description"] = _clean_text(step.get("description"))
    step["tool_hint"] = _clean_optional_text(step.get("tool_hint"))
    step["success_criteria"] = _clean_text(step.get("success_criteria"))
    step["on_failure"] = normalize_failure_policy(step.get("on_failure"))
    step["on_success"] = normalize_success_policy(step.get("on_success"))
    step["actions"] = normalize_actions(step.get("actions"))
    if step_type == "shell":
        script_id = _clean_optional_text(step.get("script_id"))
        if not script_id:
            raise WorkflowNormalizationError(
                f"step {index}: shell steps need a script_id — register the script first"
            )
        step["script_id"] = script_id
        step["script_args"] = [str(item) for item in (step.get("script_args") or [])]
    if step_type == "notify":
        raw = step.get("notify") if isinstance(step.get("notify"), dict) else {}
        title = _clean_optional_text(raw.get("title")) or step["name"]
        # A notification with no title has nothing to show in the inbox, and
        # the step name is always something the author already wrote.
        step["notify"] = {
            "title": title,
            "body": _clean_optional_text(raw.get("body")) or step["description"] or None,
            "level": _normalize_notify_level(raw.get("level")),
        }
    return step


def _normalize_notify_level(raw: Any) -> str:
    value = str(raw or "info").strip().lower()
    return value if value in {"info", "success", "warning", "error"} else "info"


def normalize_failure_policy(raw_policy: Any) -> dict[str, Any]:
    """Normalize legacy or v2 failure policy values.

    Legacy strings map as follows:
    - ``ask_user`` -> ask the user, then resume the same step.
    - ``abort`` -> stop the workflow.
    - ``retry_once`` -> retry the current step once, then abort.
    """

    if raw_policy is None:
        return {"type": "ask_user", "resume": "same_step"}

    if isinstance(raw_policy, str):
        value = raw_policy.strip()
        lowered = value.casefold()
        if lowered == "ask_user":
            return {"type": "ask_user", "resume": "same_step"}
        if lowered == "abort":
            return {"type": "abort"}
        if lowered in {"continue", "skip"}:
            # Independent work in one workflow: three phones, or a cleanup that
            # is allowed to fail. Without this the only ways past a failed step
            # were abort, wait for a human, or jump — and a jump skips whatever
            # sits between here and the target.
            return {"type": "continue"}
        if lowered == "retry_once":
            return {"type": "retry", "max_attempts": 1, "then": {"type": "abort"}}
        if lowered.startswith("goto_step:") or lowered.startswith("goto:"):
            target = value.split(":", 1)[1].strip()
            if not target:
                raise WorkflowNormalizationError("goto failure policy requires a target")
            return {"type": "goto_step", "target_step_id": target}
        raise WorkflowNormalizationError(f"unknown failure policy: {raw_policy}")

    if not isinstance(raw_policy, dict):
        raise WorkflowNormalizationError("failure policy must be a string or object")

    policy = deepcopy(raw_policy)
    raw_type = policy.get("type") or policy.get("action")
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise WorkflowNormalizationError("failure policy type is required")

    policy_type = raw_type.strip().casefold()
    if policy_type == "goto":
        policy_type = "goto_step"
    policy["type"] = policy_type
    policy.pop("action", None)

    if policy_type in {"continue", "skip"}:
        return {"type": "continue"}

    if policy_type == "abort":
        return _normalize_abort_policy(policy)
    if policy_type == "ask_user":
        return _normalize_ask_user_policy(policy)
    if policy_type == "manual_handoff":
        return _normalize_manual_handoff_policy(policy)
    if policy_type == "retry":
        return _normalize_retry_policy(policy)
    if policy_type == "goto_step":
        return _normalize_goto_policy(policy)

    raise WorkflowNormalizationError(f"unknown failure policy type: {raw_type}")


def normalize_actions(raw_actions: Any) -> list[dict[str, Any]]:
    """Normalize a step's executable action list."""

    if raw_actions is None:
        return []
    if not isinstance(raw_actions, list):
        raise WorkflowNormalizationError("actions must be a list")

    actions: list[dict[str, Any]] = []
    for index, raw_action in enumerate(raw_actions, start=1):
        if not isinstance(raw_action, dict):
            raise WorkflowNormalizationError(f"action {index} must be an object")
        action = deepcopy(raw_action)
        raw_type = action.get("type")
        if not isinstance(raw_type, str) or not raw_type.strip():
            raise WorkflowNormalizationError(f"action {index} type is required")
        action_type = raw_type.strip().casefold()
        if action_type not in ALLOWED_ACTION_TYPES:
            raise WorkflowNormalizationError(f"unknown action type: {raw_type}")
        action["type"] = action_type
        actions.append(action)
    return actions


def _normalize_step_id(raw_id: Any, *, index: int, seen_ids: set[str]) -> str:
    if raw_id is None or str(raw_id).strip() == "":
        candidate = f"step_{index}"
        suffix = 2
        while candidate in seen_ids:
            candidate = f"step_{index}_{suffix}"
            suffix += 1
        return candidate

    step_id = str(raw_id).strip()
    if not _STEP_ID_RE.fullmatch(step_id):
        raise WorkflowNormalizationError(f"invalid step id: {raw_id}")
    if step_id in seen_ids:
        raise WorkflowNormalizationError(f"duplicate step id: {step_id}")
    return step_id


def _normalize_step_type(raw_type: Any, index: int) -> str:
    if raw_type is None or str(raw_type).strip() == "":
        return "llm"
    step_type = str(raw_type).strip().casefold()
    if step_type not in ALLOWED_STEP_TYPES:
        raise WorkflowNormalizationError(f"unknown step type at step {index}: {raw_type}")
    return step_type


def _normalize_abort_policy(policy: dict[str, Any]) -> dict[str, Any]:
    out = {"type": "abort"}
    reason = _clean_optional_text(policy.get("reason"))
    if reason:
        out["reason"] = reason
    return out


def _normalize_ask_user_policy(policy: dict[str, Any]) -> dict[str, Any]:
    out = {
        "type": "ask_user",
        "resume": _clean_optional_text(policy.get("resume")) or "same_step",
    }
    prompt = _clean_optional_text(policy.get("prompt"))
    resume_step_id = _clean_optional_text(policy.get("resume_step_id"))
    if prompt:
        out["prompt"] = prompt
    if resume_step_id:
        out["resume_step_id"] = resume_step_id
    return out


def _normalize_manual_handoff_policy(policy: dict[str, Any]) -> dict[str, Any]:
    out = {
        "type": "manual_handoff",
        "resume": _clean_optional_text(policy.get("resume")) or "same_step",
    }
    prompt = _clean_optional_text(policy.get("prompt"))
    mode = _clean_optional_text(policy.get("mode"))
    resume_step_id = _clean_optional_text(policy.get("resume_step_id"))
    if prompt:
        out["prompt"] = prompt
    if mode:
        out["mode"] = mode
    if resume_step_id:
        out["resume_step_id"] = resume_step_id
    return out


def _normalize_retry_policy(policy: dict[str, Any]) -> dict[str, Any]:
    raw_attempts = policy.get("max_attempts", policy.get("max_retries", 1))
    try:
        max_attempts = int(raw_attempts)
    except (TypeError, ValueError) as exc:
        raise WorkflowNormalizationError("retry max_attempts must be an integer") from exc
    if max_attempts < 1:
        raise WorkflowNormalizationError("retry max_attempts must be at least 1")
    if max_attempts > 10:
        raise WorkflowNormalizationError("retry max_attempts must be 10 or less")

    then_policy = normalize_failure_policy(policy.get("then") or {"type": "abort"})
    return {"type": "retry", "max_attempts": max_attempts, "then": then_policy}


def _normalize_goto_policy(policy: dict[str, Any]) -> dict[str, Any]:
    target = _clean_optional_text(
        policy.get("target_step_id") or policy.get("step_id") or policy.get("target")
    )
    if not target:
        raise WorkflowNormalizationError("goto_step policy requires target_step_id")
    return {"type": "goto_step", "target_step_id": target}


def _validate_failure_targets(steps: list[dict[str, Any]]) -> None:
    step_ids = {step["id"] for step in steps}
    for step in steps:
        _validate_policy_targets(step["on_failure"], step_ids)
        _validate_policy_targets(step["on_success"], step_ids)


def normalize_success_policy(raw_policy: Any) -> dict[str, Any]:
    """What happens after a step succeeds.

    Defaults to ``continue`` — the behaviour every workflow had before this
    existed. ``end`` is what makes a diagnosis step reachable only through
    ``on_failure: goto_step``: without it a step placed after the work always
    runs, so an escalation that exists for failures also runs on every clean
    night and bills for it.
    """
    if raw_policy is None:
        return {"type": "continue"}

    if isinstance(raw_policy, str):
        value = raw_policy.strip()
        lowered = value.casefold()
        if lowered in {"continue", "next"}:
            return {"type": "continue"}
        if lowered in {"end", "stop", "done"}:
            return {"type": "end"}
        if lowered.startswith("goto_step:") or lowered.startswith("goto:"):
            target = value.split(":", 1)[1].strip()
            if not target:
                raise WorkflowNormalizationError("goto success policy requires a target")
            return {"type": "goto_step", "target_step_id": target}
        raise WorkflowNormalizationError(f"unknown success policy: {raw_policy}")

    if not isinstance(raw_policy, dict):
        raise WorkflowNormalizationError("success policy must be a string or object")

    raw_type = raw_policy.get("type") or raw_policy.get("action")
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise WorkflowNormalizationError("success policy type is required")
    policy_type = raw_type.strip().casefold()
    if policy_type == "goto":
        policy_type = "goto_step"
    if policy_type in {"continue", "next"}:
        return {"type": "continue"}
    if policy_type in {"end", "stop", "done"}:
        return {"type": "end"}
    if policy_type == "goto_step":
        target = (
            raw_policy.get("target_step_id")
            or raw_policy.get("step_id")
            or raw_policy.get("target")
        )
        if not isinstance(target, str) or not target.strip():
            raise WorkflowNormalizationError("goto success policy requires a target")
        return {"type": "goto_step", "target_step_id": target.strip()}
    raise WorkflowNormalizationError(f"unknown success policy: {raw_type}")


def _validate_policy_targets(policy: dict[str, Any], step_ids: set[str]) -> None:
    policy_type = policy.get("type")
    if policy_type == "goto_step":
        target = policy.get("target_step_id")
        if target not in step_ids:
            raise WorkflowNormalizationError(f"unknown goto_step target: {target}")
        return
    if policy_type == "retry":
        then_policy = policy.get("then")
        if isinstance(then_policy, dict):
            _validate_policy_targets(then_policy, step_ids)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_optional_text(value: Any) -> str | None:
    text = _clean_text(value)
    return text or None
