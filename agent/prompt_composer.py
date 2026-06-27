"""Build the final system prompt for one agent run."""

from __future__ import annotations

import json
from typing import Any


def compose_system_prompt(
    agent: dict,
    task_goal: str | None = None,
    *,
    max_memory_items: int = 20,
    max_chars: int = 8000,
) -> str:
    """Compose an agent's base prompt, memories, workflow, and task goal."""
    system_prompt = str(agent.get("system_prompt") or "").strip()
    memories = _memory_items(agent, max_memory_items=max_memory_items)
    workflow_steps = _workflow_steps(agent)

    workflow_section = _workflow_section(workflow_steps)
    task_section = _task_section(task_goal)
    prompt_without_memory = _join_sections(
        section for section in (system_prompt, workflow_section, task_section) if section
    )
    if not memories:
        return prompt_without_memory

    memory_lines = [f"- {memory['content']}" for memory in memories]
    prompt = _join_sections(
        section
        for section in (
            system_prompt,
            _memory_section(memory_lines),
            workflow_section,
            task_section,
        )
        if section
    )
    if len(prompt) <= max_chars:
        return prompt
    return _compose_with_truncated_memory(
        system_prompt=system_prompt,
        memory_lines=memory_lines,
        workflow_section=workflow_section,
        task_section=task_section,
        max_chars=max_chars,
        fallback=prompt_without_memory,
    )


def _compose_with_truncated_memory(
    *,
    system_prompt: str,
    memory_lines: list[str],
    workflow_section: str | None,
    task_section: str | None,
    max_chars: int,
    fallback: str,
) -> str:
    candidate_lines = list(memory_lines)
    while len(candidate_lines) > 1:
        prompt = _join_sections(
            section
            for section in (
                system_prompt,
                _memory_section(candidate_lines),
                workflow_section,
                task_section,
            )
            if section
        )
        if len(prompt) <= max_chars:
            return prompt
        candidate_lines.pop()

    if not candidate_lines:
        return fallback

    prompt = _join_sections(
        section
        for section in (
            system_prompt,
            _memory_section(candidate_lines),
            workflow_section,
            task_section,
        )
        if section
    )
    if len(prompt) <= max_chars:
        return prompt

    overhead = len(prompt) - len(candidate_lines[0])
    available = max_chars - overhead
    if available <= 0:
        return fallback
    candidate_lines[0] = _truncate_line(candidate_lines[0], available)
    prompt = _join_sections(
        section
        for section in (
            system_prompt,
            _memory_section(candidate_lines),
            workflow_section,
            task_section,
        )
        if section
    )
    return prompt if len(prompt) <= max_chars else fallback


def _memory_items(agent: dict, *, max_memory_items: int) -> list[dict[str, str]]:
    raw_memories = (
        agent.get("memories")
        or agent.get("memory")
        or agent.get("agent_memories")
        or []
    )
    if not isinstance(raw_memories, list) or max_memory_items <= 0:
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_memories:
        if isinstance(item, str):
            content = _clean_inline(item)
            pinned = False
            created_at = ""
        elif isinstance(item, dict):
            content = _clean_inline(item.get("content"))
            pinned = bool(item.get("pinned"))
            created_at = str(item.get("created_at") or "")
        else:
            continue
        if content:
            normalized.append(
                {
                    "content": content,
                    "pinned": pinned,
                    "created_at": created_at,
                }
            )

    normalized.sort(key=lambda item: item["created_at"], reverse=True)
    normalized.sort(key=lambda item: not item["pinned"])
    return [{"content": str(item["content"])} for item in normalized[:max_memory_items]]


def _workflow_steps(agent: dict) -> list[dict[str, Any]]:
    raw_flow = agent.get("flow_json") or agent.get("workflow") or agent.get("flow") or []
    if isinstance(raw_flow, str):
        try:
            raw_flow = json.loads(raw_flow)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_flow = []
    if not isinstance(raw_flow, list):
        return []
    return [step for step in raw_flow if isinstance(step, dict)]


def _workflow_section(steps: list[dict[str, Any]]) -> str | None:
    if not steps:
        return None
    lines = ["Your workflow (run this in order):"]
    for index, step in enumerate(steps, start=1):
        name = _clean_inline(step.get("name") or step.get("id") or f"Step {index}")
        step_type = _clean_inline(step.get("type")) or "llm"
        instruction = _clean_inline(step.get("instruction")) or _clean_inline(
            step.get("description")
        )
        observation = _clean_inline(step.get("observation"))
        memory_read = _clean_inline(_jsonish(step.get("memory_read") or step.get("memoryRead")))
        memory_write = _clean_inline(_jsonish(step.get("memory_write") or step.get("memoryWrite")))
        tool_hint = _clean_inline(step.get("tool_hint"))
        success = _clean_inline(step.get("success_criteria")) or "not specified"
        on_failure = _clean_inline(_jsonish(step.get("on_failure"))) or "not specified"
        actions = _clean_inline(_jsonish(step.get("actions"))) if step.get("actions") else ""
        details = [f"type: {step_type}"]
        if instruction:
            details.append(f"instruction: {instruction}")
        if observation:
            details.append(f"observation: {observation}")
        if memory_read:
            details.append(f"memory read: {memory_read}")
        if memory_write:
            details.append(f"memory write: {memory_write}")
        if tool_hint:
            details.append(f"tool_hint: {tool_hint}")
        if actions:
            details.append(f"actions: {actions}")
        lines.append(
            f"{index}. {name} "
            f"({'; '.join(details)}. success_criteria: {success}. on_failure: {on_failure}.)"
        )
    return "\n".join(lines)


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _memory_section(memory_lines: list[str]) -> str | None:
    if not memory_lines:
        return None
    return "\n".join(
        [
            "Your accumulated learnings (most recent + pinned first):",
            *memory_lines,
        ]
    )


def _task_section(task_goal: str | None) -> str | None:
    if task_goal is None:
        return None
    return f"Current task: {str(task_goal).strip()}"


def _join_sections(sections) -> str:
    return "\n\n---\n".join(str(section) for section in sections)


def _clean_inline(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _truncate_line(line: str, max_len: int) -> str:
    if len(line) <= max_len:
        return line
    if max_len <= 3:
        return line[:max_len]
    return f"{line[: max_len - 3]}..."
