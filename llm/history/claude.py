"""Claude Code native history adapter."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .base import (
    HistoryListResult,
    HistoryMessagesResult,
    HistoryNotFoundError,
    HistorySessionSummary,
    NativeHistoryAdapter,
    ResumeResult,
)


def _slugify_path(project_path: str) -> str:
    """Turn a project path into Claude CLI's project-history directory name."""
    return re.sub(r"[^A-Za-z0-9.-]", "-", project_path)


def _sessions_dir(project_path: str) -> Path:
    home = Path(os.path.expanduser("~"))
    return home / ".claude" / "projects" / _slugify_path(project_path)


def _read_first_user_message(file: Path) -> str | None:
    """Scan a jsonl file for the first human-typed user message."""
    try:
        with file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "user":
                    continue
                message = event.get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip() or None
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text")
                            if isinstance(text, str) and text.strip():
                                return text.strip()
                        elif isinstance(block, str) and block.strip():
                            return block.strip()
    except OSError:
        return None
    return None


def _extract_text_content(content: Any) -> str:
    """Flatten Claude's message.content into plain display text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
            elif isinstance(block, str):
                pieces.append(block)
        return "".join(pieces)
    return ""


def _json_preview(value: Any, limit: int = 2000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}\n... [truncated]"


def _event_timestamp(event: dict[str, Any]) -> Any:
    return event.get("timestamp")


def _system_message(content: str, timestamp: Any = None) -> dict[str, Any] | None:
    text = content.strip()
    if not text:
        return None
    return {
        "type": "system",
        "content": text,
        "timestamp": timestamp,
    }


def _extract_claude_message_events(event: dict[str, Any]) -> list[dict[str, Any]]:
    event_type = event.get("type")
    if event_type not in ("user", "assistant"):
        return []

    timestamp = _event_timestamp(event)
    message = event.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        return [
            {
                "type": event_type,
                "content": text,
                "timestamp": timestamp,
            }
        ] if text else []

    if not isinstance(content, list):
        return []

    messages: list[dict[str, Any]] = []
    text_pieces: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_pieces.append(block)
            continue
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_pieces.append(text)
        elif block_type == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                item = _system_message(f"[thinking]\n{thinking}", timestamp)
                if item:
                    messages.append(item)
        elif block_type == "tool_use":
            name = block.get("name") or "tool"
            tool_id = block.get("id")
            details = _json_preview(block.get("input"))
            header = f"[tool call] {name}"
            if tool_id:
                header = f"{header} ({tool_id})"
            item = _system_message(f"{header}\n{details}", timestamp)
            if item:
                messages.append(item)
        elif block_type == "tool_result":
            tool_id = block.get("tool_use_id") or "unknown"
            result_text = _extract_text_content(block.get("content"))
            if not result_text.strip():
                result_text = _json_preview(block)
            item = _system_message(f"[tool result] {tool_id}\n{result_text}", timestamp)
            if item:
                messages.append(item)

    text = "".join(text_pieces).strip()
    if text:
        messages.insert(
            0,
            {
                "type": event_type,
                "content": text,
                "timestamp": timestamp,
            },
        )
    return messages


def _extract_claude_attachment_event(event: dict[str, Any]) -> dict[str, Any] | None:
    attachment = event.get("attachment")
    if not isinstance(attachment, dict):
        return None

    timestamp = _event_timestamp(event)
    attachment_type = attachment.get("type") or "attachment"
    if attachment_type == "file-history-snapshot":
        snapshot = attachment.get("snapshot") or {}
        if isinstance(snapshot, dict):
            tracked = snapshot.get("trackedFileBackups")
            count = len(tracked) if isinstance(tracked, dict) else 0
            return _system_message(
                f"[file history snapshot] tracked files: {count}",
                timestamp,
            )
    if attachment_type == "deferred_tools_delta":
        added = attachment.get("addedNames")
        removed = attachment.get("removedNames")
        parts = ["[tools updated]"]
        if isinstance(added, list) and added:
            parts.append(f"added: {', '.join(str(x) for x in added[:20])}")
            if len(added) > 20:
                parts.append(f"... +{len(added) - 20} more")
        if isinstance(removed, list) and removed:
            parts.append(f"removed: {', '.join(str(x) for x in removed[:20])}")
        return _system_message("\n".join(parts), timestamp)
    if attachment_type == "mcp_instructions_delta":
        added = attachment.get("addedNames")
        names = ", ".join(str(x) for x in added) if isinstance(added, list) else ""
        return _system_message(f"[MCP instructions updated] {names}".strip(), timestamp)
    if attachment_type == "skill_listing":
        count = attachment.get("skillCount")
        return _system_message(f"[skills listed] {count or 'unknown'} skills", timestamp)

    return _system_message(f"[{attachment_type}]\n{_json_preview(attachment)}", timestamp)


def _extract_claude_session_event(event: dict[str, Any]) -> dict[str, Any] | None:
    event_type = event.get("type")
    timestamp = _event_timestamp(event)
    if event_type == "permission-mode":
        mode = event.get("permissionMode") or event.get("permission_mode")
        return _system_message(f"[permission mode] {mode}", timestamp)
    if event_type == "queue-operation":
        operation = event.get("operation")
        return _system_message(f"[queue] {operation}", timestamp)
    return None


class ClaudeHistoryAdapter(NativeHistoryAdapter):
    """Read and resume Claude Code project-scoped sessions."""

    provider_id = "anthropic"
    supports_resume = True

    def list_sessions(self, project_path: str) -> HistoryListResult:
        directory = _sessions_dir(project_path)
        if not directory.is_dir():
            return HistoryListResult(provider_id=self.provider_id, sessions=[])

        entries: list[HistorySessionSummary] = []
        for file in directory.glob("*.jsonl"):
            try:
                stat = file.stat()
            except OSError:
                continue
            preview = _read_first_user_message(file)
            if not preview:
                continue
            entries.append(
                HistorySessionSummary(
                    session_id=file.stem,
                    preview=preview[:160],
                    updated_at=stat.st_mtime,
                    size_bytes=stat.st_size,
                    provider_id=self.provider_id,
                    scope="project",
                    cwd=project_path,
                    resumable=True,
                )
            )

        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return HistoryListResult(provider_id=self.provider_id, sessions=entries)

    def get_messages(self, project_path: str, session_id: str) -> HistoryMessagesResult:
        file = _sessions_dir(project_path) / f"{session_id}.jsonl"
        if not file.is_file():
            raise HistoryNotFoundError("Session not found")

        messages: list[dict[str, Any]] = []
        try:
            with file.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    messages.extend(_extract_claude_message_events(event))
                    attachment = _extract_claude_attachment_event(event)
                    if attachment:
                        messages.append(attachment)
                    session_event = _extract_claude_session_event(event)
                    if session_event:
                        messages.append(session_event)
        except OSError as exc:
            raise RuntimeError(f"Failed to read session: {exc}") from exc

        return HistoryMessagesResult(
            session_id=session_id,
            messages=messages,
            provider_id=self.provider_id,
            resumable=True,
        )

    async def resume_session(
        self,
        project_path: str,
        session_id: str,
        live_session: Any,
    ) -> ResumeResult:
        self.require_resumable(project_path, session_id)
        await live_session.resume_session(session_id)
        return ResumeResult(
            ok=True,
            session_id=session_id,
            provider_id=self.provider_id,
            resumable=True,
        )

    def require_resumable(self, project_path: str, session_id: str) -> None:
        file = _sessions_dir(project_path) / f"{session_id}.jsonl"
        if not file.is_file():
            raise HistoryNotFoundError("Session not found")
