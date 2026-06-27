"""OpenAI Codex native history adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from .base import (
    HistoryListResult,
    HistoryMessagesResult,
    HistoryNotFoundError,
    HistoryScopeError,
    HistorySessionSummary,
    NativeHistoryAdapter,
    ResumeResult,
)


PROJECT_SCOPE_UNKNOWN = (
    "Session cwd is unavailable, so it cannot be safely matched to this project."
)


@dataclass
class _CodexFileInfo:
    cwd: str | None = None
    preview: str | None = None
    updated_at: float = 0.0
    messages: list[dict[str, Any]] | None = None


def _codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path(os.path.expanduser("~")) / ".codex"


def _codex_index_file() -> Path:
    return _codex_home() / "session_index.jsonl"


def _codex_sessions_dir() -> Path:
    return _codex_home() / "sessions"


def _parse_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            return None
    return None


def _read_codex_index() -> list[dict[str, Any]]:
    index = _codex_index_file()
    if not index.is_file():
        return []

    entries: list[dict[str, Any]] = []
    try:
        with index.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                session_id = item.get("id")
                if not isinstance(session_id, str) or not session_id.strip():
                    continue
                entries.append(item)
    except OSError:
        return []
    return entries


def _find_codex_session_file(session_id: str) -> Path | None:
    sessions_dir = _codex_sessions_dir()
    if not sessions_dir.is_dir():
        return None
    matches = sorted(sessions_dir.rglob(f"*{session_id}.jsonl"))
    for match in matches:
        if match.is_file():
            return match
    return None


def _extract_codex_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
                continue
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                pieces.append(text)
        return "".join(pieces)
    return ""


def _json_preview(value: Any, limit: int = 2000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}\n... [truncated]"


def _system_message(content: str, timestamp: Any = None) -> dict[str, Any] | None:
    text = content.strip()
    if not text:
        return None
    return {
        "type": "system",
        "content": text,
        "timestamp": timestamp,
    }


def _extract_cwd_from_mapping(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None

    for key in ("cwd", "working_directory", "working_dir", "project_path"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    for key in ("payload", "metadata", "session_meta", "session", "item"):
        candidate = _extract_cwd_from_mapping(value.get(key))
        if candidate:
            return candidate
    return None


def _normalize_path(value: str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _is_project_cwd(project_path: str, cwd: str | None) -> bool:
    if not cwd:
        return False
    try:
        project = _normalize_path(project_path)
        candidate = _normalize_path(cwd)
    except (OSError, RuntimeError, ValueError):
        return False
    return candidate == project or project in candidate.parents


def _summary_scope(project_path: str, cwd: str | None) -> str:
    if not cwd:
        return "unknown"
    return "project" if _is_project_cwd(project_path, cwd) else "global"


def _read_codex_file(file: Path, include_messages: bool = False) -> _CodexFileInfo:
    info = _CodexFileInfo(messages=[] if include_messages else None)
    primary_messages: list[dict[str, Any]] = []
    fallback_messages: list[dict[str, Any]] = []
    system_messages: list[dict[str, Any]] = []

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
                if not isinstance(event, dict):
                    continue

                if not info.cwd:
                    info.cwd = _extract_cwd_from_mapping(event)

                timestamp = event.get("timestamp")
                parsed_timestamp = _parse_timestamp(timestamp)
                if parsed_timestamp:
                    info.updated_at = max(info.updated_at, parsed_timestamp)

                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue

                if event.get("type") == "event_msg":
                    payload_type = payload.get("type")
                    role = None
                    if payload_type == "user_message":
                        role = "user"
                    elif payload_type == "agent_message":
                        role = "assistant"
                    message = payload.get("message")
                    if role and isinstance(message, str) and message.strip():
                        if not info.preview and role == "user":
                            info.preview = message.strip()[:160]
                        if include_messages:
                            primary_messages.append(
                                {
                                    "type": role,
                                    "content": message,
                                    "timestamp": timestamp,
                                }
                            )
                    elif include_messages and payload_type:
                        item = _system_message(
                            f"[codex event] {payload_type}\n{_json_preview(payload)}",
                            timestamp,
                        )
                        if item:
                            system_messages.append(item)
                    continue

                if event.get("type") == "response_item":
                    payload_type = payload.get("type")
                    if include_messages and payload_type == "function_call":
                        name = payload.get("name") or "tool"
                        call_id = payload.get("call_id")
                        arguments = payload.get("arguments")
                        header = f"[tool call] {name}"
                        if call_id:
                            header = f"{header} ({call_id})"
                        item = _system_message(
                            f"{header}\n{arguments or _json_preview(payload)}",
                            timestamp,
                        )
                        if item:
                            system_messages.append(item)
                        continue
                    if include_messages and payload_type == "function_call_output":
                        call_id = payload.get("call_id") or "unknown"
                        output = payload.get("output")
                        item = _system_message(
                            f"[tool result] {call_id}\n{output or _json_preview(payload)}",
                            timestamp,
                        )
                        if item:
                            system_messages.append(item)
                        continue
                    if include_messages and payload_type == "reasoning":
                        summary = payload.get("summary")
                        if isinstance(summary, list) and summary:
                            detail = _json_preview(summary)
                        else:
                            detail = "Reasoning was recorded by Codex."
                        item = _system_message(f"[thinking]\n{detail}", timestamp)
                        if item:
                            system_messages.append(item)
                        continue

                    role = payload.get("role")
                    if role not in ("user", "assistant"):
                        continue
                    text = _extract_codex_text_content(payload.get("content"))
                    if text.strip() and not text.lstrip().startswith("<environment_context>"):
                        if not info.preview and role == "user":
                            info.preview = text.strip()[:160]
                        if include_messages:
                            fallback_messages.append(
                                {
                                    "type": role,
                                    "content": text,
                                    "timestamp": timestamp,
                                }
                            )
    except OSError as exc:
        raise RuntimeError(f"Failed to read session: {exc}") from exc

    if include_messages:
        base_messages = primary_messages if primary_messages else fallback_messages
        info.messages = sorted(
            [*base_messages, *system_messages],
            key=lambda item: _parse_timestamp(item.get("timestamp")) or 0.0,
        )
    return info


class CodexHistoryAdapter(NativeHistoryAdapter):
    """Read and resume Codex sessions scoped by rollout cwd metadata."""

    provider_id = "openai"
    supports_resume = True

    def list_sessions(self, project_path: str) -> HistoryListResult:
        entries: list[HistorySessionSummary] = []
        for item in _read_codex_index():
            summary = self._summary_from_index_item(project_path, item)
            if summary is None:
                continue
            if summary.scope != "project":
                continue
            entries.append(summary)

        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return HistoryListResult(provider_id=self.provider_id, sessions=entries)

    def get_messages(self, project_path: str, session_id: str) -> HistoryMessagesResult:
        summary, file = self._require_project_session(project_path, session_id)
        if file is None:
            raise HistoryNotFoundError("Session file not found")
        info = _read_codex_file(file, include_messages=True)
        return HistoryMessagesResult(
            session_id=session_id,
            messages=info.messages or [],
            provider_id=self.provider_id,
            resumable=summary.resumable,
            unsupported_reason=summary.unsupported_reason,
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
        self._require_project_session(project_path, session_id)

    def _summary_from_index_item(
        self,
        project_path: str,
        item: dict[str, Any],
    ) -> HistorySessionSummary | None:
        session_id = item.get("id")
        if not isinstance(session_id, str) or not session_id.strip():
            return None

        preview = str(item.get("thread_name") or session_id)[:160]
        updated_at = _parse_timestamp(item.get("updated_at")) or 0.0
        cwd = _extract_cwd_from_mapping(item)
        size_bytes = 0

        file = _find_codex_session_file(session_id)
        if file is not None:
            try:
                stat = file.stat()
                size_bytes = stat.st_size
                if not updated_at:
                    updated_at = stat.st_mtime
            except OSError:
                pass
            file_info = _read_codex_file(file, include_messages=False)
            cwd = file_info.cwd or cwd
            if not preview or preview == session_id:
                preview = file_info.preview or preview
            if file_info.updated_at and not updated_at:
                updated_at = file_info.updated_at

        scope = _summary_scope(project_path, cwd)
        unsupported_reason = None
        if scope == "unknown":
            unsupported_reason = PROJECT_SCOPE_UNKNOWN
        elif scope != "project":
            unsupported_reason = "Session belongs to a different working directory."

        return HistorySessionSummary(
            session_id=session_id,
            preview=preview,
            updated_at=updated_at,
            size_bytes=size_bytes,
            provider_id=self.provider_id,
            scope=scope,
            cwd=cwd,
            resumable=scope == "project",
            unsupported_reason=unsupported_reason,
        )

    def _load_summary(
        self,
        project_path: str,
        session_id: str,
    ) -> tuple[HistorySessionSummary, Path | None]:
        for item in _read_codex_index():
            if item.get("id") == session_id:
                summary = self._summary_from_index_item(project_path, item)
                if summary is None:
                    break
                return summary, _find_codex_session_file(session_id)

        file = _find_codex_session_file(session_id)
        if file is None:
            raise HistoryNotFoundError("Session not found")

        try:
            stat = file.stat()
            size_bytes = stat.st_size
            updated_at = stat.st_mtime
        except OSError:
            size_bytes = 0
            updated_at = 0.0
        info = _read_codex_file(file, include_messages=False)
        scope = _summary_scope(project_path, info.cwd)
        unsupported_reason = None
        if scope == "unknown":
            unsupported_reason = PROJECT_SCOPE_UNKNOWN
        elif scope != "project":
            unsupported_reason = "Session belongs to a different working directory."
        return (
            HistorySessionSummary(
                session_id=session_id,
                preview=info.preview or session_id,
                updated_at=info.updated_at or updated_at,
                size_bytes=size_bytes,
                provider_id=self.provider_id,
                scope=scope,
                cwd=info.cwd,
                resumable=scope == "project",
                unsupported_reason=unsupported_reason,
            ),
            file,
        )

    def _require_project_session(
        self,
        project_path: str,
        session_id: str,
    ) -> tuple[HistorySessionSummary, Path | None]:
        summary, file = self._load_summary(project_path, session_id)
        if summary.scope != "project":
            raise HistoryScopeError(summary.unsupported_reason or PROJECT_SCOPE_UNKNOWN)
        return summary, file
