"""Provider-native chat history abstractions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass
class HistorySessionSummary:
    """Normalized summary for one provider-native session."""

    session_id: str
    preview: str
    updated_at: float
    size_bytes: int
    provider_id: str
    scope: str = "unknown"
    cwd: str | None = None
    resumable: bool = False
    unsupported_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HistoryListResult:
    """List response with provider-level history capability metadata."""

    provider_id: str
    sessions: list[HistorySessionSummary]
    resumable: bool = True
    unsupported_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "sessions": [session.to_dict() for session in self.sessions],
            "resumable": self.resumable,
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass
class HistoryMessagesResult:
    """Normalized messages for one provider-native session."""

    session_id: str
    messages: list[dict[str, Any]]
    provider_id: str
    resumable: bool = True
    unsupported_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResumeResult:
    """Result of asking a provider adapter to pin a native session id."""

    ok: bool
    session_id: str
    provider_id: str
    resumable: bool
    unsupported_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HistoryNotFoundError(Exception):
    """Raised when a provider-native session cannot be found."""


class HistoryScopeError(Exception):
    """Raised when a session is known not to belong to the current project."""


class NativeHistoryAdapter(Protocol):
    """Provider-specific native history and resume adapter."""

    provider_id: str
    supports_resume: bool

    def list_sessions(self, project_path: str) -> HistoryListResult:
        """List sessions visible for the project."""
        ...

    def get_messages(self, project_path: str, session_id: str) -> HistoryMessagesResult:
        """Read normalized messages for a native session."""
        ...

    def require_resumable(self, project_path: str, session_id: str) -> None:
        """Raise if a native session cannot be safely resumed for the project."""
        ...

    async def resume_session(
        self,
        project_path: str,
        session_id: str,
        live_session: Any,
    ) -> ResumeResult:
        """Pin the live provider session to a native session id."""
        ...
