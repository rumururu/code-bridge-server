"""Gemini native history adapter."""

from __future__ import annotations

from typing import Any

from .base import (
    HistoryListResult,
    HistoryMessagesResult,
    NativeHistoryAdapter,
    ResumeResult,
)


GEMINI_HISTORY_UNSUPPORTED = (
    "Gemini native CLI history/resume is not supported by this server yet."
)


class GeminiHistoryAdapter(NativeHistoryAdapter):
    """Typed unsupported adapter for Gemini native history."""

    provider_id = "google"
    supports_resume = False

    def list_sessions(self, project_path: str) -> HistoryListResult:
        return HistoryListResult(
            provider_id=self.provider_id,
            sessions=[],
            resumable=False,
            unsupported_reason=GEMINI_HISTORY_UNSUPPORTED,
        )

    def get_messages(self, project_path: str, session_id: str) -> HistoryMessagesResult:
        return HistoryMessagesResult(
            session_id=session_id,
            messages=[],
            provider_id=self.provider_id,
            resumable=False,
            unsupported_reason=GEMINI_HISTORY_UNSUPPORTED,
        )

    def require_resumable(self, project_path: str, session_id: str) -> None:
        return None

    async def resume_session(
        self,
        project_path: str,
        session_id: str,
        live_session: Any,
    ) -> ResumeResult:
        return ResumeResult(
            ok=False,
            session_id=session_id,
            provider_id=self.provider_id,
            resumable=False,
            unsupported_reason=GEMINI_HISTORY_UNSUPPORTED,
        )
