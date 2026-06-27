"""Provider-native history adapter factory."""

from __future__ import annotations

from .base import (
    HistoryListResult,
    HistoryMessagesResult,
    HistoryNotFoundError,
    HistoryScopeError,
    HistorySessionSummary,
    NativeHistoryAdapter,
    ResumeResult,
)
from .claude import ClaudeHistoryAdapter
from .codex import CodexHistoryAdapter
from .gemini import GeminiHistoryAdapter


def get_history_adapter(provider_id: str) -> NativeHistoryAdapter:
    normalized = provider_id.strip().lower()
    if normalized == "anthropic":
        return ClaudeHistoryAdapter()
    if normalized == "openai":
        return CodexHistoryAdapter()
    if normalized == "google":
        return GeminiHistoryAdapter()
    return GeminiHistoryAdapter()


__all__ = [
    "ClaudeHistoryAdapter",
    "CodexHistoryAdapter",
    "GeminiHistoryAdapter",
    "HistoryListResult",
    "HistoryMessagesResult",
    "HistoryNotFoundError",
    "HistoryScopeError",
    "HistorySessionSummary",
    "NativeHistoryAdapter",
    "ResumeResult",
    "get_history_adapter",
]
