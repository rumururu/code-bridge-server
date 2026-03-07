"""Session bootstrap helpers for websocket chat routes."""

from __future__ import annotations

from dataclasses import dataclass

from claude_session import get_session_manager
from llm_settings import get_llm_options_snapshot


class ChatSessionInitError(Exception):
    """Raised when chat session cannot be initialized from current settings."""


@dataclass(frozen=True)
class ChatProviderSelection:
    """Resolved LLM provider/model used for chat session creation."""

    provider_id: str
    provider_name: str
    model: str | None


def get_chat_provider_selection() -> ChatProviderSelection:
    """Resolve current provider/model selection from settings."""
    llm_snapshot = get_llm_options_snapshot()
    selected = llm_snapshot.get("selected") if isinstance(llm_snapshot, dict) else {}
    selected_company = selected.get("company_id") if isinstance(selected, dict) else None
    selected_model = selected.get("model") if isinstance(selected, dict) else None

    provider_id = selected_company if selected_company else "anthropic"
    provider_name = "Claude" if provider_id == "anthropic" else provider_id.title()

    if provider_id not in ("anthropic", "openai"):
        raise ChatSessionInitError(
            f"Selected LLM provider '{provider_id}' is not supported yet. "
            "Select Anthropic or OpenAI in Settings > LLM Configuration."
        )

    resolved_model = selected_model if isinstance(selected_model, str) and selected_model.strip() else None
    return ChatProviderSelection(
        provider_id=provider_id,
        provider_name=provider_name,
        model=resolved_model,
    )


async def create_chat_session(
    project_name: str,
    project_path: str,
    selection: ChatProviderSelection,
):
    """Create or fetch the per-project chat session for selected provider."""
    session_manager = get_session_manager()
    session = await session_manager.get_or_create_session(
        project_name,
        project_path,
        provider_id=selection.provider_id,
        model=selection.model,
    )
    return session
