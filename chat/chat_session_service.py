"""Session bootstrap helpers for websocket chat routes."""

from __future__ import annotations

from dataclasses import dataclass

from llm.claude_session import get_session_manager
from llm.llm_session import LlmSessionFactory, SubagentDefinition
from llm.llm_settings import get_llm_options_snapshot


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
    provider_name = {
        "anthropic": "Claude",
        "openai": "Codex",
        "google": "Gemini",
        "antigravity": "Antigravity",
    }.get(provider_id, provider_id.title())

    # Ask the factory rather than repeating the list here. A provider added to
    # one and not the other is offered in the picker and refused on use, which
    # is exactly how Antigravity first shipped: selectable, and then "not
    # supported yet" from a layer the user never sees.
    supported = LlmSessionFactory.get_supported_providers()
    if provider_id not in supported:
        raise ChatSessionInitError(
            f"Selected LLM provider '{provider_id}' is not supported yet. "
            f"Choose one of {', '.join(supported)} in Settings > LLM Configuration."
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
    subagent: SubagentDefinition | None = None,
):
    """Create or fetch the per-project chat session for selected provider.

    ``subagent`` is set when the agent being run is backed by a Claude Code
    subagent file: the session must run *as* that agent so the prompt and the
    tools the file declares are the ones in force. Only Claude can carry one —
    the factory raises ``SubagentUnsupportedError`` for any other provider
    rather than dropping it and running a plain prompt.
    """
    session_manager = get_session_manager()
    session = await session_manager.get_or_create_session(
        project_name,
        project_path,
        provider_id=selection.provider_id,
        model=selection.model,
        subagent=subagent,
    )
    return session
