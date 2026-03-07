"""Abstract LLM session management interface.

Provides a unified interface for different LLM providers (Claude, Codex, etc.)
so the server can work with any supported CLI tool.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator


class LlmSession(ABC):
    """Abstract base class for LLM provider sessions.

    Each provider (Claude, Codex, etc.) implements this interface to provide
    consistent behavior for the chat WebSocket endpoint.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Return the provider identifier (e.g., 'anthropic', 'openai')."""
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether the underlying process is alive."""
        ...

    @property
    @abstractmethod
    def session_id(self) -> str | None:
        """Provider-specific session/conversation ID."""
        ...

    @property
    @abstractmethod
    def has_pending_permission_denials(self) -> bool:
        """Whether there is a pending permission request."""
        ...

    @abstractmethod
    async def send_message(
        self,
        message: str,
        permission_mode: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Send a message and stream response events.

        Yields events in a normalized format:
        - {"type": "assistant", "message": {...}} for assistant messages
        - {"type": "result", ...} when turn completes
        - {"type": "control_request", "request": {...}} for permission prompts
        - {"type": "error", "error": {"message": str}} on errors
        """
        ...

    @abstractmethod
    async def approve_pending_permissions_and_retry(
        self,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Approve a pending permission request and continue the turn."""
        ...

    @abstractmethod
    async def deny_pending_permissions(
        self,
        message: str = "Permission denied by user.",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Deny a pending permission request and continue the turn."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the session and clean up resources."""
        ...

    @abstractmethod
    async def set_model(self, model: str | None) -> None:
        """Set the model for subsequent turns (may restart the session)."""
        ...

    @abstractmethod
    async def abort_current_turn(self) -> bool:
        """Abort the current turn if one is in progress.

        Returns True if abort was successful, False if no turn in progress.
        """
        ...


@dataclass
class DetectedProvider:
    """Information about a detected LLM provider."""

    id: str
    name: str
    command: str
    installed: bool
    chat_supported: bool
    models: list[str]
    error_message: str | None = None


class LlmSessionFactory:
    """Factory for creating LLM sessions based on provider ID."""

    @staticmethod
    def create_session(
        provider_id: str,
        project_path: str,
        model: str | None = None,
    ) -> LlmSession:
        """Create an LLM session for the specified provider.

        Args:
            provider_id: Provider identifier (e.g., 'anthropic', 'openai')
            project_path: Path to the project directory
            model: Optional model name/alias to use

        Returns:
            An LlmSession instance for the provider

        Raises:
            ValueError: If the provider is not supported
        """
        normalized_id = provider_id.strip().lower()

        if normalized_id == "anthropic":
            from claude_session import ClaudeSession
            return ClaudeSession(project_path=project_path, model=model)

        if normalized_id == "openai":
            from codex_session import CodexSession
            return CodexSession(project_path=project_path, model=model)

        raise ValueError(f"Unknown LLM provider: {provider_id}")

    @staticmethod
    def get_supported_providers() -> list[str]:
        """Return list of supported provider IDs."""
        return ["anthropic", "openai"]
