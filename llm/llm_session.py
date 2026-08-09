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
        - normalized non-Claude events should include raw_event/provider_id/session_id
          metadata when the provider CLI emits a raw JSON event.
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


@dataclass(frozen=True)
class SubagentDefinition:
    """One Claude Code subagent, read from its file, ready to hand to a session.

    Frozen and hashable on purpose: :class:`~llm.claude_session.SessionManager`
    caches sessions per scope and compares this value to decide whether a cached
    session still matches. Two workflow steps of the same task — one backed by a
    subagent, one not — must not share a session, or the second silently inherits
    the first one's tool restrictions or escapes them.

    ``tools`` is what the file declared, verbatim, including forms like
    ``Agent(plugin:child)``. It is passed through rather than translated: it is
    the Claude session that enforces it, and rewriting a tool name here would
    change what a read-only agent is allowed to do.
    """

    name: str
    description: str
    prompt: str
    source_path: str
    tools: tuple[str, ...] = ()
    model: str | None = None
    effort: str | None = None


class SubagentUnsupportedError(RuntimeError):
    """A subagent-backed agent was pointed at a provider that cannot run it.

    Only Claude accepts an agent definition (``ClaudeAgentOptions.agents``).
    Codex has no ``--agent`` and no subagent concept; Antigravity lists no
    agents at all today. Running such an agent on one of them would mean
    quietly dropping the definition and executing a plain prompt with none of
    the declared tool restrictions — a read-only agent turned loose. So it is
    refused loudly instead, and the run fails with this message rather than
    succeeding as something else. If a provider grows the capability, teach the
    factory about it here; until then this is the honest answer.
    """


class LlmSessionFactory:
    """Factory for creating LLM sessions based on provider ID."""

    @staticmethod
    def create_session(
        provider_id: str,
        project_path: str,
        model: str | None = None,
        subagent: SubagentDefinition | None = None,
    ) -> LlmSession:
        """Create an LLM session for the specified provider.

        Args:
            provider_id: Provider identifier (e.g., 'anthropic', 'openai')
            project_path: Path to the project directory
            model: Optional model name/alias to use
            subagent: A Claude Code subagent definition the session must run as

        Returns:
            An LlmSession instance for the provider

        Raises:
            ValueError: If the provider is not supported
            SubagentUnsupportedError: If ``subagent`` is set and the provider
                cannot carry an agent definition
        """
        normalized_id = provider_id.strip().lower()

        if normalized_id == "anthropic":
            from llm.claude_session import ClaudeSession
            return ClaudeSession(
                project_path=project_path, model=model, subagent=subagent
            )

        if subagent is not None:
            raise SubagentUnsupportedError(
                f"agent '{subagent.name}' runs the Claude Code subagent defined at "
                f"{subagent.source_path}, and the selected provider "
                f"'{normalized_id}' cannot carry a subagent definition — only "
                "Claude sessions can. Nothing was run: without the definition the "
                "turn would be a plain prompt with none of the declared tool "
                "restrictions. Select Claude in Settings > LLM Configuration, or "
                "run a different agent."
            )

        if normalized_id == "openai":
            from llm.codex_session import CodexSession
            return CodexSession(project_path=project_path, model=model)

        if normalized_id == "google":
            from llm.gemini_session import GeminiSession
            return GeminiSession(project_path=project_path, model=model)

        if normalized_id == "antigravity":
            from llm.antigravity_session import AntigravitySession
            return AntigravitySession(project_path=project_path, model=model)

        raise ValueError(f"Unknown LLM provider: {provider_id}")

    @staticmethod
    def get_supported_providers() -> list[str]:
        """Return list of supported provider IDs."""
        return ["anthropic", "openai", "google", "antigravity"]
