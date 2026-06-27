"""Android app action adapter boundary for workflow runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AppActionAdapterResult:
    """Normalized result returned by an app action adapter."""

    status: str
    message: str = ""
    observations: list[dict[str, Any]] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    wait_reason: str | None = None
    prompt: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def waiting_for_user(self) -> bool:
        return self.status == "waiting_for_user"

    def to_output(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "status": self.status,
            "message": self.message,
            "observations": self.observations,
            "screenshots": self.screenshots,
        }
        if self.error is not None:
            output["error"] = self.error
        if self.wait_reason:
            output["wait_reason"] = self.wait_reason
        if self.prompt:
            output["prompt"] = self.prompt
        return output


class AppActionAdapter(Protocol):
    async def run_actions(
        self,
        actions: list[dict[str, Any]],
        *,
        context: dict[str, Any],
    ) -> AppActionAdapterResult:
        """Run Android app actions and return a normalized result."""
