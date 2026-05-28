"""Pydantic models for prompt-to-local-app APIs."""

from pydantic import BaseModel


class LocalAppCreate(BaseModel):
    """Request body for creating a local app workspace from a prompt."""

    root_path: str
    app_name: str
    prompt: str
    template: str = "nextjs"
    provider_id: str | None = None
    model: str | None = None
