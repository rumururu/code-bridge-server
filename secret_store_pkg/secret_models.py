"""Pydantic models for the secrets REST surface.

Kept deliberately small. The store layer (``secret_store.upsert``) is
the source of truth for validation; these models exist so FastAPI can
reject obvious bad input at the boundary with a 422 instead of letting
it reach the store layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SecretEntry(BaseModel):
    """One row in ``GET /api/secrets``.

    Note: ``value`` is **not** a field here. The API contract is
    "names + presence flag, never the value", and the model intentionally
    has no field that could leak it.
    """

    name: str = Field(min_length=1, max_length=64)
    has_value: bool


class SecretUpsert(BaseModel):
    """Body of ``POST /api/secrets``.

    Pattern matches the store's ``_KEY_RE`` so the 422 message is
    actionable even before the store-layer error path runs.
    """

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Z][A-Z0-9_]*$",
    )
    value: str = Field(min_length=1, max_length=4096)
