"""Pydantic models for the registered-script APIs."""

from typing import Any

from pydantic import BaseModel, Field


class ScriptParameter(BaseModel):
    """One argument a script needs before it can do its job.

    Three fields, and each is load-bearing. ``name`` is what a person is asked
    for. ``required`` is what the authoring gate reads: a step that names this
    script and supplies nothing is a step that will stop at 3am and ask. And
    ``description`` is the reason the question is answerable at all — "what
    should CHECK_DIR be?" is only a fair question next to "the directory whose
    free space is checked".
    """

    name: str = Field(min_length=1, max_length=80)
    # Absent means required: a parameter whose necessity nobody stated is one
    # worth asking about, not one to wave through.
    required: bool = True
    description: str = Field(default="", max_length=400)


class ScriptRegister(BaseModel):
    """Request body for registering a script a shell step may run.

    ``parameters`` omitted (or null) is **not** "this script takes no
    arguments" — it means the caller is not stating an interface, and the
    registry falls back to whatever the file itself declares in its ``@param``
    block, and to *unknown* when it declares nothing. Send ``[]`` to state
    outright that it needs none.
    """

    name: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1)
    description: str | None = None
    interpreter: str = "bash"
    default_args: list[str] = Field(default_factory=list)
    parameters: list[ScriptParameter] | None = None
    timeout_seconds: int | None = None
    created_by: str | None = None


class ScriptUpdate(BaseModel):
    """Partial update. Only the fields present are changed.

    ``parameters`` is the one field whose explicit ``null`` is meaningful:
    unset leaves the stored interface alone, ``null`` returns it to unknown,
    and a list replaces it.
    """

    name: str | None = Field(default=None, min_length=1, max_length=80)
    path: str | None = None
    description: str | None = None
    interpreter: str | None = None
    default_args: list[str] | None = None
    parameters: list[ScriptParameter] | None = None
    timeout_seconds: int | None = None


class ScriptDraftRequest(BaseModel):
    """Ask the LLM to write a script for a described job."""

    intent: str = Field(min_length=1)


class ScriptDraftSave(BaseModel):
    """Save a reviewed draft to disk and register it.

    The body is whatever is in the editor when the user presses save — the
    point of the review step is that they can change it first.
    """

    name: str = Field(min_length=1, max_length=80)
    body: str = Field(min_length=1)
    description: str | None = None
    interpreter: str = "bash"
    timeout_seconds: int | None = None
