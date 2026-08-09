"""Pydantic models for the subagent-import APIs.

See :mod:`agent.subagent_sources` for what these describe: Claude Code
subagent markdown files discovered on this machine, and the request/response
shapes a client uses to list and import them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubagentImportRequest(BaseModel):
    """Request body for importing one discovered subagent as a Code Bridge agent.

    ``source_path`` is the only input. It is never trusted on its own — the
    import route re-runs discovery and only proceeds if this path is one a
    fresh sweep actually found under the three known roots (see
    ``agent.subagent_sources.discover_subagent_candidates``). A path that
    does not match a freshly discovered candidate is rejected before
    anything is read from it a second time, which is what stops this
    endpoint from becoming an arbitrary-file-read primitive for whatever
    filesystem path a client sends.
    """

    source_path: str = Field(min_length=1)


class SubagentAutoImportUpdate(BaseModel):
    """Turn the periodic sweep's auto-import on or off.

    Off by default and deliberately so — see
    :mod:`agent.subagent_sweep`. Turning it on affects subagents discovered
    *from then on*; it does not reach back and import everything already on
    record, which is exactly the "25 agents appeared behind my back" outcome
    the default guards against.
    """

    enabled: bool


__all__ = ["SubagentAutoImportUpdate", "SubagentImportRequest"]
