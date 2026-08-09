"""Pydantic models for the CLI-agent import APIs.

See :mod:`agent.cli_agent_sources` for what these describe: agent definition
files a user authored with Claude Code, discovered on this machine, and the
request/response shapes a client uses to list and import them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CliAgentImportRequest(BaseModel):
    """Request body for importing one discovered definition as a Code Bridge agent.

    ``source_path`` is the only input. It is never trusted on its own — the
    import route re-runs discovery and only proceeds if this path is one a
    fresh sweep actually found under the known roots (see
    ``agent.cli_agent_sources.discover_cli_agents``). A path that does not
    match a freshly discovered candidate is rejected before anything is read
    from it a second time, which is what stops this endpoint from becoming an
    arbitrary-file-read primitive for whatever filesystem path a client sends.
    """

    source_path: str = Field(min_length=1)


class CliAgentAutoImportUpdate(BaseModel):
    """Turn the periodic sweep's auto-import on or off.

    Off by default and deliberately so — see :mod:`agent.cli_agent_sweep`.
    Turning it on affects definitions discovered *from then on*; it does not
    reach back and import everything already on record, which is exactly the
    "25 agents appeared behind my back" outcome the default guards against.
    """

    enabled: bool


__all__ = ["CliAgentAutoImportUpdate", "CliAgentImportRequest"]
