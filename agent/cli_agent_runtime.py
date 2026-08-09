"""Load an imported CLI agent's definition from disk, fresh, on every run.

An agent imported from an agent definition file does not carry a copy of that
file. It carries a pointer (the ``agent_cli_agent_imports`` row written at
import time), and this module is what follows it: read the file, parse it, and
return the definition the Claude session will run as. The point is that the
file is the program. Edit the source ``.md`` and the next scheduled run picks
the change up, with no re-import; the ``tools:`` the author declared reach the
session and are enforced there, so a definition that says ``tools: Read, Glob``
genuinely cannot write.

If this regresses, three things break for the user, all of them quietly:

- **Stale execution** (:func:`resolve_cli_agent_definition` returns something
  other than the current file): the user edits their agent, the nightly run
  keeps executing last month's prompt, and nothing anywhere says so. This is
  the exact failure the pointer exists to prevent.
- **Unenforced tools**: if the definition stops reaching
  ``ClaudeAgentOptions.agents``, the declared tool list becomes decoration. An
  agent the author wrote as read-only gets whatever the CLI allows, and it can
  write files at 3am on a schedule nobody is watching. This is also why a
  definition is only ever run on Claude — no other CLI here can be handed one,
  and running the prompt as plain text elsewhere is exactly this failure with
  extra steps.
- **Fallback to a copy** when the source is gone: a deleted, renamed, or
  newly-unparseable file must fail the run *naming the file*. Falling back to
  the stored stub — or to anything else — would run something the user never
  approved while reporting success. There is no fallback here on purpose.
"""

from __future__ import annotations

import logging
from pathlib import Path

from llm.llm_session import CliAgentDefinition

from .cli_agent_sources import (
    CliAgentCandidate,
    SkippedCliAgentFile,
    _parse_cli_agent_file,
    find_import_source_path_for_agent,
)

logger = logging.getLogger(__name__)


class CliAgentSourceUnavailableError(RuntimeError):
    """The file an imported agent points at cannot be used right now.

    Always names the path. Raised instead of returning a degraded definition,
    because the whole promise of a file-backed agent is that what runs is what
    the file says — and a run that quietly substitutes something else is worse
    than a run that fails.
    """


def find_cli_agent_source_path(agent_id: str) -> str | None:
    """The definition file this agent runs, or ``None`` if it is not one.

    Reads the mapping only — never the file — so callers that merely need to
    know "is this agent file-backed" (prompt composition, plan building)
    cannot fail because a file moved. Reading and validating happens in
    :func:`resolve_cli_agent_definition`, at the moment a run actually needs it.
    """
    if not agent_id:
        return None
    try:
        return find_import_source_path_for_agent(agent_id)
    except Exception:  # noqa: BLE001 - a mapping read must not break planning
        logger.debug(
            "cli agent runtime: could not read import mapping for %s",
            agent_id,
            exc_info=True,
        )
        return None


def resolve_cli_agent_definition(agent_id: str | None) -> CliAgentDefinition | None:
    """Read the current definition for a file-backed agent.

    Returns ``None`` for an ordinary Code Bridge agent — that is not an error,
    it is most agents. Raises :class:`CliAgentSourceUnavailableError` when the
    agent *is* file-backed but the file behind it is missing, unreadable, or
    no longer parses as an agent definition.

    The path is not re-validated against the discovery roots. It is not client
    input: it was written to this server's own database only after a sweep
    found and parsed the file (``cli_agent_sources.import_cli_agent``).
    Insisting it still be inside a root would fail runs for a reason that has
    nothing to do with the file — a project deregistered, a plugin dir renamed
    — and the file is the authority here.
    """
    if not agent_id:
        return None
    source_path = find_cli_agent_source_path(agent_id)
    if not source_path:
        return None

    parsed = _parse_cli_agent_file(Path(source_path), location="runtime")
    if isinstance(parsed, SkippedCliAgentFile):
        raise CliAgentSourceUnavailableError(
            f"agent '{agent_id}' runs the Claude Code agent defined at "
            f"'{source_path}', and that file cannot be used: "
            f"{parsed.reason} ({parsed.detail}). Nothing was run — the imported "
            "record is a pointer to this file, not a copy of it, so there is "
            "nothing to fall back to. Restore or re-import the file."
        )

    return _to_definition(parsed)


def _to_definition(candidate: CliAgentCandidate) -> CliAgentDefinition:
    """Everything the session needs, taken from the file as it is right now.

    ``name`` comes from the file too, not from the Code Bridge agent record: if
    the author renamed the agent, the name that selects it in the session has
    to be the one the file declares, or the CLI resolves nothing.
    """
    return CliAgentDefinition(
        name=candidate.name,
        description=candidate.description,
        prompt=candidate.body,
        source_path=candidate.source_path,
        tools=tuple(candidate.tools),
        model=candidate.model,
        effort=candidate.effort,
    )


__all__ = [
    "CliAgentSourceUnavailableError",
    "find_cli_agent_source_path",
    "resolve_cli_agent_definition",
]
