"""Load an imported subagent's definition from disk, fresh, on every run.

An agent imported from a Claude Code subagent file does not carry a copy of
that file. It carries a pointer (the ``agent_subagent_imports`` row written at
import time), and this module is what follows it: read the file, parse it, and
return the definition the Claude session will run as. The point is that the
file is the program. Edit the source ``.md`` and the next scheduled run picks
the change up, with no re-import; the ``tools:`` the author declared reach the
session and are enforced there, so a subagent that says ``tools: Read, Glob``
genuinely cannot write.

If this regresses, three things break for the user, all of them quietly:

- **Stale execution** (:func:`resolve_subagent_definition` returns something
  other than the current file): the user edits their subagent, the nightly run
  keeps executing last month's prompt, and nothing anywhere says so. This is
  the exact failure the pointer exists to prevent.
- **Unenforced tools**: if the definition stops reaching
  ``ClaudeAgentOptions.agents``, the declared tool list becomes decoration. An
  agent the author wrote as read-only gets whatever the CLI allows, and it can
  write files at 3am on a schedule nobody is watching.
- **Fallback to a copy** when the source is gone: a deleted, renamed, or
  newly-unparseable file must fail the run *naming the file*. Falling back to
  the stored stub — or to anything else — would run something the user never
  approved while reporting success. There is no fallback here on purpose.
"""

from __future__ import annotations

import logging
from pathlib import Path

from llm.llm_session import SubagentDefinition

from .subagent_sources import (
    SkippedSubagentFile,
    SubagentCandidate,
    _parse_subagent_file,
    find_import_source_path_for_agent,
)

logger = logging.getLogger(__name__)


class SubagentSourceUnavailableError(RuntimeError):
    """The file an imported agent points at cannot be used right now.

    Always names the path. Raised instead of returning a degraded definition,
    because the whole promise of a subagent-backed agent is that what runs is
    what the file says — and a run that quietly substitutes something else is
    worse than a run that fails.
    """


def find_subagent_source_path(agent_id: str) -> str | None:
    """The subagent file this agent runs, or ``None`` if it is not one.

    Reads the mapping only — never the file — so callers that merely need to
    know "is this agent subagent-backed" (prompt composition, plan building)
    cannot fail because a file moved. Reading and validating happens in
    :func:`resolve_subagent_definition`, at the moment a run actually needs it.
    """
    if not agent_id:
        return None
    try:
        return find_import_source_path_for_agent(agent_id)
    except Exception:  # noqa: BLE001 - a mapping read must not break planning
        logger.debug(
            "subagent runtime: could not read import mapping for %s",
            agent_id,
            exc_info=True,
        )
        return None


def resolve_subagent_definition(agent_id: str | None) -> SubagentDefinition | None:
    """Read the current definition for a subagent-backed agent.

    Returns ``None`` for an ordinary Code Bridge agent — that is not an error,
    it is most agents. Raises :class:`SubagentSourceUnavailableError` when the
    agent *is* subagent-backed but the file behind it is missing, unreadable,
    or no longer parses as a subagent.

    The path is not re-validated against the discovery roots. It is not client
    input: it was written to this server's own database only after a sweep
    found and parsed the file (``subagent_sources.import_subagent``). Insisting
    it still be inside a root would fail runs for a reason that has nothing to
    do with the file — a project deregistered, a plugin dir renamed — and the
    file is the authority here.
    """
    if not agent_id:
        return None
    source_path = find_subagent_source_path(agent_id)
    if not source_path:
        return None

    parsed = _parse_subagent_file(Path(source_path), location="runtime")
    if isinstance(parsed, SkippedSubagentFile):
        raise SubagentSourceUnavailableError(
            f"agent '{agent_id}' runs the Claude Code subagent defined at "
            f"'{source_path}', and that file cannot be used: "
            f"{parsed.reason} ({parsed.detail}). Nothing was run — the imported "
            "record is a pointer to this file, not a copy of it, so there is "
            "nothing to fall back to. Restore or re-import the file."
        )

    return _to_definition(parsed)


def _to_definition(candidate: SubagentCandidate) -> SubagentDefinition:
    """Everything the session needs, taken from the file as it is right now.

    ``name`` comes from the file too, not from the Code Bridge agent record: if
    the author renamed the subagent, the name that selects it in the session
    has to be the one the file declares, or the CLI resolves nothing.
    """
    return SubagentDefinition(
        name=candidate.name,
        description=candidate.description,
        prompt=candidate.body,
        source_path=candidate.source_path,
        tools=tuple(candidate.tools),
        model=candidate.model,
        effort=candidate.effort,
    )


__all__ = [
    "SubagentSourceUnavailableError",
    "find_subagent_source_path",
    "resolve_subagent_definition",
]
