"""Decide which discovered agent definitions can run on their own, and say why not.

Code Bridge runs an imported agent unattended — on a schedule, at 3am, with
nobody watching. Most Claude Code plugin agents installed on a machine are not
written for that. They are *workers*: a parent agent or a plugin workflow
dispatches them with a structured payload (a finding, a workspace path, a
component list) and they are built to refuse anything else. Scheduling one
produces a refusal every night, forever, and the user's only signal is a
failed run whose message is the agent politely declining.

If this regresses, two different things break for the user:

- **Too permissive** (a dispatch-dependent definition is offered as
  importable): the user schedules it, and every run refuses. The agent looks
  broken and the reason — that this file was never meant to be invoked
  directly — is buried in the run transcript.
- **Too strict, or silent** (:func:`classify_candidates` excludes something and
  nothing reports it): an agent the user knows they installed is simply
  absent from the import list, with no way to find out why. That is why every
  exclusion carries a machine-readable ``reason`` and a human-readable
  ``detail`` naming the evidence, and why the discovery API returns the
  excluded set alongside the importable one rather than dropping it.

How "standalone-runnable" is decided
------------------------------------
Two signals, both *declarations by the file's own author* in fields whose
documented purpose is exactly this. Neither is a keyword sweep of the body —
the body is the system prompt, prose written for a model, and matching
phrases in it ("WORKSPACE", "refusal", "must carry") mislabels agents wholesale:
on the machine this was built against, that approach flagged the
``claude-security`` orchestrator, whose own description says it is "best as
the main agent of a session", and six other plainly user-invoked reviewers.

1. **Declared dispatch target** (:data:`EXCLUDED_DISPATCH_TARGET`). Claude Code
   frontmatter declares which *other* agents a file may dispatch, as
   ``tools: …, Agent(plugin:child-a, plugin:child-b)``. That is a
   machine-readable edge in a dispatch graph, not prose: if some other
   installed definition names this one as something it dispatches, this one
   has a parent, and the parent is what supplies the payload it expects. The
   detail names the parent file so the user can see the claim.

2. **Direct invocation disclaimed by the author**
   (:data:`EXCLUDED_DIRECT_INVOCATION_DISCLAIMED`). The ``description`` field
   is the one field Claude Code itself reads to decide whether to invoke an
   agent — it is the invocation contract, and several shipped agents write the
   answer into it in as many words ("dispatched by the fix job, not for direct
   invocation"). Reading a negation of direct invocation out of *that field*
   is reading the contract, not guessing from prose.

Either signal is enough to exclude, because the cost is asymmetric: an
excluded agent is still listed, with the reason, and the user can look at it;
an agent wrongly offered becomes a schedule that fails silently every night.

Deliberately *not* signals: a plugin workflow script naming an agent (a
workflow is itself something a user starts, so being reachable from one says
nothing about this agent alone), and the presence of required-input language
in the body (see above).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .cli_agent_sources import CliAgentCandidate

#: This definition is named in another installed definition's ``Agent(...)``
#: tool declaration, so it has a parent that dispatches it.
EXCLUDED_DISPATCH_TARGET = "dispatch_target"

#: This definition's own ``description`` says it is not for direct invocation.
EXCLUDED_DIRECT_INVOCATION_DISCLAIMED = "direct_invocation_disclaimed"


#: Negations of *direct invocation*, matched against the ``description`` field
#: only. Every alternative is anchored on the agent declining to be invoked
#: itself — never on it merely mentioning a dispatcher, because plenty of
#: standalone agents describe who typically calls them.
_DIRECT_INVOCATION_DISCLAIMER = re.compile(
    r"not\s+(?:intended\s+|meant\s+|designed\s+)?for\s+direct\s+(?:invocation|use|call)"
    r"|not\s+(?:to\s+be\s+|meant\s+to\s+be\s+|intended\s+to\s+be\s+)?"
    r"(?:invoked|called|run|used)\s+directly"
    r"|do\s+not\s+(?:invoke|call|run|use)\s+(?:this\s+\S+\s+)?directly"
    r"|internal\s+use\s+only",
    re.IGNORECASE,
)

#: ``Agent(a, b)`` inside a frontmatter ``tools:`` entry. The names inside are
#: either bare (``explore``) or plugin-qualified (``claude-security:explore``).
_AGENT_TOOL = re.compile(r"^Agent\s*\((?P<inner>.*)\)$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class CliAgentExclusion:
    """Why one discovered agent definition is not offered for import."""

    reason: str
    detail: str


def _plugin_of(location: str) -> str | None:
    """``plugin:<marketplace>/<plugin>`` -> ``<plugin>``; anything else -> None."""
    if not location.startswith("plugin:"):
        return None
    rest = location[len("plugin:") :]
    _, separator, plugin = rest.partition("/")
    return plugin if separator and plugin else None


def dispatch_references(tools: Iterable[str]) -> list[str]:
    """Every agent name an ``Agent(...)`` tool declaration dispatches.

    Names are returned exactly as written — bare or ``plugin:name`` — because
    the qualifier is what makes a reference resolvable across plugins.
    """
    referenced: list[str] = []
    for tool in tools:
        match = _AGENT_TOOL.match(str(tool).strip())
        if match is None:
            continue
        for part in match.group("inner").split(","):
            name = part.strip()
            if name:
                referenced.append(name)
    return referenced


def _index(candidates: Iterable["CliAgentCandidate"]) -> tuple[dict, dict]:
    """``(plugin, name) -> candidate`` and ``(location, name) -> candidate``."""
    by_plugin: dict[tuple[str, str], "CliAgentCandidate"] = {}
    by_location: dict[tuple[str, str], "CliAgentCandidate"] = {}
    for candidate in candidates:
        by_location.setdefault((candidate.location, candidate.name), candidate)
        plugin = _plugin_of(candidate.location)
        if plugin:
            by_plugin.setdefault((plugin, candidate.name), candidate)
    return by_plugin, by_location


def classify_candidates(
    candidates: list["CliAgentCandidate"],
) -> dict[str, CliAgentExclusion]:
    """Map ``candidate_id`` -> exclusion for every candidate that is not standalone.

    A candidate absent from the returned mapping is standalone-runnable. The
    whole sweep is classified at once because signal 1 is a relation *between*
    files: whether anything else installed on this machine declares it as a
    dispatch target.
    """
    exclusions: dict[str, CliAgentExclusion] = {}

    # Signal 2 first: the author's own words about their own file are the
    # clearest answer to "why isn't this on the list", so they win the report
    # when both signals fire.
    for candidate in candidates:
        if _DIRECT_INVOCATION_DISCLAIMER.search(candidate.description or ""):
            exclusions[candidate.candidate_id] = CliAgentExclusion(
                reason=EXCLUDED_DIRECT_INVOCATION_DISCLAIMED,
                detail=(
                    f"the description in {candidate.source_path} states this "
                    "agent is not for direct invocation"
                ),
            )

    by_plugin, by_location = _index(candidates)
    for referrer in candidates:
        referrer_plugin = _plugin_of(referrer.location)
        for reference in dispatch_references(referrer.tools):
            qualifier, separator, bare = reference.partition(":")
            if separator and bare:
                target = by_plugin.get((qualifier.strip(), bare.strip()))
            else:
                target = by_location.get((referrer.location, reference))
            if target is None or target.candidate_id == referrer.candidate_id:
                continue
            if target.candidate_id in exclusions:
                continue
            scope = f" in {referrer_plugin}" if referrer_plugin else ""
            exclusions[target.candidate_id] = CliAgentExclusion(
                reason=EXCLUDED_DISPATCH_TARGET,
                detail=(
                    f"the agent '{referrer.name}'{scope} declares it dispatches "
                    f"this one (tools: Agent(...) in {referrer.source_path})"
                ),
            )

    return exclusions


__all__ = [
    "EXCLUDED_DIRECT_INVOCATION_DISCLAIMED",
    "EXCLUDED_DISPATCH_TARGET",
    "CliAgentExclusion",
    "classify_candidates",
    "dispatch_references",
]
