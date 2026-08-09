"""Discover Claude Code subagents on this machine and import one as an agent.

A Claude subagent (YAML frontmatter + a markdown body that is its system
prompt) already exists as a file the user wrote and trusts, but it only ever
runs when a human invokes it from a Claude Code session — no schedule, no
result history, no notification. Code Bridge has all of that. This module is
the bridge: it walks the places subagent files live, parses what it finds,
and turns a chosen one into a Code Bridge agent that *points at* that file.

Importing does not copy the prompt. The file is read again on every run and
handed to the Claude session as an agent definition (see
:mod:`agent.subagent_runtime`), so what executes is what the file says today
and the ``tools:`` the author declared are actually enforced. What this module
writes into the agent record is a reference stub
(:func:`subagent_reference_prompt`) plus the one-step workflow the rest of the
system expects an agent to have.

If this regresses, three different things break for the user:

- **Discovery drift** (``discover_subagent_candidates``): a real subagent file
  stops showing up as importable, or a malformed one crashes the whole sweep
  instead of being reported and skipped. Either way "let the server discover
  them" silently stops being true, and the failure is invisible until someone
  notices an agent they expected to import just isn't in the list.
- **Offering agents that cannot run alone**: most installed subagents are
  workers a parent dispatches with a payload, and scheduling one produces a
  refusal every night. They are filtered out — and reported in ``excluded``
  with a reason, because an agent silently missing from the list is its own
  bug. See :mod:`agent.subagent_eligibility`.
- **Import drift** (``import_subagent``): importing produces an agent that
  cannot run (empty ``system_prompt``, no workflow step, so the readiness rail
  reads "steps: not ready"), or importing the same file twice quietly creates
  a second agent instead of returning the first one.

Three discovery locations, matching how Claude Code itself resolves
subagents:

- **User**: ``~/.claude/agents/*.md``
- **Project**: ``<project>/.claude/agents/*.md`` for the server's own working
  directory and every project Code Bridge has registered (``ProjectDB``) —
  Code Bridge manages many projects, and a subagent authored for one of them
  lives in that project's own ``.claude/agents``, not the server's.
- **Plugin**: ``~/.claude/plugins/marketplaces/*/plugins/*/agents/*.md`` —
  the only location with real content on the machine this was built against;
  user and project dirs are commonly empty or absent entirely.

Codex (``~/.codex/automations/``) and Antigravity have no discoverable agent
format as of this writing (the former is an empty directory; the latter has
no filesystem convention at all). Nothing here guesses at a format for them —
:func:`discover_subagent_candidates` only walks the three Claude locations
above, and a fourth source becomes a fourth entry in
``_all_source_locations`` once its format is actually known, not before.

Model / provider mapping for ``inherit``
-----------------------------------------
A subagent's ``model: inherit`` means "use whatever model the parent Claude
Code session is already running" — it is not a model id Code Bridge's LLM
layer can send anywhere. Two honest options exist: resolve it to Code
Bridge's *currently* selected model, or leave it unset. This module leaves it
unset (``provider_id=None``, ``model=None``) rather than resolving it,
because an imported agent is typically run much later, unattended, on a
schedule — freezing today's global selection at import time would make the
agent silently drift from whatever the user has since switched to, while
leaving it unset lets the existing runtime fallback
(``task_orchestrator._resolve_provider_selection``) resolve the live
selection fresh on every run, which is both simpler to implement and more
correct for a recurring background agent. A concrete (non-``inherit``) model
value in the frontmatter — ``sonnet``, ``opus``, ``haiku`` — is a different
case: those are Claude-specific aliases, not ambiguous, so they are passed
through verbatim with ``provider_id`` pinned to ``"anthropic"``.

``effort`` / ``color`` have no equivalent field in Code Bridge's agent schema
and are deliberately dropped at import time rather than stuffed into
``policy_overrides_json`` — that column is reserved for approval-gating
overrides, and decorative Claude Code UI metadata (a reasoning-effort hint,
a display color) sitting in it risks a future reader mistaking it for a real
rule. They are visible in the discovery listing (:meth:`SubagentCandidate.to_view`)
so a human can see them before deciding to import, but they do not survive
into the created agent record.

Re-import is a no-op, not a duplicate: :func:`import_subagent` records
``source_path -> agent_id`` in ``agent_subagent_imports`` (unique index on
``source_path``, see ``core.database._migrate_subagent_imports``) and a
second import of the same file returns the first agent unchanged — even if
that agent has since been archived. Hard-deleting the agent (not archiving
it) removes the mapping too, via ``delete_import_record_for_agent``, so the
file becomes importable again.

That same table is where the run-time source reference lives: read the other
way round (:func:`find_import_source_path_for_agent`) it answers "which file
does this agent run". Nothing duplicates the path onto the ``agents`` row, so
there is one place that can be wrong, and deleting the agent cannot leave a
pointer behind.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.database import get_db_connection
from core.timestamps import to_utc_iso

from .subagent_eligibility import classify_candidates
from .workflow_v2 import WorkflowNormalizationError, normalize_workflow

logger = logging.getLogger(__name__)

# --- Skip reasons -----------------------------------------------------------
#
# A file that cannot be parsed is reported with one of these, never silently
# dropped and never converted into a plausible-looking candidate.

SKIP_NOT_READABLE = "not_readable"
SKIP_NO_FRONTMATTER = "no_frontmatter"
SKIP_UNTERMINATED_FRONTMATTER = "unterminated_frontmatter"
SKIP_INVALID_YAML = "invalid_yaml"
SKIP_FRONTMATTER_NOT_A_MAPPING = "frontmatter_not_a_mapping"
SKIP_MISSING_NAME = "missing_name"


class SubagentImportError(ValueError):
    """Raised when a requested source path cannot be imported as described."""


@dataclass
class SubagentCandidate:
    """One subagent file a fresh sweep parsed successfully."""

    source_path: str
    candidate_id: str
    location: str
    name: str
    description: str
    tools: list[str]
    model: str | None
    effort: str | None
    color: str | None
    body: str

    def to_view(self) -> dict[str, Any]:
        """Metadata enough to choose from — never the body (the system prompt)."""
        return {
            "candidate_id": self.candidate_id,
            "source_path": self.source_path,
            "location": self.location,
            "name": self.name,
            "description": self.description,
            "tools": list(self.tools),
            "model": self.model,
            "effort": self.effort,
            "color": self.color,
        }


@dataclass
class SkippedSubagentFile:
    """One file the sweep found but could not parse, and why."""

    source_path: str
    location: str
    reason: str
    detail: str

    def to_view(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "location": self.location,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class ExcludedSubagent:
    """One parsed subagent that is not standalone-runnable, and why.

    Reported, never dropped: a user who knows a subagent is installed must be
    able to find out why it is not on the import list. See
    :mod:`agent.subagent_eligibility` for how the reason is decided.
    """

    candidate: SubagentCandidate
    reason: str
    detail: str

    def to_view(self) -> dict[str, Any]:
        return {
            **self.candidate.to_view(),
            "excluded_reason": self.reason,
            "excluded_detail": self.detail,
        }


@dataclass
class SubagentDiscoverySweep:
    """Result of one discovery pass.

    ``candidates`` is only what can actually be run unattended. Everything the
    sweep parsed but will not offer is in ``excluded`` with its reason, and
    everything it could not parse is in ``skipped``.
    """

    candidates: list[SubagentCandidate] = field(default_factory=list)
    skipped: list[SkippedSubagentFile] = field(default_factory=list)
    excluded: list[ExcludedSubagent] = field(default_factory=list)


@dataclass
class SubagentImportResult:
    """Result of one import attempt."""

    agent: dict[str, Any]
    created: bool
    reason: str  # "imported" | "already_imported"


# --- Discovery locations -----------------------------------------------------


def _user_agents_dir() -> Path:
    return Path.home() / ".claude" / "agents"


def _project_agent_dirs() -> list[tuple[str, Path]]:
    """The server's own working directory plus every registered project.

    Best-effort: a project-registry read failure (e.g. the DB is mid-
    migration in a test) drops project-scoped discovery for this sweep
    rather than aborting it — the user/plugin locations still work.
    """
    seen: dict[Path, str] = {}
    cwd = Path.cwd()
    seen[cwd.resolve()] = f"project:{cwd.name}"
    try:
        from core.database import get_project_db

        for project in get_project_db().get_all():
            raw_path = project.get("path")
            name = project.get("name")
            if not raw_path or not name:
                continue
            resolved = Path(str(raw_path)).expanduser().resolve()
            seen.setdefault(resolved, f"project:{name}")
    except Exception:
        logger.debug(
            "subagent discovery: could not read project registry", exc_info=True
        )
    return [(label, path / ".claude" / "agents") for path, label in seen.items()]


def _plugin_agent_dirs() -> list[tuple[str, Path]]:
    """``<marketplace>/plugins/<plugin>/agents`` for every installed plugin."""
    root = Path.home() / ".claude" / "plugins" / "marketplaces"
    dirs: list[tuple[str, Path]] = []
    if not root.is_dir():
        return dirs
    try:
        marketplace_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return dirs
    for marketplace_dir in marketplace_dirs:
        plugins_dir = marketplace_dir / "plugins"
        if not plugins_dir.is_dir():
            continue
        try:
            plugin_dirs = sorted(p for p in plugins_dir.iterdir() if p.is_dir())
        except OSError:
            continue
        for plugin_dir in plugin_dirs:
            agents_dir = plugin_dir / "agents"
            if agents_dir.is_dir():
                dirs.append(
                    (f"plugin:{marketplace_dir.name}/{plugin_dir.name}", agents_dir)
                )
    return dirs


def _all_source_locations() -> list[tuple[str, Path]]:
    """Every ``(location_label, directory)`` pair a sweep walks, in priority order."""
    locations: list[tuple[str, Path]] = [("user", _user_agents_dir())]
    locations.extend(_project_agent_dirs())
    locations.extend(_plugin_agent_dirs())
    return locations


# --- Parsing ------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[str, str] | None:
    """Split ``---\\nYAML\\n---\\nbody`` into ``(yaml_text, body)``.

    Returns ``None`` if the file does not open with a frontmatter block at
    all, and also if it opens with one but never closes it — callers tell
    the two apart by re-checking the first line, so the reported skip reason
    says which happened.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return None


def _clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _split_top_level_commas(text: str) -> list[str]:
    """Split on commas that are not inside parentheses.

    ``Agent(a, b)`` is one tool declaration naming two subagents, not three
    tools. Splitting naively produced ``"Agent(claude-security:scan-inventory"``
    and ``"claude-security:explore)"`` — fragments that are neither a tool the
    CLI would accept when the definition is handed back to it, nor something
    the dispatch-graph reader could resolve.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _parse_tools(raw: Any) -> list[str]:
    """Accept the two shapes real frontmatter uses: a YAML list, or a
    comma-separated string (``"Bash, Agent(claude-security:explore)"``)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return _split_top_level_commas(raw)
    text = str(raw).strip()
    return [text] if text else []


def _candidate_id(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
    return f"subagent_{digest}"


def _parse_frontmatter_leniently(raw: str) -> dict[str, Any] | None:
    """Read `key: value` frontmatter that strict YAML refuses.

    Only used after :func:`yaml.safe_load` has already failed. These blocks
    are flat — every field a subagent declares is a scalar — so splitting each
    line on its *first* colon recovers the author's intent without guessing:
    a colon later in the line belongs to the prose, which is precisely what
    YAML got wrong.

    Returns None when the block does not look like `key: value` lines at all,
    so a genuinely broken file is still reported as broken rather than
    quietly reduced to whatever happened to parse.
    """
    fields: dict[str, Any] = {}
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") or ":" not in stripped:
            # A list item or a continuation line — this is not the flat
            # shape this fallback claims to understand.
            return None
        key, _, value = stripped.partition(":")
        key = key.strip()
        if not key or key != key.split()[0]:
            return None
        fields[key] = value.strip().strip("'\"")
    return fields or None


def _parse_subagent_file(
    path: Path, *, location: str
) -> SubagentCandidate | SkippedSubagentFile:
    resolved = str(path.resolve())
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return SkippedSubagentFile(resolved, location, SKIP_NOT_READABLE, str(exc))

    split = _split_frontmatter(text)
    if split is None:
        lines = text.split("\n")
        if lines and lines[0].strip() == "---":
            return SkippedSubagentFile(
                resolved,
                location,
                SKIP_UNTERMINATED_FRONTMATTER,
                "no closing '---' line found",
            )
        return SkippedSubagentFile(
            resolved,
            location,
            SKIP_NO_FRONTMATTER,
            "file does not start with a '---' YAML frontmatter block",
        )

    raw_frontmatter, body = split
    try:
        parsed = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError as exc:
        # Strict YAML is stricter than these files actually are. A real
        # shipped agent (pr-review-toolkit/silent-failure-hunter) writes a
        # long unquoted description containing "Context: Daisy has just…" —
        # a colon-space inside a plain scalar, which YAML reads as a nested
        # mapping and rejects. Claude Code loads that file fine, so calling
        # it malformed would hide a working agent behind a parser detail the
        # author never had to think about.
        parsed = _parse_frontmatter_leniently(raw_frontmatter)
        if parsed is None:
            return SkippedSubagentFile(resolved, location, SKIP_INVALID_YAML, str(exc))

    if not isinstance(parsed, dict):
        return SkippedSubagentFile(
            resolved,
            location,
            SKIP_FRONTMATTER_NOT_A_MAPPING,
            f"frontmatter parsed as {type(parsed).__name__}, expected a mapping",
        )

    name = str(parsed.get("name") or "").strip()
    if not name:
        return SkippedSubagentFile(
            resolved,
            location,
            SKIP_MISSING_NAME,
            "frontmatter has no non-empty 'name' field",
        )

    return SubagentCandidate(
        source_path=resolved,
        candidate_id=_candidate_id(path),
        location=location,
        name=name,
        description=_clean_scalar(parsed.get("description")) or "",
        tools=_parse_tools(parsed.get("tools")),
        model=_clean_scalar(parsed.get("model")),
        effort=_clean_scalar(parsed.get("effort")),
        color=_clean_scalar(parsed.get("color")),
        body=body.strip("\n"),
    )


def discover_subagent_candidates() -> SubagentDiscoverySweep:
    """Walk every known location once and parse every ``*.md`` file found.

    A file reachable from two locations at once (e.g. the server's cwd is
    also a registered project) is only parsed once, keeping whichever
    location was checked first (user, then project, then plugin).

    Parsed files are then split: the ones that can be run on their own land in
    ``candidates``, the ones that exist to be dispatched by something else land
    in ``excluded`` with the reason (see :mod:`agent.subagent_eligibility`).
    Scheduling a dispatch-dependent subagent produces a refusal every night, so
    they are not offered — but they are still reported, because "the agent I
    installed is missing from the list" with no explanation is its own bug.
    """
    parsed: list[SubagentCandidate] = []
    skipped: list[SkippedSubagentFile] = []
    seen_paths: set[str] = set()

    for location, directory in _all_source_locations():
        if not directory.is_dir():
            continue
        try:
            files = sorted(directory.glob("*.md"))
        except OSError:
            logger.debug(
                "subagent discovery: could not list %s", directory, exc_info=True
            )
            continue
        for file_path in files:
            resolved = str(file_path.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            result = _parse_subagent_file(file_path, location=location)
            if isinstance(result, SubagentCandidate):
                parsed.append(result)
            else:
                skipped.append(result)

    exclusions = classify_candidates(parsed)
    candidates: list[SubagentCandidate] = []
    excluded: list[ExcludedSubagent] = []
    for candidate in parsed:
        exclusion = exclusions.get(candidate.candidate_id)
        if exclusion is None:
            candidates.append(candidate)
        else:
            excluded.append(
                ExcludedSubagent(
                    candidate=candidate,
                    reason=exclusion.reason,
                    detail=exclusion.detail,
                )
            )

    return SubagentDiscoverySweep(
        candidates=candidates, skipped=skipped, excluded=excluded
    )


# --- Conversion to a Code Bridge agent -----------------------------------------


def _resolve_model_and_provider(raw_model: str | None) -> tuple[str | None, str | None]:
    """See the module docstring's "Model / provider mapping" section."""
    if raw_model and raw_model.strip().lower() != "inherit":
        return "anthropic", raw_model.strip()
    return None, None


def subagent_reference_prompt(name: str, source_path: str) -> str:
    """The text stored where a copy of the subagent's prompt used to sit.

    An imported subagent runs *by reference*: the file is read fresh on every
    run and handed to the Claude session as an agent definition, so the prompt
    and the declared tools that execute are always the ones in the file right
    now. Storing a copy of the body here would be a copy nothing reads — shown
    in the agent editor as if editing it changed something, and stale the
    moment the author touches the file. This stub is what the rest of the
    system sees instead, and it says so in the text, because an agent whose
    prompt box silently does nothing is worse than one that explains itself.
    """
    return (
        f"This agent runs the Claude Code subagent '{name}', defined at "
        f"{source_path}.\n\n"
        "That file is read at the start of every run and its prompt, declared "
        "tools, and model are what actually execute. Editing this text changes "
        "nothing — edit the file. If the file is missing or unparseable when a "
        "run starts, the run fails naming it rather than running anything else."
    )


def candidate_to_agent_create_kwargs(candidate: SubagentCandidate) -> dict[str, Any]:
    """Build the ``AgentStore.create_agent`` kwargs for one candidate.

    Mapping: ``name``->``name``, ``description``->``description``,
    ``tools``->``tools_json`` (the raw Claude Code tool/agent names, carried
    through as a plain list of strings — they are not Code Bridge MCP
    capability ids, so storing them verbatim is honest rather than silently
    inventing an MCP mapping that does not exist; at run time the same names go
    to the Claude session, which is what enforces them).

    What does *not* get mapped is the body. The rest of the system expects an
    agent to have a system prompt and a workflow, so both exist — a single
    ``llm`` step, the honest shape of a one-prompt subagent — but they carry
    :func:`subagent_reference_prompt`, not a copy of the file. The file is read
    on every run (``agent.subagent_runtime``). A copy here would drift from it
    silently, and the drift is invisible precisely because the copy is the part
    a human sees in the editor.
    """
    provider_id, model = _resolve_model_and_provider(candidate.model)
    name = candidate.name[:80]
    reference = subagent_reference_prompt(candidate.name, candidate.source_path)
    step: dict[str, Any] = {
        "id": "subagent_instruction",
        "type": "llm",
        "name": name or "Imported subagent step",
        "instruction": reference,
    }
    if candidate.description:
        step["description"] = candidate.description[:2000]

    return {
        "name": name,
        "description": candidate.description or None,
        "system_prompt": reference,
        "provider_id": provider_id,
        "model": model,
        "tools_json": list(candidate.tools),
        "flow_json": [step],
        "policy_overrides_json": {},
    }


# --- Import-provenance store (dedup) -------------------------------------------


def _find_import_by_source_path(source_path: str) -> dict[str, Any] | None:
    with get_db_connection(use_row_factory=True) as conn:
        row = conn.execute(
            "SELECT * FROM agent_subagent_imports WHERE source_path = ?",
            (source_path,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "source_path": row["source_path"],
        "agent_id": row["agent_id"],
        "imported_at": to_utc_iso(row["imported_at"]),
    }


def find_import_source_path_for_agent(agent_id: str) -> str | None:
    """The subagent file this Code Bridge agent runs, or None if it is not one.

    This mapping row *is* the source reference — there is no copy of the path
    on the agent record, so there is exactly one place that can be wrong. It
    was already written at import time (commit 223ca57) purely to make a
    re-import a no-op; reading it back at run time is what turns an imported
    agent from a frozen copy into a pointer at a live file.
    """
    with get_db_connection(use_row_factory=True) as conn:
        row = conn.execute(
            """
            SELECT source_path FROM agent_subagent_imports
            WHERE agent_id = ?
            ORDER BY imported_at DESC
            LIMIT 1
            """,
            (agent_id,),
        ).fetchone()
    if row is None:
        return None
    source_path = row["source_path"]
    return str(source_path) if source_path else None


def _record_import(source_path: str, agent_id: str) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_subagent_imports (id, source_path, agent_id)
            VALUES (?, ?, ?)
            """,
            (f"subimport_{uuid.uuid4().hex}", source_path, agent_id),
        )
        conn.commit()


def _delete_import_record_by_source_path(source_path: str) -> None:
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM agent_subagent_imports WHERE source_path = ?",
            (source_path,),
        )
        conn.commit()


def delete_import_record_for_agent(agent_id: str) -> None:
    """Drop the import-provenance row(s) for a hard-deleted agent.

    Call this from the agent hard-delete path (``AgentStore.archive_agent``
    with ``archive=False``) — not from a soft archive. A real delete means
    the agent record is gone for good, so the mapping to it is stale; leaving
    it behind would permanently block re-importing that file. An archive is
    reversible and the mapping should keep pointing at the archived agent.
    """
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM agent_subagent_imports WHERE agent_id = ?",
            (agent_id,),
        )
        conn.commit()


# --- Import ---------------------------------------------------------------------


def import_subagent(source_path: str) -> SubagentImportResult:
    """Import one discovered subagent file as a Code Bridge agent.

    Re-runs discovery and only proceeds on an exact match against a path a
    fresh sweep both found and could parse — a path that is not currently
    discoverable (wrong location, moved, deleted, or simply never a subagent
    file) is refused rather than read a second time on the caller's say-so.
    This is what keeps the endpoint from being an arbitrary-file-read
    primitive for whatever path a client sends.

    Re-importing the same file is a no-op: the previously created agent is
    returned with ``created=False`` instead of a duplicate being made, even
    if that agent has since been archived.
    """
    from .agent_store import get_agent_store

    resolved = str(Path(source_path).expanduser().resolve())

    existing = _find_import_by_source_path(resolved)
    if existing is not None:
        store = get_agent_store()
        agent = store.get_agent(existing["agent_id"])
        if agent is not None:
            return SubagentImportResult(agent=agent, created=False, reason="already_imported")
        # The mapped agent record is gone (a hard delete that did not go
        # through delete_import_record_for_agent) — the mapping is stale,
        # not authoritative. Drop it and import fresh.
        _delete_import_record_by_source_path(resolved)

    sweep = discover_subagent_candidates()
    candidate = next((c for c in sweep.candidates if c.source_path == resolved), None)
    if candidate is None:
        excluded = next(
            (e for e in sweep.excluded if e.candidate.source_path == resolved), None
        )
        if excluded is not None:
            # Not "unknown path" — this file was found and parsed fine. Saying
            # so, with the reason, is the difference between "Code Bridge lost
            # my agent" and "this agent only runs when something dispatches it".
            raise SubagentImportError(
                f"'{source_path}' is not standalone-runnable and would refuse "
                f"every scheduled run ({excluded.reason}: {excluded.detail})"
            )
        skip = next((s for s in sweep.skipped if s.source_path == resolved), None)
        if skip is not None:
            raise SubagentImportError(
                f"'{source_path}' was found but could not be parsed as a subagent "
                f"({skip.reason}: {skip.detail})"
            )
        raise SubagentImportError(
            f"'{source_path}' is not a currently discoverable subagent file"
        )

    kwargs = candidate_to_agent_create_kwargs(candidate)
    try:
        kwargs["flow_json"] = normalize_workflow(kwargs["flow_json"])
    except WorkflowNormalizationError as exc:  # pragma: no cover - defensive
        raise SubagentImportError(f"generated workflow failed validation: {exc}") from exc

    store = get_agent_store()
    agent = store.create_agent(**kwargs)
    _record_import(resolved, agent["id"])
    return SubagentImportResult(agent=agent, created=True, reason="imported")


__all__ = [
    "SKIP_FRONTMATTER_NOT_A_MAPPING",
    "SKIP_INVALID_YAML",
    "SKIP_MISSING_NAME",
    "SKIP_NOT_READABLE",
    "SKIP_NO_FRONTMATTER",
    "SKIP_UNTERMINATED_FRONTMATTER",
    "ExcludedSubagent",
    "SubagentCandidate",
    "SubagentDiscoverySweep",
    "SubagentImportError",
    "SubagentImportResult",
    "SkippedSubagentFile",
    "candidate_to_agent_create_kwargs",
    "delete_import_record_for_agent",
    "discover_subagent_candidates",
    "find_import_source_path_for_agent",
    "import_subagent",
    "subagent_reference_prompt",
]
