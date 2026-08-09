"""Look for agent definition files on a schedule, and remember what was there.

Discovery already worked, but only when asked: ``GET /api/agent/cli-agents``
walks the disk on every request and returns what exists *now*
(:mod:`agent.cli_agent_sources`). That is a good answer to "what is there" and
no answer at all to "what changed", because nothing was ever written down. The
server therefore knew about a definition only while a client happened to be
looking at it.

This module is the memory. It runs the same sweep on a timer, diffs the result
against the last one, and stores both the set and the diff. Two things become
possible that were not before:

- **New files are noticed without anyone looking.** An agent the user wrote
  last night shows up in the stored view — and, if auto-import is on, becomes
  a Code Bridge agent — without the app being opened.
- **A vanished file is noticed *before* it costs a run.** An imported agent is
  a pointer, not a copy (:mod:`agent.cli_agent_runtime`): delete the ``.md``
  and the next scheduled run fails naming a file. Unattended, that means 3am,
  and the user finds out from a failure. A sweep that spots the file missing
  says so while there is still time to restore it.

If this regresses, the failures are quiet by construction:

- **The sweep stops running** (nobody calls
  :func:`maybe_run_due_cli_agent_sweep`, or it starts raising): everything
  still works when a client asks, so nothing looks broken — the server simply
  goes back to knowing nothing on its own, and the vanished-file warning stops
  arriving. The only visible symptom is a ``last_sweep`` timestamp that stops
  advancing, which is why the stored view returns it.
- **The seen-set is lost or overwritten by a failed sweep**: every path reads
  as new again, so the next sweep notifies about 25 agents the user has had
  for months, and — with auto-import on — imports them a second time. This is
  why a failed sweep records the failure and leaves ``agent_cli_agent_seen``
  untouched, and why an empty result is never written as a success.
- **Notification spam**: a notification on a sweep where nothing changed
  teaches people to swipe these away, and then the one that matters (your
  agent's source file is gone) gets swiped away too. Nothing here notifies
  unless something genuinely appeared or genuinely disappeared.

Why six hours
-------------
:data:`DEFAULT_SWEEP_INTERVAL_SECONDS` is 6 hours. Definition files are written
by hand — a user adds one every few days at most, and a plugin install that
brings a batch of them is a deliberate act. Sweeping is cheap (~15ms of
``glob`` and small file reads on the machine this was built against) but it is
not free, and it is also not urgent: the two things the sweep exists to catch
tolerate a few hours of latency. Anyone who *does* care right now already has
an instant answer — ``GET /api/agent/cli-agents`` still walks the disk live on
every request, so "I just wrote a file, hit Refresh" works exactly as it did.

Six hours also matches the cadence at which scheduled agents actually run
here, which is the deadline that matters for the missing-file warning: the
warning is only useful if it lands before the run it predicts.

The interval is measured from the last *attempt* (success or failure), not
from the last success, so a sweep that keeps failing retries four times a day
instead of on every 30-second scheduler tick.

Where things live
-----------------
- The seen-set and the last sweep record: ``agent_cli_agent_seen`` and
  ``agent_cli_agent_sweeps`` (see ``core.database._migrate_cli_agent_sweeps``).
- The auto-import switch: the ``app_settings`` key-value table where every
  other server setting already lives (``core.database.SettingsDB``, the same
  place ``allow_ip_login`` is kept), under
  :data:`SETTING_AUTO_IMPORT_ENABLED`. Default off, and it stays default off:
  there are 25 eligible candidates on a developer machine, and each imported
  agent then wants a schedule and standing permissions. That is a decision to
  put in front of someone, not one to make on their behalf.

Whatever that switch says, the sweep still runs and still records. The switch
only decides whether a newly discovered *eligible* candidate is imported, and
it is deliberately not retroactive: turning it on does not reach back and
import the candidates already on record, because that is precisely the "25
agents appeared behind my back" outcome the default exists to prevent.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from core.database import get_db_connection, get_settings_db

from .cli_agent_sources import (
    CliAgentDiscoverySweep,
    CliAgentImportError,
    _find_import_by_source_path,
    discover_cli_agents,
    import_cli_agent,
)

logger = logging.getLogger(__name__)

#: ``app_settings`` key for the optional auto-import switch. Off unless the
#: stored value is exactly ``"true"``.
SETTING_AUTO_IMPORT_ENABLED = "cli_agent_auto_import_enabled"

#: The key this setting used before the feature was renamed. Read as a
#: fallback, never written: a user who deliberately turned auto-import *on*
#: must not have it silently switched off by a rename they never asked for,
#: and the first write under the new key retires the old one for good.
LEGACY_SETTING_AUTO_IMPORT_ENABLED = "subagent_auto_import_enabled"

#: See the module docstring for why six hours.
DEFAULT_SWEEP_INTERVAL_SECONDS = 6 * 60 * 60

#: Override for tests and for anyone who genuinely wants a different cadence.
#: Same shape as ``CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS`` in
#: :mod:`agent.scheduler`.
SWEEP_INTERVAL_ENV = "CODEBRIDGE_CLI_AGENT_SWEEP_INTERVAL_SECONDS"

#: The name this override had before the feature was renamed. Still honoured so
#: an existing deployment's environment does not silently revert to the 6-hour
#: default, which would look like the sweep having stopped.
LEGACY_SWEEP_INTERVAL_ENV = "CODEBRIDGE_SUBAGENT_SWEEP_INTERVAL_SECONDS"

#: The sweeps table holds exactly one row, under this id.
LATEST_SWEEP_ID = "latest"

STATUS_OK = "ok"
STATUS_FAILED = "failed"

STATE_CANDIDATE = "candidate"
STATE_EXCLUDED = "excluded"
STATE_SKIPPED = "skipped"

# How many names a notification body lists before it stops and says "and N
# more". A push notification nobody can read is the same as no notification.
_MAX_NAMES_IN_BODY = 8


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


# --- The auto-import setting ---------------------------------------------------


def get_auto_import_enabled() -> bool:
    """Whether a newly discovered eligible definition is imported automatically.

    Default ``False``, and an unreadable or unrecognised stored value also
    reads as ``False`` — the safe direction for a switch whose "on" state
    creates agents. The pre-rename key is consulted only when the current one
    has never been written, so an existing choice survives the rename.
    """
    try:
        settings = get_settings_db()
        raw = settings.get(SETTING_AUTO_IMPORT_ENABLED, None)
        if raw is None:
            raw = settings.get(LEGACY_SETTING_AUTO_IMPORT_ENABLED, "false")
    except Exception:  # noqa: BLE001 - a settings read must not break a sweep
        logger.debug(
            "cli agent sweep: could not read auto-import setting", exc_info=True
        )
        return False
    return str(raw or "").strip().lower() == "true"


def set_auto_import_enabled(enabled: bool) -> bool:
    """Turn auto-import on or off. Returns the new value."""
    get_settings_db().set(SETTING_AUTO_IMPORT_ENABLED, "true" if enabled else "false")
    return bool(enabled)


def sweep_interval_seconds() -> int:
    """The configured interval, floored at 60s so no override can busy-loop."""
    raw = os.environ.get(SWEEP_INTERVAL_ENV)
    if raw is None:
        raw = os.environ.get(LEGACY_SWEEP_INTERVAL_ENV)
    if raw is None:
        return DEFAULT_SWEEP_INTERVAL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SWEEP_INTERVAL_SECONDS
    return max(60, value)


# --- The seen-set ---------------------------------------------------------------


@dataclass(frozen=True)
class SeenEntry:
    """One source path as this sweep found it."""

    source_path: str
    candidate_id: str
    location: str
    name: str
    description: str
    state: str
    detail: str | None

    def to_view(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "candidate_id": self.candidate_id,
            "location": self.location,
            "name": self.name,
            "description": self.description,
            "state": self.state,
            "detail": self.detail,
        }


def _entries_from_sweep(sweep: CliAgentDiscoverySweep) -> list[SeenEntry]:
    """Flatten a discovery sweep into one entry per file, keeping the verdict.

    All three buckets are remembered, not just the importable ones. A file that
    currently fails to parse is still a file that exists, and if an agent was
    imported from it back when it parsed, its later *disappearance* is the
    event worth warning about — which cannot be detected if unparseable files
    were never recorded as present.
    """
    entries: list[SeenEntry] = []
    for candidate in sweep.candidates:
        entries.append(
            SeenEntry(
                source_path=candidate.source_path,
                candidate_id=candidate.candidate_id,
                location=candidate.location,
                name=candidate.name,
                description=candidate.description,
                state=STATE_CANDIDATE,
                detail=None,
            )
        )
    for excluded in sweep.excluded:
        entries.append(
            SeenEntry(
                source_path=excluded.candidate.source_path,
                candidate_id=excluded.candidate.candidate_id,
                location=excluded.candidate.location,
                name=excluded.candidate.name,
                description=excluded.candidate.description,
                state=STATE_EXCLUDED,
                detail=f"{excluded.reason}: {excluded.detail}",
            )
        )
    for skipped in sweep.skipped:
        entries.append(
            SeenEntry(
                source_path=skipped.source_path,
                candidate_id="",
                location=skipped.location,
                name="",
                description="",
                state=STATE_SKIPPED,
                detail=f"{skipped.reason}: {skipped.detail}",
            )
        )
    return entries


def _seen_row(row: Any) -> dict[str, Any]:
    return {
        "source_path": row["source_path"],
        "candidate_id": row["candidate_id"],
        "location": row["location"],
        "name": row["name"],
        "description": row["description"],
        "state": row["state"],
        "detail": row["detail"],
        "first_seen_at": row["first_seen_at"],
        "last_seen_at": row["last_seen_at"],
        "missing_since": row["missing_since"],
    }


def load_seen() -> dict[str, dict[str, Any]]:
    """Every source path ever recorded, keyed by path."""
    with get_db_connection(use_row_factory=True) as conn:
        rows = conn.execute("SELECT * FROM agent_cli_agent_seen").fetchall()
    return {str(row["source_path"]): _seen_row(row) for row in rows}


def _record_present(entries: list[SeenEntry], *, at: str) -> None:
    """Upsert everything this sweep found, clearing any earlier ``missing_since``."""
    if not entries:
        return
    with get_db_connection() as conn:
        conn.executemany(
            """
            INSERT INTO agent_cli_agent_seen (
                source_path, candidate_id, location, name, description,
                state, detail, first_seen_at, last_seen_at, missing_since
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(source_path) DO UPDATE SET
                candidate_id = excluded.candidate_id,
                location = excluded.location,
                name = excluded.name,
                description = excluded.description,
                state = excluded.state,
                detail = excluded.detail,
                last_seen_at = excluded.last_seen_at,
                missing_since = NULL
            """,
            [
                (
                    entry.source_path,
                    entry.candidate_id,
                    entry.location,
                    entry.name,
                    entry.description,
                    entry.state,
                    entry.detail,
                    at,
                    at,
                )
                for entry in entries
            ],
        )
        conn.commit()


def _record_missing(source_paths: list[str], *, at: str) -> None:
    """Stamp ``missing_since`` on paths that were present last time and are not now.

    ``missing_since IS NULL`` in the WHERE clause keeps the *first* time a path
    went missing, so a file gone for a week does not read as freshly gone on
    every sweep — and, more importantly, does not re-notify every six hours.
    """
    if not source_paths:
        return
    with get_db_connection() as conn:
        conn.executemany(
            """
            UPDATE agent_cli_agent_seen SET missing_since = ?
            WHERE source_path = ? AND missing_since IS NULL
            """,
            [(at, path) for path in source_paths],
        )
        conn.commit()


# --- The last-sweep record --------------------------------------------------------


def _store_sweep_record(record: dict[str, Any]) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_cli_agent_sweeps (
                id, status, trigger, started_at, finished_at, error, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                trigger = excluded.trigger,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                error = excluded.error,
                result_json = excluded.result_json
            """,
            (
                LATEST_SWEEP_ID,
                record["status"],
                record.get("trigger"),
                record["started_at"],
                record.get("finished_at"),
                json.dumps(record.get("error")) if record.get("error") else None,
                json.dumps(record),
            ),
        )
        conn.commit()


def get_last_sweep_record() -> dict[str, Any] | None:
    """The most recent sweep attempt, or ``None`` if the server never swept."""
    with get_db_connection(use_row_factory=True) as conn:
        row = conn.execute(
            "SELECT * FROM agent_cli_agent_sweeps WHERE id = ?", (LATEST_SWEEP_ID,)
        ).fetchone()
    if row is None:
        return None
    raw = row["result_json"]
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError):
            logger.warning("cli agent sweep: stored record is not readable JSON")
    # The blob is unreadable but the columns are not — report what is certain
    # rather than pretending there was never a sweep.
    return {
        "status": row["status"],
        "trigger": row["trigger"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": {"type": "UnreadableRecord", "message": str(row["error"] or "")},
        "counts": None,
        "new": [],
        "missing": [],
    }


def get_stored_view() -> dict[str, Any]:
    """Everything a client can know without touching the filesystem.

    ``known`` is the durable set from :func:`load_seen` — what the server
    believes is on disk — split by verdict, plus the paths currently recorded
    as missing. ``last_sweep`` is when the server last looked and what changed
    then. The two are separate on purpose: a failed sweep replaces
    ``last_sweep`` with a failure and leaves ``known`` exactly as it was, so
    "the walk broke" never renders as "you have no agents".

    This is a summary, not a substitute for ``GET /api/agent/cli-agents``: it
    carries what identifies an agent (name, description, path, location,
    verdict) and not the fuller per-candidate metadata a live sweep returns.
    """
    known: dict[str, list[dict[str, Any]]] = {
        "candidates": [],
        "excluded": [],
        "skipped": [],
        "missing": [],
    }
    for entry in sorted(load_seen().values(), key=lambda row: str(row["source_path"])):
        if entry.get("missing_since"):
            known["missing"].append(entry)
        elif entry.get("state") == STATE_CANDIDATE:
            known["candidates"].append(entry)
        elif entry.get("state") == STATE_EXCLUDED:
            known["excluded"].append(entry)
        else:
            known["skipped"].append(entry)

    return {
        "last_sweep": get_last_sweep_record(),
        "interval_seconds": sweep_interval_seconds(),
        "auto_import_enabled": get_auto_import_enabled(),
        "known": known,
    }


# --- Notifications ------------------------------------------------------------------


def _notify_best_effort(
    *, title: str, body: str, level: str, agent_id: str | None = None
) -> bool:
    """Store a notification and try to push it. Never raises.

    Same contract as ``task_orchestrator._notify_waiting_for_user_best_effort``:
    the thing being reported has already happened and is already recorded by
    the time this runs, so a notification-store failure or a dead FCM key must
    change nothing about it. The worst acceptable outcome is the user reading
    it in the app instead of on a lock screen.
    """
    try:
        from agent.notification_store import get_notification_store

        notification = get_notification_store().create(
            title=title, body=body, level=level, agent_id=agent_id
        )
        try:
            from agent.task_orchestrator import _push_notification_best_effort

            _push_notification_best_effort(
                notification=notification, title=title, body=body, level=level
            )
        except Exception:
            logger.exception(
                "cli agent sweep: push failed; the notification is still stored"
            )
        return True
    except Exception:
        logger.exception(
            "cli agent sweep: could not record a notification; the sweep result stands"
        )
        return False


def _name_lines(entries: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"- {entry.get('name') or entry.get('source_path')} ({entry.get('location')})"
        for entry in entries[:_MAX_NAMES_IN_BODY]
    ]
    remaining = len(entries) - len(lines)
    if remaining > 0:
        lines.append(f"…and {remaining} more")
    return lines


def _notify_new_candidates(
    new_candidates: list[dict[str, Any]], imported: list[dict[str, Any]]
) -> bool:
    """Tell the user about agents that can be imported and were not there before.

    Only *importable* new files ring. A newly appeared file that is excluded
    (it exists to be dispatched by a parent) or unparseable is recorded in the
    stored view but does not notify: neither is something the user can act on,
    and a half-written markdown file mid-edit would otherwise page them every
    six hours. That is the noise that trains people to ignore the one
    notification here that really matters.
    """
    if not new_candidates:
        return False
    count = len(new_candidates)
    if count == 1:
        title = f"New CLI agent found: {new_candidates[0].get('name') or 'unnamed'}"
    else:
        title = f"{count} new CLI agents found"
    lines = _name_lines(new_candidates)
    if imported:
        lines.append("")
        lines.append(
            f"Auto-import is on — {len(imported)} of these "
            f"{'was' if len(imported) == 1 else 'were'} imported as Code Bridge agents."
        )
    else:
        lines.append("")
        lines.append("Import one from the agents screen to give it a schedule.")
    return _notify_best_effort(title=title, body="\n".join(lines), level="info")


def _notify_missing_import_sources(missing: list[dict[str, Any]]) -> int:
    """Warn once per agent whose source file has disappeared.

    This is the only case where a file the sweep *cannot* find has already
    been turned into something that will run: an imported agent holds a
    pointer to the file, not a copy, so the next scheduled run fails naming
    it. Warning now is the difference between fixing it in the morning and
    reading about it in a 3am failure.

    A missing file nothing was imported from does not notify. It is recorded
    in the stored view — the user can see it went away — but nothing is going
    to break because of it, and a warning about a file with no consequences is
    the kind that gets ignored.
    """
    sent = 0
    for entry in missing:
        agent_id = entry.get("imported_agent_id")
        if not agent_id:
            continue
        agent_name = entry.get("imported_agent_name") or "An imported agent"
        title = f"{agent_name} lost its definition file"
        body = "\n".join(
            [
                f"{entry.get('source_path')} was found by an earlier sweep and is "
                "now gone.",
                "",
                "This agent runs that file by reference — it holds a pointer, not "
                "a copy — so its next run will fail naming the file. Restore it, "
                "or delete the agent.",
            ]
        )
        if _notify_best_effort(title=title, body=body, level="error", agent_id=str(agent_id)):
            sent += 1
    return sent


# --- Auto-import ----------------------------------------------------------------------


def _auto_import(new_candidates: list[dict[str, Any]]) -> tuple[list, list]:
    """Import each newly discovered eligible candidate, one failure at a time.

    Reuses :func:`agent.cli_agent_sources.import_cli_agent` unchanged, which
    means each import re-runs discovery to validate the path it was handed —
    the same guard that stops the HTTP import route being an arbitrary-file
    read. That costs one extra walk per import, which is the right trade for
    not having a second, differently-validated import path.

    Idempotent across sweeps twice over: a path recorded in the seen-set is no
    longer "new", so it is not offered here again; and if it were,
    ``import_cli_agent`` returns the existing agent rather than creating a
    second one.
    """
    imported: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for entry in new_candidates:
        source_path = str(entry.get("source_path") or "")
        if not source_path:
            continue
        try:
            result = import_cli_agent(source_path)
        except CliAgentImportError as exc:
            failed.append(
                {"source_path": source_path, "name": entry.get("name"), "error": str(exc)}
            )
            continue
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the sweep
            logger.exception("cli agent sweep: auto-import failed for %s", source_path)
            failed.append(
                {
                    "source_path": source_path,
                    "name": entry.get("name"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        imported.append(
            {
                "source_path": source_path,
                "name": entry.get("name"),
                "agent_id": (result.agent or {}).get("id"),
                "created": result.created,
            }
        )
    return imported, failed


# --- The sweep -------------------------------------------------------------------------


def _importing_agent(source_path: str) -> tuple[str | None, str | None]:
    """``(agent_id, agent_name)`` for the agent imported from this path, if any."""
    try:
        record = _find_import_by_source_path(source_path)
    except Exception:  # noqa: BLE001 - provenance lookup must not end the sweep
        logger.debug(
            "cli agent sweep: could not read import provenance for %s",
            source_path,
            exc_info=True,
        )
        return None, None
    if not record:
        return None, None
    agent_id = str(record.get("agent_id") or "") or None
    if agent_id is None:
        return None, None
    try:
        from .agent_store import get_agent_store

        agent = get_agent_store().get_agent(agent_id)
    except Exception:  # noqa: BLE001
        logger.debug("cli agent sweep: could not read agent %s", agent_id, exc_info=True)
        return agent_id, None
    name = agent.get("name") if isinstance(agent, dict) else None
    return agent_id, (str(name) if name else None)


def run_cli_agent_sweep(*, trigger: str = "scheduled") -> dict[str, Any]:
    """Walk the disk once, diff against memory, record, notify, maybe import.

    Always writes a record. On failure that record says ``status: "failed"``
    with the reason and leaves the seen-set untouched — a sweep that could not
    look must never be stored as a sweep that looked and found nothing, because
    an empty list reads as "you have no agents", which is a different and
    much worse statement than "I could not check".
    """
    started = _now()
    started_iso = _iso(started)
    auto_import_enabled = get_auto_import_enabled()

    try:
        sweep = discover_cli_agents()
    except Exception as exc:  # noqa: BLE001 - the failure is the result here
        logger.exception("cli agent sweep: discovery failed")
        finished = _now()
        record = {
            "status": STATUS_FAILED,
            "trigger": trigger,
            "started_at": started_iso,
            "finished_at": _iso(finished),
            "duration_ms": int((finished - started).total_seconds() * 1000),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            # Not zeros: nothing was counted, and a count of zero would claim
            # the sweep looked and found nothing.
            "counts": None,
            "new": [],
            "missing": [],
            "auto_import": {
                "enabled": auto_import_enabled,
                "imported": [],
                "failed": [],
            },
            "notified": {"new": False, "missing_imported": 0},
        }
        _store_sweep_record(record)
        return record

    entries = _entries_from_sweep(sweep)
    present_paths = {entry.source_path for entry in entries}
    previous = load_seen()

    # "New" means "not present at the last sweep" — a path with no row at all,
    # or one that had gone missing and has come back. A returning file is news
    # for the same reason a brand new one is: it changes what the server can
    # offer, and (with auto-import on) it is importable again.
    new_entries = [
        entry
        for entry in entries
        if entry.source_path not in previous
        or previous[entry.source_path].get("missing_since")
    ]
    missing_rows = [
        row
        for path, row in previous.items()
        if not row.get("missing_since") and path not in present_paths
    ]

    now_iso = _iso(_now())
    _record_present(entries, at=now_iso)
    _record_missing([str(row["source_path"]) for row in missing_rows], at=now_iso)

    new_views = [entry.to_view() for entry in new_entries]
    new_candidates = [view for view in new_views if view["state"] == STATE_CANDIDATE]

    missing_views: list[dict[str, Any]] = []
    for row in missing_rows:
        agent_id, agent_name = _importing_agent(str(row["source_path"]))
        missing_views.append(
            {
                "source_path": row["source_path"],
                "name": row.get("name"),
                "location": row.get("location"),
                "state": row.get("state"),
                "missing_since": now_iso,
                "imported_agent_id": agent_id,
                "imported_agent_name": agent_name,
            }
        )

    imported: list[dict[str, Any]] = []
    import_failures: list[dict[str, Any]] = []
    if auto_import_enabled and new_candidates:
        imported, import_failures = _auto_import(new_candidates)

    notified_new = _notify_new_candidates(new_candidates, imported)
    notified_missing = _notify_missing_import_sources(missing_views)

    finished = _now()
    record = {
        "status": STATUS_OK,
        "trigger": trigger,
        "started_at": started_iso,
        "finished_at": _iso(finished),
        "duration_ms": int((finished - started).total_seconds() * 1000),
        "error": None,
        "counts": {
            "candidates": len(sweep.candidates),
            "excluded": len(sweep.excluded),
            "skipped": len(sweep.skipped),
            "new": len(new_views),
            "new_candidates": len(new_candidates),
            "missing": len(missing_views),
        },
        "new": new_views,
        "missing": missing_views,
        "auto_import": {
            "enabled": auto_import_enabled,
            "imported": imported,
            "failed": import_failures,
        },
        "notified": {"new": notified_new, "missing_imported": notified_missing},
    }
    _store_sweep_record(record)
    return record


def sweep_is_due(*, now: datetime | None = None) -> bool:
    """Whether enough time has passed since the last *attempt*.

    A server that has never swept is due immediately. Measuring from the
    attempt rather than the last success is what stops a persistently failing
    sweep from re-running on every 30-second scheduler tick.
    """
    record = get_last_sweep_record()
    if not record:
        return True
    started = _parse_iso(record.get("started_at"))
    if started is None:
        return True
    moment = now or _now()
    return (moment - started).total_seconds() >= sweep_interval_seconds()


def maybe_run_due_cli_agent_sweep(*, trigger: str = "scheduled") -> dict[str, Any] | None:
    """Run a sweep if one is due, else do nothing. Returns the record or ``None``.

    Called from the scheduler tick (:mod:`agent.scheduler`), which is the one
    background timer this server owns. The interval gate lives here, on a
    persisted timestamp, rather than in a second timer — so a server restarted
    hourly does not sweep hourly, and there is exactly one place that decides
    when a sweep happens.
    """
    if not sweep_is_due():
        return None
    return run_cli_agent_sweep(trigger=trigger)


__all__ = [
    "DEFAULT_SWEEP_INTERVAL_SECONDS",
    "LATEST_SWEEP_ID",
    "LEGACY_SETTING_AUTO_IMPORT_ENABLED",
    "SETTING_AUTO_IMPORT_ENABLED",
    "STATE_CANDIDATE",
    "STATE_EXCLUDED",
    "STATE_SKIPPED",
    "STATUS_FAILED",
    "STATUS_OK",
    "SWEEP_INTERVAL_ENV",
    "SeenEntry",
    "get_auto_import_enabled",
    "get_last_sweep_record",
    "get_stored_view",
    "load_seen",
    "maybe_run_due_cli_agent_sweep",
    "run_cli_agent_sweep",
    "set_auto_import_enabled",
    "sweep_interval_seconds",
    "sweep_is_due",
]
