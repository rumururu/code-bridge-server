"""The server has to notice agent definition files by itself, and say what changed.

Discovery used to happen only when a client asked, and was never written down.
So the server knew about a definition exactly while someone was looking at one,
and could never answer the two questions that matter when nobody is looking:
what appeared, and what went away.

The second one is the expensive one. An imported agent holds a *pointer* to
its ``.md`` file, not a copy, so deleting that file breaks the agent — and
because these agents run unattended, the user finds out from a 3am failure
naming a file they deleted last week. A sweep that spots it missing turns that
into a message they can act on in the morning.

What these tests defend:

- A sweep records what it found, and a later sweep can tell new from known.
- A file that disappeared is reported — and *notifies* only when something was
  imported from it. A missing file nothing points at is recorded and stays
  quiet, because a warning with no consequence is the kind people learn to
  ignore, and then they ignore the one that matters too.
- Nothing changed means nothing rings.
- Auto-import is off unless asked, imports only what can actually run alone,
  and does not create a second agent on the next sweep.
- A sweep that could not look is recorded as failed with its reason. Never as
  an empty success: an empty list reads as "you have no agents", which is a
  different and much worse statement than "I could not check".
- A notification failure never changes what the sweep did or recorded.
- The seen-set forgets a long-gone path nothing was imported from, and never
  forgets one an agent depends on. This is the residue rule, and it was
  learned the hard way: reinstalling the Claude plugin marketplace moved 31
  definition files to new paths and left 31 rows for files that will never
  exist again, every one of them a red "this agent lost its source file"
  warning for an agent nobody had created — and another 31 waiting on the
  next reinstall.
- A path that comes back has its missing mark cleared. A row left marked
  missing while its file sits on disk warns forever about an agent that is
  working fine.
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, notification_store, cli_agent_sources, cli_agent_sweep  # noqa: E402
from core import database  # noqa: E402

_ELIGIBLE = """\
---
name: {name}
description: {description}
model: sonnet
---

Do the thing.
"""


class CliAgentSweepTestCase(unittest.TestCase):
    """Isolated DB plus an isolated pretend `~/.claude/agents` directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.agents_dir = Path(self._tmp.name) / "agents"
        self.agents_dir.mkdir()

        self._original_db = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "sweep.db"
        agent_store._agent_store = None
        notification_store._notification_store = None
        database._settings_db = None
        database.init_db()
        self.addCleanup(self._restore)

        # Discovery walks the real machine otherwise, which would make every
        # assertion below depend on whatever this developer has installed.
        patcher = mock.patch.object(
            cli_agent_sources,
            "_all_source_locations",
            lambda: [("user", self.agents_dir)],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _restore(self) -> None:
        database.DB_PATH = self._original_db
        agent_store._agent_store = None
        notification_store._notification_store = None
        database._settings_db = None

    # --- helpers ---------------------------------------------------------

    def _write(self, filename: str, text: str) -> Path:
        path = self.agents_dir / filename
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return path

    def _write_eligible(self, filename: str, *, name: str, description: str = "Runs nightly.") -> Path:
        return self._write(filename, _ELIGIBLE.format(name=name, description=description))

    def _notifications(self, level: str | None = None) -> list[dict]:
        rows = notification_store.get_notification_store().list_notifications(limit=200)
        return [row for row in rows if level is None or row["level"] == level]


class SweepRecordsWhatItFoundTest(CliAgentSweepTestCase):
    def test_a_sweep_records_what_it_found(self):
        path = self._write_eligible("reviewer.md", name="reviewer")

        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(record["status"], cli_agent_sweep.STATUS_OK)
        self.assertEqual(record["counts"]["candidates"], 1)
        self.assertEqual(record["counts"]["new"], 1)

        seen = cli_agent_sweep.load_seen()
        self.assertIn(str(path.resolve()), seen)
        entry = seen[str(path.resolve())]
        self.assertEqual(entry["name"], "reviewer")
        self.assertEqual(entry["state"], cli_agent_sweep.STATE_CANDIDATE)
        self.assertIsNone(entry["missing_since"])

    def test_the_stored_view_says_when_it_ran_without_walking_the_disk(self):
        self._write_eligible("reviewer.md", name="reviewer")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        # If the view walked the filesystem, this would raise.
        with mock.patch.object(
            cli_agent_sweep,
            "discover_cli_agents",
            side_effect=AssertionError("the stored view must not sweep"),
        ):
            view = cli_agent_sweep.get_stored_view()

        self.assertIsNotNone(view["last_sweep"]["started_at"])
        self.assertEqual(view["last_sweep"]["status"], cli_agent_sweep.STATUS_OK)
        self.assertEqual([e["name"] for e in view["known"]["candidates"]], ["reviewer"])
        self.assertFalse(view["auto_import_enabled"])

    def test_a_sweep_is_not_due_again_immediately(self):
        self.assertTrue(cli_agent_sweep.sweep_is_due())
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        self.assertFalse(cli_agent_sweep.sweep_is_due())
        self.assertIsNone(cli_agent_sweep.maybe_run_due_cli_agent_sweep())


class NewAndMissingTest(CliAgentSweepTestCase):
    def test_a_second_sweep_reports_an_added_file_as_new(self):
        self._write_eligible("first.md", name="first")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        added = self._write_eligible("second.md", name="second")
        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual([entry["source_path"] for entry in record["new"]], [str(added.resolve())])
        self.assertEqual(record["counts"]["candidates"], 2)

    def test_a_removed_file_is_reported_as_missing(self):
        self._write_eligible("stays.md", name="stays")
        goes = self._write_eligible("goes.md", name="goes")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        goes.unlink()
        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(
            [entry["source_path"] for entry in record["missing"]], [str(goes.resolve())]
        )
        seen = cli_agent_sweep.load_seen()
        self.assertIsNotNone(seen[str(goes.resolve())]["missing_since"])
        # The one that is still there must not be collateral damage.
        self.assertIsNone(seen[str((self.agents_dir / "stays.md").resolve())]["missing_since"])

    def test_a_file_that_stays_missing_is_only_reported_missing_once(self):
        goes = self._write_eligible("goes.md", name="goes")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        goes.unlink()
        first = cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        second = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(len(first["missing"]), 1)
        # Still gone, still recorded as gone — but not news a second time, or
        # a deleted file would page the user every six hours forever.
        self.assertEqual(second["missing"], [])

    def test_a_file_that_comes_back_is_new_again(self):
        path = self._write_eligible("blinker.md", name="blinker")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        path.unlink()
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        self._write_eligible("blinker.md", name="blinker")

        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual([entry["name"] for entry in record["new"]], ["blinker"])
        self.assertIsNone(cli_agent_sweep.load_seen()[str(path.resolve())]["missing_since"])

    def test_a_path_that_is_found_again_has_its_missing_mark_cleared(self):
        """A file back on disk must stop reading as missing, everywhere.

        The mark is not just bookkeeping: it is what puts an entry in the
        stored view's `missing` bucket, which is what the dashboard paints a
        blocking warning from, and it is what the forgetting rule reads to
        decide what to delete. A row left marked while its file is right there
        warns forever about an agent that is working fine, and queues a live
        entry for deletion. `_record_present` clears it on *every* sweep that
        finds the file, not only the first one after it returns.
        """
        path = self._write_eligible("blinker.md", name="blinker")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        path.unlink()
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        self.assertIsNotNone(
            cli_agent_sweep.load_seen()[str(path.resolve())]["missing_since"]
        )

        self._write_eligible("blinker.md", name="blinker")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        # And again, on a sweep where nothing changed at all.
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertIsNone(
            cli_agent_sweep.load_seen()[str(path.resolve())]["missing_since"]
        )
        view = cli_agent_sweep.get_stored_view()
        self.assertEqual(view["known"]["missing"], [])
        self.assertEqual(
            [entry["source_path"] for entry in view["known"]["candidates"]],
            [str(path.resolve())],
        )


class ForgettingMissingEntriesTest(CliAgentSweepTestCase):
    """The seen-set must not grow a row per vanished file forever.

    Every path ever discovered gets a row, and a vanished one keeps its row
    with `missing_since` stamped. That is right for the case the sweep exists
    to catch — an imported agent whose file is gone will fail its next
    scheduled run, and the warning has to stand until someone acts on it — and
    wrong for everything else.

    The concrete failure: reinstalling the Claude plugin marketplace renames
    the old checkout and installs a fresh one, so 31 definition files move to
    31 new paths. Nothing was imported from any of them. Left alone the seen
    set keeps all 31 dead paths, the dashboard renders 31 red "lost its source
    file" warnings for agents that were never created, and the next reinstall
    adds 31 more. If this regresses, that is what comes back — with the real
    warning, the one about an agent that is actually breaking, buried
    somewhere in the middle of it.
    """

    def _sweep_at(self, moment: datetime) -> dict:
        with mock.patch.object(cli_agent_sweep, "_now", lambda: moment):
            return cli_agent_sweep.run_cli_agent_sweep(trigger="test")

    def _gone(self, filename: str = "reviewer.md", *, name: str = "reviewer") -> Path:
        """A file that was seen, then deleted, and is now recorded missing."""
        path = self._write_eligible(filename, name=name)
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        path.unlink()
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        self.assertIsNotNone(
            cli_agent_sweep.load_seen()[str(path.resolve())]["missing_since"]
        )
        return path

    def test_a_missing_file_an_agent_was_imported_from_is_never_forgotten(self):
        path = self._write_eligible("reviewer.md", name="reviewer")
        result = cli_agent_sources.import_cli_agent(str(path))
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        path.unlink()
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        # A year later, and with the retention set to nothing at all, on the
        # theory that whoever tunes this knob must not be able to switch the
        # warning off with it.
        with mock.patch.dict(
            "os.environ", {cli_agent_sweep.MISSING_RETENTION_ENV: "0"}
        ):
            record = self._sweep_at(datetime.now(UTC) + timedelta(days=365))

        self.assertEqual(record["forgotten"], [])
        seen = cli_agent_sweep.load_seen()
        self.assertIn(str(path.resolve()), seen)
        self.assertIsNotNone(seen[str(path.resolve())]["missing_since"])

        missing = cli_agent_sweep.get_stored_view()["known"]["missing"]
        self.assertEqual(
            [entry["imported_agent_id"] for entry in missing], [result.agent["id"]]
        )

    def test_a_missing_file_nothing_was_imported_from_is_forgotten_after_a_fortnight(self):
        path = self._gone()

        record = self._sweep_at(datetime.now(UTC) + timedelta(days=15))

        self.assertEqual(record["forgotten"], [str(path.resolve())])
        self.assertEqual(record["counts"]["forgotten"], 1)
        self.assertNotIn(str(path.resolve()), cli_agent_sweep.load_seen())
        self.assertEqual(cli_agent_sweep.get_stored_view()["known"]["missing"], [])

    def test_a_file_missing_for_less_than_that_is_still_on_record(self):
        """The grace period is the point: files come back.

        An unmounted drive, a fortnight away, a branch checked out over a
        weekend — forget too eagerly and the same file reads as brand new
        every time the machine changes shape.
        """
        path = self._gone()

        record = self._sweep_at(datetime.now(UTC) + timedelta(days=13))

        self.assertEqual(record["forgotten"], [])
        self.assertIn(str(path.resolve()), cli_agent_sweep.load_seen())

    def test_a_file_is_reported_missing_before_it_can_be_forgotten(self):
        """Even with no retention at all, the sweep that notices still reports.

        Forgetting runs last for this reason. A file that disappears and gets
        dropped in the same sweep must still appear in that sweep's diff —
        otherwise the one moment the disappearance was visible is the one
        moment it was silently discarded.
        """
        path = self._write_eligible("reviewer.md", name="reviewer")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        path.unlink()

        with mock.patch.dict(
            "os.environ", {cli_agent_sweep.MISSING_RETENTION_ENV: "0"}
        ):
            record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(
            [entry["source_path"] for entry in record["missing"]], [str(path.resolve())]
        )
        self.assertEqual(record["forgotten"], [str(path.resolve())])
        self.assertNotIn(str(path.resolve()), cli_agent_sweep.load_seen())

    def test_a_present_file_is_never_forgotten_however_old_the_row_is(self):
        self._write_eligible("stays.md", name="stays")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        record = self._sweep_at(datetime.now(UTC) + timedelta(days=400))

        self.assertEqual(record["forgotten"], [])
        self.assertEqual(len(cli_agent_sweep.load_seen()), 1)

    def test_an_unreadable_import_table_forgets_nothing(self):
        """Fail closed: without provenance, everything might be actionable."""
        path = self._gone()

        with mock.patch.object(
            cli_agent_sweep, "_import_map", side_effect=RuntimeError("db is down")
        ):
            record = self._sweep_at(datetime.now(UTC) + timedelta(days=99))

        self.assertEqual(record["forgotten"], [])
        self.assertIn(str(path.resolve()), cli_agent_sweep.load_seen())

    def test_a_failed_sweep_forgets_nothing(self):
        path = self._gone()

        with mock.patch.object(
            cli_agent_sweep,
            "discover_cli_agents",
            side_effect=OSError("the disk went away"),
        ):
            record = self._sweep_at(datetime.now(UTC) + timedelta(days=99))

        self.assertEqual(record["status"], cli_agent_sweep.STATUS_FAILED)
        self.assertEqual(record["forgotten"], [])
        self.assertIn(str(path.resolve()), cli_agent_sweep.load_seen())


class StoredViewProvenanceTest(CliAgentSweepTestCase):
    """The stored view has to say which entries anything depends on.

    The dashboard cannot work this out from `state`: that describes the file
    (importable, excluded, unparseable), not whether a Code Bridge agent was
    ever made from it. Guessing from `state == 'candidate'` is what turned a
    plugin marketplace reinstall into 31 blocking warnings about agents that
    do not exist.
    """

    def test_every_entry_says_whether_an_agent_was_imported_from_it(self):
        imported_path = self._write_eligible("reviewer.md", name="reviewer")
        untouched_path = self._write_eligible("drafter.md", name="drafter")
        result = cli_agent_sources.import_cli_agent(str(imported_path))

        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        view = cli_agent_sweep.get_stored_view()

        by_path = {
            entry["source_path"]: entry for entry in view["known"]["candidates"]
        }
        self.assertEqual(
            by_path[str(imported_path.resolve())]["imported_agent_id"],
            result.agent["id"],
        )
        self.assertEqual(
            by_path[str(imported_path.resolve())]["imported_agent_name"],
            result.agent["name"],
        )
        self.assertIsNone(by_path[str(untouched_path.resolve())]["imported_agent_id"])
        self.assertIsNone(by_path[str(untouched_path.resolve())]["imported_agent_name"])

    def test_a_missing_entry_carries_the_agent_that_will_break(self):
        path = self._write_eligible("reviewer.md", name="reviewer")
        result = cli_agent_sources.import_cli_agent(str(path))
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        path.unlink()
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        missing = cli_agent_sweep.get_stored_view()["known"]["missing"]

        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["imported_agent_id"], result.agent["id"])
        self.assertEqual(missing[0]["imported_agent_name"], result.agent["name"])

    def test_a_missing_entry_nothing_points_at_is_marked_as_such(self):
        """The 31-ghosts case: recorded, listed, and not actionable."""
        path = self._write_eligible("offered.md", name="offered")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        path.unlink()
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        missing = cli_agent_sweep.get_stored_view()["known"]["missing"]

        self.assertEqual(len(missing), 1)
        self.assertIsNone(missing[0]["imported_agent_id"])
        self.assertIsNone(missing[0]["imported_agent_name"])


class DashboardWarnsOnlyOnActionableEntriesTest(unittest.TestCase):
    """The card must warn about agents that will break, not files that left.

    Reading the shipped template because there is no other way to hold a page
    of inline JavaScript to a behaviour, and the behaviour is small and named:
    `cliAgentMissingAgents` is the one place the card and the page-wide banner
    decide what counts as a problem.

    It used to decide from `state === 'candidate'` — "this file could have
    produced an agent" — because the stored view carried no link to the agent
    imported from a path. A plugin marketplace reinstall then moved 31
    definition files and painted 31 red warnings for agents nobody had ever
    created. If this regresses, the real warning goes back to being one red
    box among dozens of meaningless ones.
    """

    @classmethod
    def setUpClass(cls):
        cls.template = (
            SERVER_DIR / "dashboard" / "templates" / "agents.html"
        ).read_text(encoding="utf-8")

    def test_the_warning_is_driven_by_the_import_mapping(self):
        self.assertIn(
            "return missing.filter((entry) => !!entry.imported_agent_id);",
            self.template,
        )
        self.assertNotIn("missing.filter((entry) => entry.state === 'candidate')", self.template)

    def test_the_rest_is_counted_quietly_rather_than_dropped(self):
        """The user can still see a file went away; it just is not an alarm."""
        self.assertIn("function cliAgentMissingQuietCount(sweep)", self.template)
        self.assertIn("return missing.filter((entry) => !entry.imported_agent_id).length;", self.template)
        self.assertIn("t('cli_sweep_missing_quiet')", self.template)

    def test_the_warning_names_the_agent_not_just_the_file(self):
        self.assertIn("entry.imported_agent_name || entry.name", self.template)

    def test_the_strings_exist_in_both_dashboard_locales(self):
        for key in ("cli_sweep_missing_quiet", "cli_sweep_missing_detail"):
            # Once under `en`, once under `ko`.
            self.assertEqual(
                self.template.count(f"{key}: '"), 2, f"{key} is not in both locales"
            )


class NotificationTest(CliAgentSweepTestCase):
    def test_a_new_definition_is_recorded_but_never_pushed(self):
        """Discovering a file is not worth interrupting anyone.

        This did notify, and it was wrong: a push landed on the user's phone
        saying a file they could register exists. That is only useful once
        they are already on the registration screen, which lists it anyway.
        It still has to be *recorded* — the screen and the stored view are
        built from that — so this asserts both halves: found, not pushed.
        """
        self._write_eligible("reviewer.md", name="reviewer")

        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual([entry["name"] for entry in record["new"]], ["reviewer"])
        self.assertFalse(record["notified"]["new"])
        titles = [row["title"] for row in self._notifications()]
        self.assertEqual(titles, [], f"a discovery must not notify: {titles}")

    def test_a_sweep_where_nothing_changed_does_not_notify(self):
        self._write_eligible("reviewer.md", name="reviewer")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        before = len(self._notifications())

        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertFalse(record["notified"]["new"])
        self.assertEqual(record["notified"]["missing_imported"], 0)
        self.assertEqual(len(self._notifications()), before)

    def test_a_missing_file_an_imported_agent_points_at_notifies(self):
        path = self._write_eligible("reviewer.md", name="reviewer")
        result = cli_agent_sources.import_cli_agent(str(path))
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        path.unlink()
        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(record["notified"]["missing_imported"], 1)
        errors = self._notifications(level="error")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["agent_id"], result.agent["id"])
        self.assertIn(str(path.resolve()), errors[0]["body"])

    def test_a_missing_file_nothing_points_at_does_not_notify(self):
        path = self._write_eligible("reviewer.md", name="reviewer")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        path.unlink()
        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        # Recorded, so the user can see it went away...
        self.assertEqual(len(record["missing"]), 1)
        self.assertIsNone(record["missing"][0]["imported_agent_id"])
        # ...but nothing is going to break, so nothing rings.
        self.assertEqual(record["notified"]["missing_imported"], 0)
        self.assertEqual(self._notifications(level="error"), [])

    def test_a_notification_failure_does_not_affect_the_sweep(self):
        self._write_eligible("reviewer.md", name="reviewer")

        with mock.patch.object(
            notification_store,
            "get_notification_store",
            side_effect=RuntimeError("notification store is down"),
        ):
            record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(record["status"], cli_agent_sweep.STATUS_OK)
        self.assertFalse(record["notified"]["new"])
        # The sweep result is what matters and it is intact and stored.
        self.assertEqual(record["counts"]["candidates"], 1)
        self.assertEqual(len(cli_agent_sweep.load_seen()), 1)
        self.assertEqual(
            cli_agent_sweep.get_last_sweep_record()["status"], cli_agent_sweep.STATUS_OK
        )


class AutoImportTest(CliAgentSweepTestCase):
    def _write_a_mixed_directory(self) -> Path:
        """One standalone definition, one dispatch-only worker, one broken file."""
        eligible = self._write_eligible("orchestrator.md", name="orchestrator")
        self._write(
            "worker.md",
            """\
            ---
            name: worker
            description: Does one piece of the job.
            ---

            Worker body.
            """,
        )
        # The orchestrator declares it dispatches `worker`, which is what makes
        # `worker` ineligible: scheduling it alone would refuse every night.
        self._write(
            "orchestrator.md",
            """\
            ---
            name: orchestrator
            description: Reviews the repository end to end.
            tools: Bash, Agent(worker)
            ---

            Orchestrator body.
            """,
        )
        self._write("broken.md", "Not an agent definition at all.\n")
        return eligible

    def test_auto_import_is_off_by_default_and_creates_nothing(self):
        self._write_a_mixed_directory()

        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertFalse(record["auto_import"]["enabled"])
        self.assertEqual(record["auto_import"]["imported"], [])
        self.assertEqual(agent_store.get_agent_store().list_agents(), [])

    def test_auto_import_on_imports_only_eligible_candidates(self):
        self._write_a_mixed_directory()
        cli_agent_sweep.set_auto_import_enabled(True)

        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertTrue(record["auto_import"]["enabled"])
        imported_names = sorted(entry["name"] for entry in record["auto_import"]["imported"])
        self.assertEqual(imported_names, ["orchestrator"])
        self.assertEqual(record["auto_import"]["failed"], [])
        # The dispatch-only worker and the unparseable file are recorded, and
        # neither became an agent.
        self.assertEqual(
            sorted(agent["name"] for agent in agent_store.get_agent_store().list_agents()),
            ["orchestrator"],
        )
        self.assertEqual(record["counts"]["excluded"], 1)
        self.assertEqual(record["counts"]["skipped"], 1)

    def test_auto_import_is_idempotent_across_sweeps(self):
        self._write_a_mixed_directory()
        cli_agent_sweep.set_auto_import_enabled(True)
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        second = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(second["auto_import"]["imported"], [])
        self.assertEqual(len(agent_store.get_agent_store().list_agents()), 1)

    def test_turning_auto_import_on_is_not_retroactive(self):
        """The default being off is only meaningful if switching it on later
        does not then import the backlog in one go."""
        self._write_a_mixed_directory()
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        cli_agent_sweep.set_auto_import_enabled(True)
        record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(record["auto_import"]["imported"], [])
        self.assertEqual(agent_store.get_agent_store().list_agents(), [])

        # ...but a definition written after the switch is imported.
        self._write_eligible("fresh.md", name="fresh")
        after = cli_agent_sweep.run_cli_agent_sweep(trigger="test")
        self.assertEqual(
            [entry["name"] for entry in after["auto_import"]["imported"]], ["fresh"]
        )


class FailedSweepTest(CliAgentSweepTestCase):
    def test_a_failed_sweep_is_recorded_as_failed_not_as_empty(self):
        self._write_eligible("reviewer.md", name="reviewer")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        with mock.patch.object(
            cli_agent_sweep,
            "discover_cli_agents",
            side_effect=OSError("the disk went away"),
        ):
            record = cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(record["status"], cli_agent_sweep.STATUS_FAILED)
        self.assertEqual(record["error"]["type"], "OSError")
        self.assertIn("the disk went away", record["error"]["message"])
        # Not zeros. A count of zero would claim the sweep looked and found
        # nothing, which is the lie this guards against.
        self.assertIsNone(record["counts"])
        self.assertEqual(cli_agent_sweep.get_last_sweep_record()["status"], "failed")

    def test_a_failed_sweep_leaves_the_known_set_alone(self):
        path = self._write_eligible("reviewer.md", name="reviewer")
        cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        with mock.patch.object(
            cli_agent_sweep,
            "discover_cli_agents",
            side_effect=OSError("the disk went away"),
        ):
            cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        seen = cli_agent_sweep.load_seen()
        self.assertIn(str(path.resolve()), seen)
        # Crucially not stamped missing: the file is fine, the walk was not.
        self.assertIsNone(seen[str(path.resolve())]["missing_since"])

        view = cli_agent_sweep.get_stored_view()
        self.assertEqual(view["last_sweep"]["status"], "failed")
        self.assertEqual([e["name"] for e in view["known"]["candidates"]], ["reviewer"])

    def test_a_failed_sweep_does_not_notify(self):
        with mock.patch.object(
            cli_agent_sweep,
            "discover_cli_agents",
            side_effect=OSError("the disk went away"),
        ):
            cli_agent_sweep.run_cli_agent_sweep(trigger="test")

        self.assertEqual(self._notifications(), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
