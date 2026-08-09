"""An imported CLI agent runs the file, not a copy of it — or fails saying so.

Importing an agent definition used to copy its prompt into a Code Bridge
agent. Three things went wrong with the copy, and these tests hold each one
shut:

The copy went stale. Edit the source file and the scheduled run kept
executing last month's text, with nothing anywhere reporting the divergence.

The declared tools were decoration. `tools: Read, Glob` was stored and never
passed to anything, so a Claude agent its author wrote as read-only ran with
whatever the CLI allows — writing files at 3am on a schedule nobody watches.

And most installed Claude plugin agents are not standalone at all. They are
workers a parent dispatches with a payload and refuse anything else, so
scheduling one buys a refusal every night. Offering them is a trap; hiding
them without saying why is a different trap, so an excluded definition must
still be reportable with its reason.

Two failure modes must stay loud. A source that is gone or unparseable fails
the run naming the file — never falls back to the stored stub. And a provider
that cannot carry an agent definition is refused, because dropping the
definition and running the prompt as plain text is exactly how a read-only
agent gets write access without anyone deciding that.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store, cli_agent_sources  # noqa: E402
from agent.cli_agent_eligibility import (  # noqa: E402
    EXCLUDED_DIRECT_INVOCATION_DISCLAIMED,
    EXCLUDED_DISPATCH_TARGET,
    classify_candidates,
)
from agent.cli_agent_runtime import (  # noqa: E402
    CliAgentSourceUnavailableError,
    resolve_cli_agent_definition,
)
from agent.task_orchestrator import (  # noqa: E402
    execute_task_orchestration,
    prepare_task_orchestration,
)
from core import database  # noqa: E402
from llm.claude_session import ClaudeSession, SessionManager  # noqa: E402
from llm.llm_session import (  # noqa: E402
    LlmSessionFactory,
    CliAgentDefinition,
    CliAgentUnsupportedError,
)

CLI_AGENT_FILE = textwrap.dedent(
    """\
    ---
    name: disk-watch
    description: Reports on disk pressure.
    tools: Read, Glob
    model: sonnet
    ---

    You are the ORIGINAL disk watcher. Report free space and stop.
    """
)


def _candidate(**over):
    base = dict(
        source_path="/tmp/a.md",
        candidate_id="c-a",
        location="plugin:market/demo",
        name="a",
        description="Does a thing.",
        tools=[],
        model=None,
        effort=None,
        color=None,
        body="Body.",
    )
    base.update(over)
    return cli_agent_sources.CliAgentCandidate(**base)


class StandaloneEligibilityTest(unittest.TestCase):
    """Only definitions that can run alone are offered — and the rest say why."""

    def test_an_author_who_disclaims_direct_invocation_is_taken_at_their_word(self):
        """`description` is the field Claude Code reads to decide invocation.

        Shipped agents write the answer into it: "dispatched by the fix job,
        not for direct invocation". Scheduling one refuses every night.
        """
        worker = _candidate(
            candidate_id="c-worker",
            name="patch-generator",
            description=(
                "Implements the fix for one finding inside a scratch workspace "
                "clone; dispatched by the fix job, not for direct invocation."
            ),
        )

        exclusions = classify_candidates([worker])

        self.assertIn("c-worker", exclusions)
        self.assertEqual(
            exclusions["c-worker"].reason, EXCLUDED_DIRECT_INVOCATION_DISCLAIMED
        )
        self.assertIn("/tmp/a.md", exclusions["c-worker"].detail)

    def test_an_agent_another_agent_declares_it_dispatches_is_excluded(self):
        """`tools: Agent(plugin:child)` is a dispatch edge, not prose.

        The detail names the parent, so "why is my agent missing" has an
        answer a human can go and check.
        """
        parent = _candidate(
            candidate_id="c-parent",
            source_path="/tmp/orchestrator.md",
            name="claude-security",
            description="The orchestrator. Best as the main agent of a session.",
            tools=["Read", "Bash", "Agent(demo:explore, demo:verifier)"],
        )
        child = _candidate(
            candidate_id="c-child",
            source_path="/tmp/explore.md",
            name="explore",
            description="Read-only code explorer.",
        )

        exclusions = classify_candidates([parent, child])

        self.assertNotIn("c-parent", exclusions)
        self.assertEqual(exclusions["c-child"].reason, EXCLUDED_DISPATCH_TARGET)
        self.assertIn("claude-security", exclusions["c-child"].detail)
        self.assertIn("/tmp/orchestrator.md", exclusions["c-child"].detail)

    def test_an_ordinary_reviewer_stays_on_the_list(self):
        """The filter must not eat the agents the feature exists to schedule.

        A body full of "WORKSPACE"/"refusal"/"must carry" language is normal
        for a reviewer; matching on it excluded seven user-invoked agents on
        the machine this was built against.
        """
        reviewer = _candidate(
            candidate_id="c-review",
            name="code-reviewer",
            description=(
                "Use this agent when you need to review code for adherence to "
                "project guidelines. Use proactively before creating pull requests."
            ),
            body=(
                "Report a FINDING for each issue. If the WORKSPACE is dirty say so. "
                "Set refusal when you must carry on without a diff."
            ),
        )

        self.assertEqual(classify_candidates([reviewer]), {})

    def test_agent_tool_arguments_are_one_declaration_not_three_tools(self):
        """`Agent(a, b)` split on every comma produced unresolvable fragments.

        Those fragments are also what would be handed back to the CLI as the
        enforced tool list.
        """
        self.assertEqual(
            cli_agent_sources._parse_tools("Read, Bash, Agent(x:one, x:two)"),
            ["Read", "Bash", "Agent(x:one, x:two)"],
        )


class CliAgentImportFilteringTest(unittest.TestCase):
    """Discovery offers the standalone ones and reports the rest."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "worker.md").write_text(
            "---\nname: worker\ndescription: Dispatched by the scan workflow; "
            "not for direct invocation.\n---\n\nWorker body.\n",
            encoding="utf-8",
        )
        (self.root / "watch.md").write_text(CLI_AGENT_FILE, encoding="utf-8")
        self._locations = mock.patch.object(
            cli_agent_sources,
            "_all_source_locations",
            return_value=[("user", self.root)],
        )
        self._locations.start()

    def tearDown(self):
        self._locations.stop()
        self._tmp.cleanup()

    def test_a_dispatch_dependent_definition_is_excluded_but_still_reported(self):
        sweep = cli_agent_sources.discover_cli_agents()

        self.assertEqual([c.name for c in sweep.candidates], ["disk-watch"])
        self.assertEqual([e.candidate.name for e in sweep.excluded], ["worker"])
        view = sweep.excluded[0].to_view()
        self.assertEqual(
            view["excluded_reason"], EXCLUDED_DIRECT_INVOCATION_DISCLAIMED
        )
        self.assertTrue(view["excluded_detail"])
        self.assertEqual(view["name"], "worker")

    def test_importing_an_excluded_definition_is_refused_with_the_reason(self):
        """"Not discoverable" would be a lie — the file parsed fine."""
        with mock.patch.object(
            cli_agent_sources, "_find_import_by_source_path", return_value=None
        ):
            with self.assertRaises(cli_agent_sources.CliAgentImportError) as caught:
                cli_agent_sources.import_cli_agent(str(self.root / "worker.md"))

        message = str(caught.exception)
        self.assertIn("not standalone-runnable", message)
        self.assertIn(EXCLUDED_DIRECT_INVOCATION_DISCLAIMED, message)


class ClaudeSessionCliAgentOptionsTest(unittest.TestCase):
    """The definition has to actually reach the SDK, or tools stay decoration."""

    @staticmethod
    def _definition(**over):
        base = dict(
            name="disk-watch",
            description="Reports on disk pressure.",
            prompt="You are the disk watcher.",
            source_path="/tmp/watch.md",
            tools=("Read", "Glob"),
            model="sonnet",
            effort="medium",
        )
        base.update(over)
        return CliAgentDefinition(**base)

    def test_the_definition_and_the_selection_both_reach_the_options(self):
        """Both halves are load-bearing.

        The definition is what makes `tools` enforced; `--agent` is what
        selects it, and is also the only way a plugin agent runs at all.
        """
        session = ClaudeSession(project_path="/tmp", cli_agent=self._definition())

        options = session._build_options()

        self.assertEqual(list(options.agents), ["disk-watch"])
        definition = options.agents["disk-watch"]
        self.assertEqual(definition.prompt, "You are the disk watcher.")
        self.assertEqual(definition.tools, ["Read", "Glob"])
        self.assertEqual(definition.model, "sonnet")
        self.assertEqual(definition.effort, "medium")
        self.assertEqual(options.extra_args.get("agent"), "disk-watch")

    def test_an_unknown_effort_is_dropped_rather_than_failing_the_run(self):
        session = ClaudeSession(
            project_path="/tmp", cli_agent=self._definition(effort="ludicrous")
        )

        options = session._build_options()

        self.assertIsNone(options.agents["disk-watch"].effort)

    def test_an_ordinary_session_attaches_nothing(self):
        options = ClaudeSession(project_path="/tmp")._build_options()

        self.assertIsNone(options.agents)
        self.assertNotIn("agent", options.extra_args or {})

    def test_a_cached_session_is_not_reused_across_different_definitions(self):
        """The cache key is the task, so two steps of one task share a session.

        Reuse would apply one agent's tool restrictions to the other step — or
        let a restricted step escape them.
        """
        created: list[CliAgentDefinition | None] = []

        class FakeSession:
            provider_id = "anthropic"

            def __init__(self, cli_agent):
                self.project_path = "/tmp"
                self.cli_agent = cli_agent
                self.closed = False

            async def close(self):
                self.closed = True

            async def set_model(self, model):
                return None

        def fake_create(*, provider_id, project_path, model, cli_agent):
            created.append(cli_agent)
            return FakeSession(cli_agent)

        manager = SessionManager()
        first = self._definition()
        second = self._definition(prompt="You are the EDITED disk watcher.")

        with mock.patch.object(LlmSessionFactory, "create_session", fake_create):
            session_a = asyncio.run(
                manager.get_or_create_session("task:1", "/tmp", cli_agent=first)
            )
            session_b = asyncio.run(
                manager.get_or_create_session("task:1", "/tmp", cli_agent=first)
            )
            session_c = asyncio.run(
                manager.get_or_create_session("task:1", "/tmp", cli_agent=second)
            )

        self.assertIs(session_a, session_b)
        self.assertIsNot(session_b, session_c)
        self.assertTrue(session_a.closed)
        self.assertEqual(created, [first, second])

    def test_a_provider_that_cannot_carry_a_definition_refuses(self):
        """Silently running the prompt as plain text unrestricts the agent."""
        with self.assertRaises(CliAgentUnsupportedError) as caught:
            LlmSessionFactory.create_session(
                provider_id="openai",
                project_path="/tmp",
                cli_agent=self._definition(),
            )

        message = str(caught.exception)
        self.assertIn("openai", message)
        self.assertIn("/tmp/watch.md", message)
        self.assertIn("Nothing was run", message)

    def test_antigravity_is_refused_too_despite_having_definition_files(self):
        """It has agent files and an --agent flag; neither makes it able to run one.

        Verified: `agy --agent <name>` does not apply the persona, and an
        unknown name is accepted silently. Accepting it here would mean pasting
        the prompt into a plain session and calling that "running your agent".
        """
        with self.assertRaises(CliAgentUnsupportedError) as caught:
            LlmSessionFactory.create_session(
                provider_id="antigravity",
                project_path="/tmp",
                cli_agent=self._definition(),
            )

        self.assertIn("antigravity", str(caught.exception))
        self.assertIn("Nothing was run", str(caught.exception))


class CliAgentBackedRunTest(unittest.TestCase):
    """End to end: what a run hands the session is the file, read just now."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = (self.root / "watch.md").resolve()
        self.source.write_text(CLI_AGENT_FILE, encoding="utf-8")

        self._db = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._db.name) / "cli_agent_runtime.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

        self._locations = mock.patch.object(
            cli_agent_sources,
            "_all_source_locations",
            return_value=[("user", self.root)],
        )
        self._locations.start()
        self.agent = cli_agent_sources.import_cli_agent(str(self.source)).agent

    def tearDown(self):
        self._locations.stop()
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self._db.cleanup()
        self._tmp.cleanup()

    def _run(self, provider_id: str = "anthropic"):
        """Execute one run, returning (definition handed to the session, messages)."""
        task = self.store.create_task(
            title="Check the disk",
            assigned_agent_id=self.agent["id"],
            goal="Report free space.",
        )
        prepared = prepare_task_orchestration(
            task["id"], provider_id=provider_id, auto_start=False
        )
        assert prepared is not None

        handed: list[object] = []
        messages: list[str] = []

        async def fake_create_chat_session(**kwargs):
            handed.append(kwargs.get("cli_agent"))
            return object()

        async def fake_stream(_sink, _session, **kwargs):
            messages.append(str(kwargs.get("user_message") or ""))
            return True

        with mock.patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), mock.patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            asyncio.run(execute_task_orchestration(prepared["execution"]))

        steps = self.store.list_task_steps(task["id"])
        self._failures = [
            str(((event.get("app_event") or {}).get("error") or {}).get("message") or "")
            for event in self.store.list_events(prepared["run"]["id"])
            if event.get("event_type") == "task.step.failed"
        ]
        return handed, messages, steps

    def _only_failure(self) -> str:
        """The message the run recorded for the step that did not complete.

        Read off the ``task.step.failed`` event rather than the step row: a
        step that lands on a checkpoint has its ``output`` replaced by the
        checkpoint, and the event is the durable record the app timeline
        shows.
        """
        self.assertEqual(len(self._failures), 1, self._failures)
        return self._failures[0]

    def test_a_run_hands_the_session_the_files_prompt_and_declared_tools(self):
        handed, messages, steps = self._run()

        self.assertEqual(len(handed), 1)
        definition = handed[0]
        assert definition is not None
        self.assertEqual(definition.name, "disk-watch")
        self.assertIn("ORIGINAL disk watcher", definition.prompt)
        self.assertEqual(definition.tools, ("Read", "Glob"))
        self.assertEqual(definition.source_path, str(self.source))
        self.assertEqual(steps[0]["status"], "completed")
        # The stale copy must not be replayed as a second set of instructions.
        self.assertNotIn("ORIGINAL disk watcher", messages[0])
        self.assertIn(str(self.source), messages[0])

    def test_editing_the_source_changes_what_the_next_run_gets(self):
        """No re-import, no cache: the file is the program."""
        first_handed, _messages, _steps = self._run()

        self.source.write_text(
            CLI_AGENT_FILE.replace("ORIGINAL", "EDITED").replace(
                "tools: Read, Glob", "tools: Read"
            ),
            encoding="utf-8",
        )
        second_handed, _messages2, _steps2 = self._run()

        self.assertIn("ORIGINAL disk watcher", first_handed[0].prompt)
        self.assertIn("EDITED disk watcher", second_handed[0].prompt)
        self.assertEqual(second_handed[0].tools, ("Read",))
        # And the record on disk still says nothing about either version.
        stored = self.store.get_agent(self.agent["id"]) or {}
        self.assertNotIn("ORIGINAL", stored.get("system_prompt") or "")
        self.assertNotIn("EDITED", stored.get("system_prompt") or "")

    def test_a_deleted_source_fails_the_run_naming_the_file(self):
        self.source.unlink()

        handed, _messages, steps = self._run()

        self.assertEqual(handed, [])  # no session was ever opened
        self.assertNotEqual(steps[0]["status"], "completed")
        message = self._only_failure()
        self.assertIn(str(self.source), message)
        self.assertIn("Nothing was run", message)

    def test_an_unparseable_source_fails_the_run_naming_the_file(self):
        self.source.write_text("no frontmatter here at all\n", encoding="utf-8")

        handed, _messages, steps = self._run()

        self.assertEqual(handed, [])
        self.assertNotEqual(steps[0]["status"], "completed")
        message = self._only_failure()
        self.assertIn(str(self.source), message)
        self.assertIn(cli_agent_sources.SKIP_NO_FRONTMATTER, message)

    def test_a_non_claude_provider_is_reported_not_silently_downgraded(self):
        """The real session factory runs here — it is the thing being tested.

        Nothing is spawned: the refusal happens before any provider session is
        constructed. A downgrade would show up as a completed step, which is
        precisely the outcome that must not be reachable.
        """
        task = self.store.create_task(
            title="Check the disk",
            assigned_agent_id=self.agent["id"],
            goal="Report free space.",
        )
        prepared = prepare_task_orchestration(
            task["id"], provider_id="openai", auto_start=False
        )
        assert prepared is not None

        async def fake_stream(_sink, _session, **_kwargs):
            raise AssertionError("a turn must not start on an unsupported provider")

        with mock.patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            asyncio.run(execute_task_orchestration(prepared["execution"]))

        steps = self.store.list_task_steps(task["id"])
        self.assertNotEqual(steps[0]["status"], "completed")
        failures = [
            str(((event.get("app_event") or {}).get("error") or {}).get("message") or "")
            for event in self.store.list_events(prepared["run"]["id"])
            if event.get("event_type") == "task.step.failed"
        ]
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("openai", failures[0])
        self.assertIn("cannot carry an agent definition", failures[0])
        self.assertIn(str(self.source), failures[0])

    def test_an_ordinary_agent_is_unaffected(self):
        """resolve_cli_agent_definition returning None is the common case."""
        plain = self.store.create_agent(name="plain", flow_json=[])

        self.assertIsNone(resolve_cli_agent_definition(plain["id"]))
        self.assertIsNone(resolve_cli_agent_definition(None))

    def test_the_resolver_raises_rather_than_returning_the_stored_stub(self):
        self.source.unlink()

        with self.assertRaises(CliAgentSourceUnavailableError) as caught:
            resolve_cli_agent_definition(self.agent["id"])

        self.assertIn(str(self.source), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
