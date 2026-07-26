"""Shell steps: run a registered script, and make its failure diagnosable.

Two things matter here. First, a step may only name a *registered* script —
a workflow that could carry a command line would be remote code execution
behind whatever standing policy rule lets it run unattended. Second, when a
script fails, the exit code and the tail of both streams have to survive into
the next step's evidence, because the next step is usually an LLM asked to
work out what changed (the UI moved, the script needs editing). "It failed" is
useless for that; the last 40 lines of stderr are not.
"""

import asyncio
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import script_store as script_store_module
from agent.script_store import ScriptRegistrationError
from agent.shell_step_executor import build_command, run_registered_script
from agent.task_orchestrator import _workflow_step_output_summary
from agent.workflow_v2 import WorkflowNormalizationError, normalize_workflow
from core import database


def _write_script(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class ScriptRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "scripts_test.db"
        script_store_module._script_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = script_store_module.get_script_store()

    def _restore(self) -> None:
        script_store_module._script_store = None
        database.DB_PATH = self._original_db_path

    def test_registers_an_existing_script(self):
        path = _write_script(self.dir, "cycle.sh", "#!/bin/bash\necho hi\n")
        script = self.store.register(name="cycle", path=str(path))
        self.assertEqual(script["path"], str(path))
        self.assertEqual(script["interpreter"], "bash")
        self.assertEqual(script["default_args"], [])

    def test_missing_script_is_rejected_at_registration(self):
        # A typo must fail now, not at 3am inside a scheduled run.
        with self.assertRaises(ScriptRegistrationError):
            self.store.register(name="ghost", path=str(self.dir / "nope.sh"))

    def test_relative_path_is_rejected(self):
        with self.assertRaises(ScriptRegistrationError):
            self.store.register(name="rel", path="scripts/cycle.sh")

    def test_directory_is_rejected(self):
        with self.assertRaises(ScriptRegistrationError):
            self.store.register(name="dir", path=str(self.dir))

    def test_unknown_interpreter_is_rejected(self):
        path = _write_script(self.dir, "x.sh", "#!/bin/bash\n")
        with self.assertRaises(ScriptRegistrationError):
            self.store.register(name="x", path=str(path), interpreter="rm")

    def test_same_path_cannot_be_registered_twice(self):
        path = _write_script(self.dir, "once.sh", "#!/bin/bash\n")
        self.store.register(name="once", path=str(path))
        with self.assertRaises(ScriptRegistrationError):
            self.store.register(name="again", path=str(path))

    def test_timeout_is_capped(self):
        path = _write_script(self.dir, "slow.sh", "#!/bin/bash\n")
        script = self.store.register(
            name="slow", path=str(path), timeout_seconds=99_999_999
        )
        self.assertLessEqual(script["timeout_seconds"], 6 * 3600)

    def test_delete(self):
        path = _write_script(self.dir, "gone.sh", "#!/bin/bash\n")
        script = self.store.register(name="gone", path=str(path))
        self.assertTrue(self.store.delete(script["id"]))
        self.assertIsNone(self.store.get(script["id"]))


class ShellStepNormalizationTest(unittest.TestCase):
    def test_shell_step_requires_a_script_id(self):
        with self.assertRaises(WorkflowNormalizationError):
            normalize_workflow([{"type": "shell", "name": "run"}])

    def test_shell_step_keeps_script_fields(self):
        step = normalize_workflow(
            [{"type": "shell", "name": "run", "script_id": "script_1", "script_args": ["a", 2]}]
        )[0]
        self.assertEqual(step["script_id"], "script_1")
        self.assertEqual(step["script_args"], ["a", "2"])

    def test_shell_step_can_escalate_on_failure(self):
        # The pattern the step type exists for: script fails -> LLM diagnoses.
        steps = normalize_workflow(
            [
                {
                    "type": "shell",
                    "id": "run_cycle",
                    "script_id": "script_1",
                    "on_failure": "goto_step:diagnose",
                },
                {"type": "llm", "id": "diagnose", "name": "Diagnose"},
            ]
        )
        self.assertEqual(steps[0]["on_failure"]["type"], "goto_step")
        self.assertEqual(steps[0]["on_failure"]["target_step_id"], "diagnose")


class RunScriptTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _script(self, body: str, **overrides):
        path = _write_script(self.dir, overrides.pop("name", "s.sh"), body)
        script = {
            "path": str(path),
            "interpreter": "bash",
            "default_args": [],
            "timeout_seconds": 30,
        }
        script.update(overrides)
        return script

    async def test_successful_script(self):
        result = await run_registered_script(self._script("#!/bin/bash\necho done\n"))
        self.assertTrue(result.completed)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("done", result.stdout)

    async def test_failing_script_keeps_exit_code_and_stderr(self):
        result = await run_registered_script(
            self._script("#!/bin/bash\necho 'login form not found' >&2\nexit 3\n")
        )
        self.assertFalse(result.completed)
        self.assertEqual(result.exit_code, 3)
        self.assertIn("login form not found", result.stderr)

    async def test_arguments_are_passed_through(self):
        script = self._script('#!/bin/bash\necho "$1|$2"\n', default_args=["first"])
        result = await run_registered_script(script, extra_args=["second"])
        self.assertIn("first|second", result.stdout)

    async def test_timeout_kills_the_script(self):
        script = self._script("#!/bin/bash\nsleep 30\n", timeout_seconds=1)
        result = await run_registered_script(script)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.completed)
        self.assertIn("timeout", (result.error or "").lower())

    async def test_missing_binary_is_reported_not_raised(self):
        script = self._script("#!/bin/bash\n")
        script["interpreter"] = "direct"
        script["path"] = str(self.dir / "not-there")
        result = await run_registered_script(script)
        self.assertFalse(result.completed)
        self.assertIsNotNone(result.error)

    async def test_long_output_is_truncated_from_the_front(self):
        # Keep the tail: that is where the failure is.
        script = self._script(
            "#!/bin/bash\nfor i in $(seq 1 5000); do echo \"line $i\"; done\n"
        )
        result = await run_registered_script(script)
        self.assertIn("line 5000", result.stdout)
        self.assertIn("truncated", result.stdout)

    def test_build_command_direct_interpreter(self):
        self.assertEqual(
            build_command({"interpreter": "direct", "path": "/bin/echo", "default_args": ["x"]}),
            ["/bin/echo", "x"],
        )

    def test_environment_is_inherited(self):
        self.assertIn("PATH", os.environ)


class ShellEvidenceTest(unittest.TestCase):
    def test_failure_evidence_carries_what_a_diagnosis_needs(self):
        summary = _workflow_step_output_summary(
            {
                "shell": {
                    "status": "failed",
                    "exit_code": 3,
                    "command": ["bash", "/tmp/cycle.sh", "DEVICE"],
                    "stdout": "step 1 ok",
                    "stderr": "login form not found",
                    "timed_out": False,
                }
            }
        )
        joined = "\n".join(summary)
        self.assertIn("shell.exit_code: 3", joined)
        self.assertIn("login form not found", joined)
        self.assertIn("cycle.sh", joined)

    def test_timeout_is_called_out(self):
        summary = _workflow_step_output_summary(
            {"shell": {"status": "failed", "exit_code": None, "timed_out": True}}
        )
        self.assertIn("shell.timed_out: true", "\n".join(summary))


if __name__ == "__main__":
    unittest.main()


class EscalationEvidenceTest(unittest.TestCase):
    """The step an escalation exists to explain must reach the LLM.

    Evidence used to be gathered from completed steps only, so the failed step
    that triggered ``on_failure: goto_step`` was the one thing excluded. The
    diagnosis step then had nothing to work from and — correctly — refused to
    invent a cause, which looks like the feature not working at all.
    """

    def test_failed_step_is_included_in_previous_evidence(self):
        from agent.task_orchestrator import _previous_completed_workflow_steps

        steps = [
            {
                "id": "s1",
                "sequence": 1,
                "status": "failed",
                "title": "Run login cycle",
                "input": {"workflow_step_id": "run_cycle"},
                "output": {"shell": {"status": "failed", "exit_code": 4}},
            },
            {"id": "s2", "sequence": 2, "status": "running", "title": "Diagnose"},
        ]
        previous = _previous_completed_workflow_steps(steps, current_step=steps[1])
        self.assertEqual([step["id"] for step in previous], ["s1"])

    def test_running_and_queued_steps_are_still_excluded(self):
        from agent.task_orchestrator import _previous_completed_workflow_steps

        steps = [
            {"id": "s1", "sequence": 1, "status": "queued", "title": "A"},
            {"id": "s2", "sequence": 2, "status": "running", "title": "B"},
            {"id": "s3", "sequence": 3, "status": "waiting_for_user", "title": "C"},
            {"id": "s4", "sequence": 4, "status": "running", "title": "D"},
        ]
        previous = _previous_completed_workflow_steps(steps, current_step=steps[3])
        self.assertEqual(previous, [])

    def test_failed_shell_output_reaches_the_prompt_text(self):
        from agent.task_orchestrator import _workflow_previous_evidence

        evidence = _workflow_previous_evidence(
            [
                {
                    "id": "s1",
                    "title": "Run login cycle",
                    "input": {"workflow_step_id": "run_cycle"},
                    "output": {
                        "shell": {
                            "status": "failed",
                            "exit_code": 4,
                            "stderr": "ERROR: login button not found — layout changed",
                        }
                    },
                }
            ]
        )
        self.assertIn("exit_code: 4", evidence)
        self.assertIn("login button not found", evidence)


class ScriptDraftingTest(unittest.TestCase):
    """Drafting writes nothing and registers nothing until a human says so.

    "Write me a script" must not be equivalent to "run whatever you just
    wrote": the draft is text the caller reviews and can edit first.
    """

    def setUp(self) -> None:
        from routes import scripts as scripts_routes

        self.routes = scripts_routes

    def test_code_fences_are_stripped(self):
        # Models add ```bash fences even when told not to.
        self.assertEqual(
            self.routes._strip_code_fences("```bash\n#!/bin/bash\necho hi\n```"),
            "#!/bin/bash\necho hi",
        )

    def test_unfenced_text_is_untouched(self):
        self.assertEqual(
            self.routes._strip_code_fences("#!/bin/bash\necho hi"), "#!/bin/bash\necho hi"
        )

    def test_slug_keeps_non_ascii_names_distinct(self):
        # Stripping non-ASCII collapsed every Korean name onto the same slug.
        first = self.routes._slugify("스몰뎁 사이클")
        second = self.routes._slugify("나무트리 사이클")
        self.assertNotEqual(first, second)
        self.assertIn("스몰뎁", first)

    def test_slug_cannot_escape_the_managed_directory(self):
        slug = self.routes._slugify("../../etc/passwd")
        self.assertNotIn("/", slug)

    def test_blank_name_still_produces_a_filename(self):
        self.assertEqual(self.routes._slugify("   "), "script")

    def test_generated_scripts_do_not_share_the_server_scripts_dir(self):
        # On a script install the server tree *is* ~/.code-bridge, and its
        # scripts/ holds the server's own tooling. Writing drafts there mixes
        # generated files with shipped code.
        self.assertNotEqual(self.routes._MANAGED_SCRIPTS_DIR.name, "scripts")

    def test_suggested_name_is_short_not_the_whole_intent(self):
        name = self.routes._suggested_name(
            "Take the android device serial as $1, check it is connected with adb and report"
        )
        self.assertLessEqual(len(name.split()), 5)
        self.assertNotIn(",", name)

    def test_suggested_name_handles_non_ascii(self):
        self.assertTrue(
            self.routes._suggested_name("스몰뎁 앱에서 대기 중인 테스트에 참여").startswith("스몰뎁")
        )

    def test_suggested_name_never_empty(self):
        self.assertEqual(self.routes._suggested_name("!!!"), "script")


class LlmStepResultTest(unittest.TestCase):
    """A completed LLM step has to carry what the model said.

    The escalation step exists to answer "why did the script fail". It was
    storing only "Workflow step completed." — the answer lived in the event
    log, which is not where anyone reading a run looks.
    """

    def _sink(self):
        from agent.task_orchestrator import AgentTaskRunSink

        return AgentTaskRunSink(run_id="run_1")

    def test_complete_event_is_captured(self):
        sink = self._sink()
        asyncio.run(sink.send_json({"type": "complete", "content": "  tapped the wrong row  "}))
        self.assertEqual(sink.result_text, "tapped the wrong row")

    def test_result_event_is_captured(self):
        sink = self._sink()
        asyncio.run(sink.send_json({"type": "result", "result": "exit 1, home tab missing"}))
        self.assertEqual(sink.result_text, "exit 1, home tab missing")

    def test_blank_content_is_ignored(self):
        sink = self._sink()
        asyncio.run(sink.send_json({"type": "complete", "content": "   "}))
        self.assertIsNone(sink.result_text)

    def test_unrelated_events_leave_it_alone(self):
        sink = self._sink()
        asyncio.run(sink.send_json({"type": "assistant", "message": {"content": "thinking"}}))
        self.assertIsNone(sink.result_text)
