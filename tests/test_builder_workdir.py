"""Where a conversationally built agent works, and what happens when nobody says.

The Configurator used to never ask. ``cwd`` and ``workspace`` appeared nowhere
in its system prompt, so ``task_draft.cwd`` stayed empty, ``builder_commit``
stored no ``cwd`` in the task metadata,
``task_orchestrator._resolve_project_path`` fell through to
``_global_task_path()``, and the run's workspace root became Code Bridge's
shared global chat directory. Every file in the user's actual project is then
outside that root, which ``policy/path_guard.py`` downgrades to
``confirm_each`` -- so an agent built to run unattended stopped to ask
permission for each routine read, on every fire.

These tests pin the three halves of the fix:

- the prompt offers the user's registered projects by name, so the model has
  something to ask about that it did not invent;
- a project the user names is resolved to that project's real path and carried
  into the task draft the commit turns into task metadata;
- a value that names nothing is refused rather than written into the run's
  root, and the turn says so out loud instead of defaulting silently.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.agent_models import TaskDraft  # noqa: E402
from agent.configurator import (  # noqa: E402
    build_configurator_system_prompt,
    create_builder_session,
    resolve_task_draft_workdir,
    workdir_disclosure,
)

_PROJECTS = [
    {"name": "code_bridge", "path": "/Users/mankil/AndroidStudioProjects/code_bridge"},
    {"name": "lottosignal", "path": "/Users/mankil/AndroidStudioProjects/lottosignal"},
]


class _FakeProjectDB:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows

    def get_all(self) -> list[dict[str, str]]:
        return list(self._rows)


def _with_projects(rows: list[dict[str, str]]):
    """Patch the project table `_registered_project_rows` imports at call time."""
    return mock.patch("core.database.get_project_db", return_value=_FakeProjectDB(rows))


def _draft_response(cwd_line: str, *, ready: bool = False) -> str:
    ready_marker = "READY_TO_COMMIT\n" if ready else ""
    return f"""
Here is the agent.

```draft
{{
  "name": "Nightly tidy",
  "description": "Tidies the repository overnight.",
  "system_prompt": "Tidy the repository.",
  "provider_id": "anthropic",
  "tools": [],
  "flow": [
    {{
      "id": "report",
      "type": "notify",
      "name": "Report",
      "description": "Tell the user what happened.",
      "notify": {{"title": "Nightly tidy", "body": "done", "level": "info"}},
      "success_criteria": "The user was told"
    }}
  ],
  "memory_seeds": []
}}
```

```task_draft
{{
  "goal": "Tidy the repository overnight",
  "schedule": "daily 03:00"{cwd_line}
}}
```
{ready_marker}"""


class RegisteredProjectsPromptTest(unittest.TestCase):
    def test_prompt_lists_registered_projects_by_name_and_path(self) -> None:
        with _with_projects(_PROJECTS):
            prompt = build_configurator_system_prompt()

        self.assertIn("Registered projects", prompt)
        self.assertIn("code_bridge: /Users/mankil/AndroidStudioProjects/code_bridge", prompt)
        self.assertIn("lottosignal: /Users/mankil/AndroidStudioProjects/lottosignal", prompt)
        # The question itself, not only the data behind it.
        self.assertIn("task_draft.cwd", prompt)

    def test_prompt_with_no_projects_offers_no_choice_and_forbids_inventing_one(self) -> None:
        with _with_projects([]):
            prompt = build_configurator_system_prompt()

        self.assertIn("Registered projects: none.", prompt)
        self.assertIn("Never invent a path.", prompt)

    def test_prompt_survives_an_unavailable_project_table(self) -> None:
        # A builder turn must not fail because the project table is mid-migration.
        with mock.patch("core.database.get_project_db", side_effect=RuntimeError("no db")):
            prompt = build_configurator_system_prompt()

        self.assertIn("Agent Builder Configurator", prompt)
        self.assertNotIn("Registered projects", prompt)


class ResolveWorkdirTest(unittest.TestCase):
    def test_project_name_becomes_that_projects_path(self) -> None:
        with _with_projects(_PROJECTS):
            resolution = resolve_task_draft_workdir(TaskDraft(goal="g", cwd="code_bridge"))

        self.assertEqual(
            resolution.task_draft.cwd,
            "/Users/mankil/AndroidStudioProjects/code_bridge",
        )
        self.assertEqual(resolution.project_name, "code_bridge")
        self.assertIsNone(resolution.refused)

    def test_project_name_matches_case_insensitively(self) -> None:
        with _with_projects(_PROJECTS):
            resolution = resolve_task_draft_workdir(TaskDraft(goal="g", cwd="Code_Bridge"))

        self.assertEqual(
            resolution.task_draft.cwd,
            "/Users/mankil/AndroidStudioProjects/code_bridge",
        )

    def test_a_registered_path_is_kept_and_credited_to_its_project(self) -> None:
        with _with_projects(_PROJECTS):
            resolution = resolve_task_draft_workdir(
                TaskDraft(goal="g", cwd="/Users/mankil/AndroidStudioProjects/lottosignal/")
            )

        self.assertEqual(
            resolution.task_draft.cwd,
            "/Users/mankil/AndroidStudioProjects/lottosignal",
        )
        self.assertEqual(resolution.project_name, "lottosignal")

    def test_an_unregistered_absolute_path_is_kept_as_written(self) -> None:
        # Somebody typed a real path in the phone's working-folder field. The
        # builder does not overrule an explicit instruction.
        with _with_projects(_PROJECTS):
            resolution = resolve_task_draft_workdir(TaskDraft(goal="g", cwd="/tmp/scratch"))

        self.assertEqual(resolution.task_draft.cwd, "/tmp/scratch")
        self.assertIsNone(resolution.project_name)
        self.assertIsNone(resolution.refused)

    def test_a_value_that_names_nothing_is_refused_not_stored(self) -> None:
        with _with_projects(_PROJECTS):
            resolution = resolve_task_draft_workdir(
                TaskDraft(goal="g", cwd="the project folder")
            )

        self.assertIsNone(resolution.task_draft.cwd)
        self.assertEqual(resolution.refused, "the project folder")

        note = workdir_disclosure(resolution=resolution, announce_global_fallback=False)
        self.assertIsNotNone(note)
        self.assertIn("the project folder", note)
        self.assertIn("공용 작업 폴더", note)

    def test_no_disclosure_when_a_folder_was_established(self) -> None:
        with _with_projects(_PROJECTS):
            resolution = resolve_task_draft_workdir(TaskDraft(goal="g", cwd="code_bridge"))

        self.assertIsNone(
            workdir_disclosure(resolution=resolution, announce_global_fallback=True)
        )


class BuilderTurnWorkdirTest(unittest.TestCase):
    def test_a_named_project_reaches_the_task_draft_as_a_real_path(self) -> None:
        session = create_builder_session(system_prompt="test")

        with _with_projects(_PROJECTS):
            session.apply_llm_response(
                _draft_response(',\n  "cwd": "code_bridge"', ready=True),
                user_message="매일 새벽 3시에 code_bridge 정리해줘",
            )

        self.assertIsNotNone(session.task_draft)
        self.assertEqual(
            session.task_draft.cwd,
            "/Users/mankil/AndroidStudioProjects/code_bridge",
        )

    def test_the_ready_turn_with_no_folder_says_so_instead_of_defaulting_quietly(
        self,
    ) -> None:
        session = create_builder_session(system_prompt="test")

        with _with_projects(_PROJECTS):
            parsed = session.apply_llm_response(
                _draft_response("", ready=True),
                user_message="매일 새벽 3시에 정리해줘",
            )

        self.assertIsNone(session.task_draft.cwd)
        self.assertIn("일할 폴더를 정하지 않았습니다", parsed.assistant_message)
        self.assertIn("승인을 요청합니다", parsed.assistant_message)

    def test_a_mid_conversation_turn_is_not_nagged(self) -> None:
        # Same missing folder, but the draft is not finished: the model still
        # has turns left to ask rule 6a's question itself.
        session = create_builder_session(system_prompt="test")

        with _with_projects(_PROJECTS):
            parsed = session.apply_llm_response(
                _draft_response(""),
                user_message="매일 새벽 3시에 정리해줘",
            )

        self.assertNotIn("일할 폴더를 정하지 않았습니다", parsed.assistant_message)

    def test_a_refused_folder_is_disclosed_even_mid_conversation(self) -> None:
        session = create_builder_session(system_prompt="test")

        with _with_projects(_PROJECTS):
            parsed = session.apply_llm_response(
                _draft_response(',\n  "cwd": "my project"'),
                user_message="매일 새벽 3시에 내 프로젝트 정리해줘",
            )

        self.assertIsNone(session.task_draft.cwd)
        self.assertIn("my project", parsed.assistant_message)


class WorkdirReachesTheRunTest(unittest.TestCase):
    """The half of the fix that lives outside the builder.

    `builder_commit` copies `task_draft.cwd` into the task's metadata
    (`routes/agents.py`, `metadata={... "cwd": task_draft.cwd}`), and the
    orchestrator reads it back when it plans the run. Without that link the
    resolution above would be a field nobody consumes, so it is asserted here
    rather than assumed.
    """

    def test_metadata_cwd_becomes_the_runs_root(self) -> None:
        from agent.task_orchestrator import _global_task_path, _resolve_project_path

        task = {
            "id": "task_1",
            "metadata": {"cwd": "/Users/mankil/AndroidStudioProjects/code_bridge"},
        }
        self.assertEqual(
            _resolve_project_path(task, cwd=None),
            "/Users/mankil/AndroidStudioProjects/code_bridge",
        )
        # And the behaviour the disclosure warns about, with no cwd at all.
        self.assertEqual(_resolve_project_path({"id": "task_2"}, cwd=None), _global_task_path())


class BuilderCommitCarriesTheProjectTest(unittest.TestCase):
    """The task the builder writes must name the project, not just the folder.

    `POST /tasks` has always passed `project_name` (`routes/agents.py`), but the
    builder's own `create_task` call did not — so an agent built by talking got
    a task under the `__global__` sentinel even when the conversation had named
    a real project. Everything downstream keys off that name: the run's files
    land outside any workspace, so the path guard asks permission for routine
    reads, and a standing rule has no project to attach to and falls back to the
    run (which the next scheduled fire does not share). The `cwd` alone does not
    fix it; the name has to travel too.
    """

    def test_the_commit_route_passes_the_resolved_project_name(self) -> None:
        import inspect

        from routes import agents as agents_routes

        source = inspect.getsource(agents_routes.builder_commit)
        # The resolver runs before the task is written...
        self.assertIn("resolve_task_draft_workdir(task_draft)", source)
        # ...and its verdict is what `create_task` is told.
        self.assertIn("project_name=workdir.project_name", source)

    def test_the_resolver_names_the_project_the_commit_will_store(self) -> None:
        with _with_projects(_PROJECTS):
            resolution = resolve_task_draft_workdir(
                TaskDraft(goal="Check the build", cwd="code_bridge")
            )
        self.assertEqual(resolution.project_name, "code_bridge")
        self.assertIsNone(resolution.refused)


if __name__ == "__main__":
    unittest.main()
