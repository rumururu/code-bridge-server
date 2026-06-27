"""End-to-end test for step.log streaming over /ws/agent/runs/{run_id}.

Covers three layers:

1. TerminalSession.execute_with_emitter — chunks reach on_chunk callback as
   stdout/stderr arrives.
2. task_orchestrator._make_step_log_emitter — chunks become step.log events
   in agent_events, line-batched for stdout/stderr separately.
3. WS pickup — agent_store.list_events sees step.log events after the run
   has been created, ordered by sequence.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from agent import agent_store  # noqa: E402
from agent.agent_store import get_agent_store  # noqa: E402
from agent.task_orchestrator import _make_step_log_emitter  # noqa: E402
from core import database  # noqa: E402
from terminal import TerminalSession  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_agent_db(tmp_path):
    """Keep step-log persistence tests out of the live development database."""
    original_db_path = database.DB_PATH
    database.DB_PATH = Path(tmp_path) / "code_bridge_step_log_test.db"
    agent_store._agent_store = None
    database.init_db()
    yield
    agent_store._agent_store = None
    database.DB_PATH = original_db_path


# ---------------------------------------------------------------------------
# Layer 1: TerminalSession.execute_with_emitter
# ---------------------------------------------------------------------------


def test_execute_with_emitter_calls_back_for_stdout():
    async def run():
        with tempfile.TemporaryDirectory() as cwd:
            session = TerminalSession(project_path=cwd)
            chunks: list[dict] = []

            result = await session.execute_with_emitter(
                "printf 'line1\\nline2\\n'",
                on_chunk=lambda c: chunks.append(c),
                timeout=5,
            )

            assert result.exit_code == 0
            assert "line1" in result.stdout
            assert "line2" in result.stdout
            assert chunks, "emitter received no chunks"
            joined = "".join(c["data"] for c in chunks if c["stream"] == "stdout")
            assert "line1" in joined
            assert "line2" in joined

    asyncio.run(run())


def test_execute_with_emitter_captures_stderr():
    async def run():
        with tempfile.TemporaryDirectory() as cwd:
            session = TerminalSession(project_path=cwd)
            chunks: list[dict] = []

            await session.execute_with_emitter(
                "printf 'oops\\n' 1>&2",
                on_chunk=lambda c: chunks.append(c),
                timeout=5,
            )

            stderr_chunks = [c for c in chunks if c["stream"] == "stderr"]
            assert stderr_chunks, "expected stderr chunk emission"

    asyncio.run(run())


def test_execute_with_emitter_swallows_callback_errors():
    """A broken listener must NOT corrupt command execution."""

    async def run():
        with tempfile.TemporaryDirectory() as cwd:
            session = TerminalSession(project_path=cwd)

            def broken(_chunk):
                raise RuntimeError("listener exploded")

            result = await session.execute_with_emitter(
                "echo hi",
                on_chunk=broken,
                timeout=5,
            )

            assert result.exit_code == 0
            assert "hi" in result.stdout

    asyncio.run(run())


def test_execute_with_emitter_supports_async_callback():
    async def run():
        with tempfile.TemporaryDirectory() as cwd:
            session = TerminalSession(project_path=cwd)
            chunks: list[dict] = []

            async def async_collect(chunk):
                chunks.append(chunk)

            await session.execute_with_emitter(
                "echo async-friendly",
                on_chunk=async_collect,
                timeout=5,
            )

            assert any("async-friendly" in c["data"] for c in chunks)

    asyncio.run(run())


def test_execute_with_emitter_runs_without_callback():
    """on_chunk is optional; the method must still return a CommandResult."""

    async def run():
        with tempfile.TemporaryDirectory() as cwd:
            session = TerminalSession(project_path=cwd)
            result = await session.execute_with_emitter("echo nocb", timeout=5)
            assert result.exit_code == 0
            assert "nocb" in result.stdout

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Layer 2: _make_step_log_emitter — chunks -> step.log events
# ---------------------------------------------------------------------------


def _create_run_for_test() -> str:
    store = get_agent_store()
    run = store.create_run(
        project_name=None,
        workspace_id=None,
        provider_id="test_provider",
        model="test_model",
        title="step log test",
    )
    return run["id"]


def test_make_step_log_emitter_returns_none_when_no_run_id():
    emitter = _make_step_log_emitter(
        run_id=None,
        task_id="task_x",
        step_id="step_x",
        command="echo hi",
    )
    assert emitter is None


def test_emitter_flushes_line_terminated_chunks_into_events():
    run_id = _create_run_for_test()
    store = get_agent_store()

    emitter = _make_step_log_emitter(
        run_id=run_id,
        task_id="task_emitter",
        step_id="step_emitter",
        command="echo hi",
    )
    assert emitter is not None

    # Two complete stdout lines plus one stderr line.
    emitter({"stream": "stdout", "data": "alpha\n"})
    emitter({"stream": "stdout", "data": "beta\n"})
    emitter({"stream": "stderr", "data": "warn\n"})

    events = store.list_events(run_id, after_sequence=0, limit=50)
    log_events = [e for e in events if e["event_type"] == "step.log"]
    assert len(log_events) == 3

    payloads = [e["app_event"] for e in log_events]
    streams = [p["stream"] for p in payloads]
    datas = [p["data"] for p in payloads]
    assert streams == ["stdout", "stdout", "stderr"]
    assert datas == ["alpha", "beta", "warn"]
    for p in payloads:
        assert p["task_id"] == "task_emitter"
        assert p["step_id"] == "step_emitter"
        assert p["command"] == "echo hi"


def test_emitter_coalesces_partial_chunks_until_newline():
    run_id = _create_run_for_test()
    store = get_agent_store()

    emitter = _make_step_log_emitter(
        run_id=run_id,
        task_id="task_partial",
        step_id="step_partial",
        command="echo partial",
    )

    # A line trickled in three sub-chunks.
    emitter({"stream": "stdout", "data": "hel"})
    emitter({"stream": "stdout", "data": "lo wo"})
    # No event yet — line not complete.
    interim_events = [
        e for e in store.list_events(run_id, after_sequence=0, limit=50)
        if e["event_type"] == "step.log"
    ]
    assert interim_events == []

    emitter({"stream": "stdout", "data": "rld\n"})
    final_events = [
        e for e in store.list_events(run_id, after_sequence=0, limit=50)
        if e["event_type"] == "step.log"
    ]
    assert len(final_events) == 1
    assert final_events[0]["app_event"]["data"] == "hello world"


def test_emitter_ignores_malformed_chunks():
    run_id = _create_run_for_test()
    store = get_agent_store()

    emitter = _make_step_log_emitter(
        run_id=run_id,
        task_id="task_bad",
        step_id="step_bad",
        command="noop",
    )

    # Various invalid shapes — none should produce events or raise.
    emitter("not a dict")
    emitter({"stream": "unknown", "data": "ignored"})
    emitter({"stream": "stdout"})  # no data
    emitter({"stream": "stdout", "data": 12345})  # non-string

    events = [
        e for e in store.list_events(run_id, after_sequence=0, limit=50)
        if e["event_type"] == "step.log"
    ]
    assert events == []


def test_emitter_flushes_long_partial_line_without_newline():
    """Progress bars use \\r without \\n; emitter must still surface output."""

    run_id = _create_run_for_test()
    store = get_agent_store()

    emitter = _make_step_log_emitter(
        run_id=run_id,
        task_id="task_progress",
        step_id="step_progress",
        command="show progress",
    )

    # Push 300 chars of progress in a single chunk — emitter should flush
    # once the partial line crosses the 256-byte threshold.
    emitter({"stream": "stdout", "data": "P" * 300})

    events = [
        e for e in store.list_events(run_id, after_sequence=0, limit=50)
        if e["event_type"] == "step.log"
    ]
    assert events, "expected emitter to flush long partial line"
    assert events[0]["app_event"]["data"].startswith("PPP")


# ---------------------------------------------------------------------------
# Layer 3: end-to-end terminal + emitter -> events ordered by sequence
# ---------------------------------------------------------------------------


def test_terminal_emitter_writes_events_in_sequence_order():
    run_id = _create_run_for_test()
    store = get_agent_store()

    emitter = _make_step_log_emitter(
        run_id=run_id,
        task_id="task_seq",
        step_id="step_seq",
        command="printf 'one\\ntwo\\nthree\\n'",
    )
    assert emitter is not None

    async def run():
        with tempfile.TemporaryDirectory() as cwd:
            session = TerminalSession(project_path=cwd)
            result = await session.execute_with_emitter(
                "printf 'one\\ntwo\\nthree\\n'",
                on_chunk=emitter,
                timeout=5,
            )
            assert result.exit_code == 0

    asyncio.run(run())

    events = [
        e for e in store.list_events(run_id, after_sequence=0, limit=50)
        if e["event_type"] == "step.log"
    ]
    # Three lines of stdout become three step.log events in order.
    stdout_events = [e for e in events if e["app_event"]["stream"] == "stdout"]
    sequences = [e["sequence"] for e in stdout_events]
    assert sequences == sorted(sequences)
    datas = [e["app_event"]["data"] for e in stdout_events]
    assert datas == ["one", "two", "three"]
