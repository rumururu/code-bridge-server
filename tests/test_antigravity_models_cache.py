"""Listing one provider's models must never be able to take the server down.

This exists because it did. `_get_antigravity_models` shells out to the `agy`
CLI, and it is reached from `get_llm_options_snapshot`, which sits on request
paths — chat session creation, LLM commands. Those run on the event loop, so a
synchronous `subprocess.run` there stops the *entire server* answering for as
long as the CLI takes. With a 10s timeout, an uncached probe, and an `agy` that
did not return in time, every port stayed open and every request timed out.
From the phone it looked like "connection timeout" while the dashboard still
showed the server connected — because a server that cannot answer also cannot
correct a stale "connected".

So the cache here is not a speed optimisation and must not be removed as one:
it is the thing that bounds how much of the server one slow external binary can
consume. The same goes for the short timeout and the single-prober lock — drop
any of the three and a hanging CLI becomes a server outage again.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import llm_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cache():
    llm_settings.reset_antigravity_models_cache()
    yield
    llm_settings.reset_antigravity_models_cache()


def _completed(stdout: str = "gemini-3.1-pro-high\n", returncode: int = 0):
    result = mock.Mock()
    result.stdout = stdout
    result.returncode = returncode
    return result


def test_cli_is_consulted_once_not_per_call():
    """The probe is per-TTL, not per-request.

    Without this, every snapshot build spawns a process; N concurrent callers
    are N blocked event-loop turns.
    """
    with mock.patch.object(llm_settings.shutil, "which", return_value="/usr/bin/agy"), \
            mock.patch.object(llm_settings.subprocess, "run", return_value=_completed()) as run:
        first = llm_settings._get_antigravity_models()
        for _ in range(20):
            llm_settings._get_antigravity_models()

    assert run.call_count == 1, "the CLI must not be re-run for every caller"
    assert [m["id"] for m in first] == ["gemini-3.1-pro-high"]


def test_probe_timeout_is_short_enough_to_not_wedge_the_loop():
    """A slow CLI gets a few seconds, not ten.

    This number is how long the whole server can stop answering on a cache
    miss, so it is a budget, not a preference.
    """
    with mock.patch.object(llm_settings.shutil, "which", return_value="/usr/bin/agy"), \
            mock.patch.object(llm_settings.subprocess, "run", return_value=_completed()) as run:
        llm_settings._get_antigravity_models()

    timeout = run.call_args.kwargs.get("timeout")
    assert timeout is not None, "an un-timed subprocess can hang the server forever"
    assert timeout <= 5, f"probe timeout {timeout}s is too much event-loop time to give up"


def test_a_hanging_cli_serves_the_last_known_list_instead_of_blocking_again():
    """After a timeout, later callers get an answer immediately.

    The failure mode being prevented: a CLI that always times out, with no
    cache, means every single request pays the full timeout.
    """
    import subprocess as _subprocess

    with mock.patch.object(llm_settings.shutil, "which", return_value="/usr/bin/agy"), \
            mock.patch.object(
                llm_settings.subprocess,
                "run",
                side_effect=_subprocess.TimeoutExpired(cmd="agy", timeout=3),
            ) as run:
        first = llm_settings._get_antigravity_models()
        started = time.monotonic()
        second = llm_settings._get_antigravity_models()
        elapsed = time.monotonic() - started

    assert run.call_count == 1, "a timed-out probe must not be retried on the next request"
    assert first == second
    assert first, "a failed probe still has to return usable model names"
    assert elapsed < 0.5


def test_concurrent_cache_misses_spawn_one_process_not_many():
    """Ten simultaneous callers must not become ten simultaneous CLIs.

    A thundering herd on a cold cache is the same outage with more processes.
    """
    barrier = threading.Barrier(10)
    calls: list[float] = []

    def slow_run(*_args, **_kwargs):
        calls.append(time.monotonic())
        time.sleep(0.2)
        return _completed()

    with mock.patch.object(llm_settings.shutil, "which", return_value="/usr/bin/agy"), \
            mock.patch.object(llm_settings.subprocess, "run", side_effect=slow_run):
        def worker():
            barrier.wait()
            llm_settings._get_antigravity_models()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    assert len(calls) == 1, f"{len(calls)} concurrent probes; the lock is not holding"


def test_missing_cli_does_not_probe_repeatedly():
    """No `agy` installed is a stable answer, so stop asking the filesystem."""
    with mock.patch.object(llm_settings.shutil, "which", return_value=None) as which:
        for _ in range(5):
            models = llm_settings._get_antigravity_models()

    assert which.call_count == 1
    assert models, "a missing CLI still has to yield usable model names"
