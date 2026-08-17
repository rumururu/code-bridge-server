"""A value one step reads must reach the step that uses it.

A workflow is written before it runs, so it cannot contain the ids a site only
reveals at runtime. The Configurator authors exactly the right shape for that —
one step extracts `cafe_id` from the page, the next navigates to a URL built
from `{{cafe_id}}` — but each step is its own adapter call, and the value died
at the boundary. Measured: step 1 completed with `cafe_id = 31245773`, step 2
parked with `browser_action_needs_concrete_target` on the reference step 1 had
just answered.

Scoped to the run on purpose. An id read yesterday is not this run's evidence,
and quietly reusing it would send the step somewhere nobody looked at.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.task_orchestrator import _bindings_from_earlier_steps  # noqa: E402


def _step(step_id, run_id, extracted=None, **extra):
    step = {"id": step_id, "run_id": run_id, **extra}
    if extracted is not None:
        step["output"] = {"browser_action": {"extracted": extracted}}
    return step


class _Store:
    def __init__(self, steps) -> None:
        self._steps = steps

    def list_task_steps(self, task_id):
        return self._steps


class _BrokenStore:
    def list_task_steps(self, task_id):
        raise RuntimeError("db is gone")


class CrossStepBindingsTest(unittest.TestCase):
    def _bindings(self, steps, *, step_id="s2", run_id="run_1"):
        return _bindings_from_earlier_steps(
            _Store(steps), run_id=run_id, task_id="task_1", step_id=step_id
        )

    def test_an_earlier_extraction_reaches_the_later_step(self) -> None:
        bindings = self._bindings(
            [_step("s1", "run_1", [{"name": "cafe_id", "value": "31245773"}])]
        )
        self.assertEqual(bindings, {"cafe_id": "31245773"})

    def test_a_value_from_another_run_is_not_reused(self) -> None:
        """Yesterday's id is not this run's evidence."""
        bindings = self._bindings(
            [_step("s1", "run_0", [{"name": "cafe_id", "value": "999"}])]
        )
        self.assertEqual(bindings, {})

    def test_the_step_does_not_read_its_own_output(self) -> None:
        """On a resume the step already has output; feeding it back would let a
        stale reading answer the reference it is about to take again."""
        bindings = self._bindings(
            [_step("s2", "run_1", [{"name": "cafe_id", "value": "stale"}])]
        )
        self.assertEqual(bindings, {})

    def test_the_freshest_reading_wins(self) -> None:
        bindings = self._bindings(
            [
                _step("s1", "run_1", [{"name": "token", "value": "old"}]),
                _step("s1b", "run_1", [{"name": "token", "value": "new"}]),
            ]
        )
        self.assertEqual(bindings, {"token": "new"})

    def test_unnamed_extractions_are_ignored(self) -> None:
        """`extract` without a name is a plain read, not a variable."""
        bindings = self._bindings(
            [_step("s1", "run_1", [{"text": "some page text"}])]
        )
        self.assertEqual(bindings, {})

    def test_a_pattern_that_matched_nothing_binds_nothing(self) -> None:
        """No `value` key means the pattern failed; binding an empty string
        would send the next action to a truncated URL instead of stopping."""
        bindings = self._bindings(
            [_step("s1", "run_1", [{"name": "cafe_id", "matched": False}])]
        )
        self.assertEqual(bindings, {})

    def test_steps_without_browser_output_are_skipped(self) -> None:
        bindings = self._bindings(
            [
                {"id": "s0", "run_id": "run_1", "output": {"llm": {"text": "hi"}}},
                {"id": "s0b", "run_id": "run_1"},
                _step("s1", "run_1", [{"name": "cafe_id", "value": "1"}]),
            ]
        )
        self.assertEqual(bindings, {"cafe_id": "1"})

    def test_an_unreadable_store_parks_rather_than_crashes(self) -> None:
        """A missing binding stops the step and asks. A raised exception here
        would fail the whole run over a lookup."""
        self.assertEqual(
            _bindings_from_earlier_steps(
                _BrokenStore(), run_id="run_1", task_id="task_1", step_id="s2"
            ),
            {},
        )


class TheAdapterAcceptsSeededBindingsTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_seeded_binding_fills_a_reference(self) -> None:
        from agent.browser_action_adapter import bind_action

        self.assertEqual(
            bind_action(
                {"type": "navigate", "url": "https://x.test/{{cafe_id}}"},
                {"cafe_id": "31245773"},
            )["url"],
            "https://x.test/31245773",
        )


if __name__ == "__main__":
    unittest.main()
