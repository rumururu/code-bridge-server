"""What the run pane says after a run, and while one is still going.

Two defects found by creating an agent through the Configurator and pressing
"Run once now" — neither visible from reading the code, and neither caught by
any suite that was already green.

**A finished run went on saying "running".** The run-once POST returns the
moment the run *starts*, so the reload it triggers snapshots `running`, and
nothing ever asked again: the label stayed frozen until the whole page was
reloaded. The same silence covered a scheduled run firing while the page sat
open — the case nobody would think to reload for.

**A settled step printed its record as JSON.** The server writes a whole
sentence for a step that never ran (which condition took which arm, or where
the run ended). The phone renders that sentence; this pane fell through to
the raw-JSON branch and showed `{"skipped": {"reason": …}}` to the reader.

Both are checked by executing the template's own functions in node. A
`switch` arm that stopped matching, or a poll that never stops, passes any
string-containment test ever written.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard_js import (  # noqa: E402
    COMMON_STUBS,
    NODE,
    js_const,
    js_function,
    run_js,
)


@unittest.skipIf(NODE is None, "node is not installed on this machine")
class SkippedStepReadsAsASentenceTest(unittest.TestCase):
    #: Copied from a real run: a linear flow whose first step ended it.
    STEP = {
        "id": "step_6a52",
        "title": "용량 경고 알림",
        "status": "skipped",
        "input": {"workflow_type": "notify", "workflow_step_id": "notify_high"},
        "output": {
            "skipped": {
                "reason": "run_ended",
                "message": (
                    "이 run은 '루트 파일시스템 사용률 점검'에서 끝났습니다. "
                    "이 스텝은 차례가 오지 않았고, 이번 run에서는 실행되지 않습니다."
                ),
                "run_status": "completed",
                "last_workflow_step_id": "check_disk",
            }
        },
    }

    def _render(self, step: dict) -> str:
        return run_js(
            "\n".join(
                [
                    COMMON_STUBS,
                    js_function("renderMcpToolsEvent"),
                    js_function("renderStepResult"),
                    f"console.log(renderStepResult({json.dumps(step)}, []));",
                ]
            )
        )

    def test_the_sentence_is_shown_and_the_json_is_not(self):
        html = self._render(self.STEP)
        self.assertIn("차례가 오지 않았고", html)
        self.assertIn("run_ended", html)

        # The tell-tale of the raw dump this replaced, and it has to be
        # written the way the page writes it: `escapeHtml` turns `"` into
        # `&quot;`, so an assertion against `'"reason"'` can never fail —
        # which is what the first version of this test did, and it passed
        # happily with the sentence branch deleted.
        self.assertNotIn("&quot;reason&quot;", html)
        self.assertNotIn("last_workflow_step_id", html)
        # Belt and braces: the JSON dump is a <pre class="log-pane">, and the
        # sentence is not.
        self.assertNotIn("log-pane", html)

    def test_a_branch_skip_reads_the_same_way(self):
        # The other reason a step settles: an arm the condition did not take.
        # One renderer for both, so a new reason cannot fall back to JSON.
        step = json.loads(json.dumps(self.STEP))
        step["output"]["skipped"] = {
            "reason": "branch_not_taken",
            "message": "'check_exit' 조건이 갈래 0('실패')를 선택해 'diagnose'로 갔습니다.",
            "condition_step_id": "check_exit",
        }
        html = self._render(step)
        self.assertIn("조건이 갈래", html)
        self.assertNotIn("condition_step_id", html)
        self.assertNotIn("log-pane", html)

    def test_a_record_with_no_sentence_still_shows_everything_it_has(self):
        # Never lose information: an older record, or one this build does not
        # understand, falls back to the raw dump rather than rendering blank.
        step = json.loads(json.dumps(self.STEP))
        step["output"]["skipped"] = {"reason": "something_new"}
        html = self._render(step)
        self.assertIn("something_new", html)
        # It fell through to the raw dump rather than rendering an empty box.
        self.assertIn("log-pane", html)

    def test_an_ordinary_step_is_untouched(self):
        html = self._render(
            {
                "id": "s1",
                "title": "루트 파일시스템 사용률 점검",
                "status": "completed",
                "input": {"workflow_type": "shell"},
                "output": {
                    "shell": {"exit_code": 0, "stdout": "USED_PCT=11\n", "stderr": ""}
                },
            }
        )
        self.assertIn("USED_PCT=11", html)
        self.assertIn("exit 0", html)


@unittest.skipIf(NODE is None, "node is not installed on this machine")
class AnInFlightRunIsFollowedTest(unittest.TestCase):
    """The poll runs while something is live and stops when nothing is.

    Both halves matter. Without the first, a finished run keeps saying
    "running"; without the second, an idle dashboard asks the server for the
    run list every three seconds for as long as it is open.
    """

    #: A fake clock, `api`, and DOM, so the template's own functions run.
    HARNESS = f"""
    {js_const('LIVE_RUN')}
    let runs = [];
    let selectedRunId = null;
    let requests = 0;
    let ticks = [];
    let timerId = 0;
    let liveRunTimer = null;
    const LIVE_RUN_POLL_MS = 3000;
    const timers = new Map();
    globalThis.setInterval = (fn, ms) => {{ timerId += 1; timers.set(timerId, fn); return timerId; }};
    globalThis.clearInterval = (id) => {{ timers.delete(id); }};
    const renderRunList = () => {{}};
    const selectRun = (id) => {{ ticks.push('detail:' + id); }};
    """

    def _run(
        self, held: list[dict], answers: list[list[dict]], pumps: int
    ) -> dict:
        """Drive the follow loop.

        `held` is what the page already has when the follow starts; `answers`
        is what each successive poll gets back. Kept as two arguments because
        an earlier version passed one list and popped its head for `held`
        *after* serializing it — so the first poll replayed the page's own
        state and the request count was one higher than the test claimed. The
        code was right and the harness was not.
        """
        script = "\n".join(
            [
                self.HARNESS,
                f"const ANSWERS = {json.dumps(answers)};",
                """
                const api = async () => {
                  requests += 1;
                  const next = ANSWERS.shift();
                  return next === undefined ? { runs } : { runs: next };
                };
                """,
                js_function("refreshRunsOnce"),
                js_function("followLiveRuns"),
                # `anyRunLive` is an arrow const, not a function declaration.
                js_const("anyRunLive"),
                f"""
                (async () => {{
                  runs = {json.dumps(held)};
                  followLiveRuns();
                  for (let i = 0; i < {pumps}; i += 1) {{
                    for (const fn of [...timers.values()]) await fn();
                  }}
                  console.log(JSON.stringify({{
                    requests,
                    timersLeft: timers.size,
                    statuses: runs.map((run) => run.status),
                    ticks,
                  }}));
                }})();
                """,
            ]
        )
        return json.loads(run_js(script).strip())

    def test_a_run_that_finishes_stops_being_called_running(self):
        result = self._run(
            held=[{"id": "run_1", "status": "running"}],
            answers=[
                [{"id": "run_1", "status": "running"}],
                [{"id": "run_1", "status": "completed"}],
            ],
            pumps=4,
        )
        self.assertEqual(result["statuses"], ["completed"])
        # And it stopped: the timer is gone, so the extra pumps cost nothing.
        self.assertEqual(result["timersLeft"], 0)
        self.assertEqual(result["requests"], 2)

    def test_an_idle_page_polls_nothing_at_all(self):
        result = self._run(
            held=[{"id": "run_1", "status": "completed"}], answers=[], pumps=4
        )
        self.assertEqual(result["requests"], 0)
        self.assertEqual(result["timersLeft"], 0)

    def test_the_open_detail_is_reloaded_only_when_its_run_moves(self):
        script = "\n".join(
            [
                self.HARNESS,
                """
                let SEQUENCE = [
                  [{id: 'run_1', status: 'running'}],
                  [{id: 'run_1', status: 'completed'}],
                ];
                const api = async () => {
                  requests += 1;
                  const next = SEQUENCE.shift();
                  return next === undefined ? { runs } : { runs: next };
                };
                """,
                js_function("refreshRunsOnce"),
                """
                (async () => {
                  runs = [{id: 'run_1', status: 'running'}];
                  selectedRunId = 'run_1';
                  await refreshRunsOnce();  // status unchanged
                  const afterSame = ticks.length;
                  await refreshRunsOnce();  // status moved
                  console.log(JSON.stringify({ afterSame, ticks }));
                })();
                """,
            ]
        )
        result = json.loads(run_js(script).strip())
        # Watching a long run must not reload its whole step list every tick.
        self.assertEqual(result["afterSame"], 0)
        self.assertEqual(result["ticks"], ["detail:run_1"])


if __name__ == "__main__":
    unittest.main()
