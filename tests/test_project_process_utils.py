import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_process_utils import is_process_running, terminate_process_safely


class _FakeProcess:
    def __init__(
        self,
        *,
        poll_value=None,
        terminate_error: Exception | None = None,
        wait_error_on_first: Exception | None = None,
        kill_error: Exception | None = None,
    ):
        self._poll_value = poll_value
        self._terminate_error = terminate_error
        self._wait_error_on_first = wait_error_on_first
        self._kill_error = kill_error
        self.terminate_called = 0
        self.kill_called = 0
        self.wait_calls: list[float] = []

    def poll(self):
        return self._poll_value

    def terminate(self):
        self.terminate_called += 1
        if self._terminate_error is not None:
            raise self._terminate_error

    def wait(self, timeout: float):
        self.wait_calls.append(timeout)
        if len(self.wait_calls) == 1 and self._wait_error_on_first is not None:
            raise self._wait_error_on_first

    def kill(self):
        self.kill_called += 1
        if self._kill_error is not None:
            raise self._kill_error


class ProjectProcessUtilsTest(unittest.TestCase):
    def test_is_process_running(self):
        self.assertTrue(is_process_running(_FakeProcess(poll_value=None)))
        self.assertFalse(is_process_running(_FakeProcess(poll_value=1)))

    def test_terminate_process_safely_graceful(self):
        process = _FakeProcess()

        terminate_process_safely(process, terminate_timeout=7.0, kill_timeout=3.0)

        self.assertEqual(process.terminate_called, 1)
        self.assertEqual(process.kill_called, 0)
        self.assertEqual(process.wait_calls, [7.0])

    def test_terminate_process_safely_falls_back_to_kill(self):
        process = _FakeProcess(wait_error_on_first=TimeoutError("timeout"))

        terminate_process_safely(process, terminate_timeout=5.0, kill_timeout=2.0)

        self.assertEqual(process.terminate_called, 1)
        self.assertEqual(process.kill_called, 1)
        self.assertEqual(process.wait_calls, [5.0, 2.0])

    def test_terminate_process_safely_swallows_all_errors(self):
        process = _FakeProcess(
            terminate_error=RuntimeError("terminate failed"),
            kill_error=RuntimeError("kill failed"),
        )

        terminate_process_safely(process)

        self.assertEqual(process.terminate_called, 1)
        self.assertEqual(process.kill_called, 1)


if __name__ == "__main__":
    unittest.main()
