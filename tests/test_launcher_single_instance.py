"""Only one tray launcher may run at a time.

Start-at-login is registered by the installer, so opening ``start-menubar.sh``
by hand is enough to end up with two menu-bar icons. Only the first one's
server can bind 8766/8767; the second shows a menu that controls nothing and
whose Start fails with "address already in use". This is invisible in normal
use — the icons look identical — so it is pinned here.
"""

import importlib.util
import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "desktop_server_app" / "launcher.py"
SPEC = importlib.util.spec_from_file_location("desktop_launcher", SCRIPT_PATH)
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _try_lock_in_child(lock_path: str, result: "multiprocessing.Queue[bool]") -> None:
    handle = launcher.acquire_single_instance_lock(Path(lock_path))
    result.put(handle is not None)
    if handle is not None:
        handle.close()


class SingleInstanceLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.lock_path = Path(self.tmpdir.name) / "launcher.lock"

    def _child_can_lock(self) -> bool:
        # A real second process, not a second call in this one: POSIX flock is
        # per-open-file-description, so an in-process re-lock would happily
        # succeed and prove nothing.
        context = multiprocessing.get_context("spawn")
        queue: "multiprocessing.Queue[bool]" = context.Queue()
        process = context.Process(
            target=_try_lock_in_child, args=(str(self.lock_path), queue)
        )
        process.start()
        process.join(timeout=30)
        self.assertFalse(process.is_alive(), "lock child hung")
        return queue.get(timeout=5)

    def test_first_caller_gets_the_lock(self):
        handle = launcher.acquire_single_instance_lock(self.lock_path)
        self.addCleanup(handle.close)
        self.assertIsNotNone(handle)
        self.assertTrue(self.lock_path.exists())

    def test_second_process_is_refused_while_the_first_holds_it(self):
        handle = launcher.acquire_single_instance_lock(self.lock_path)
        self.addCleanup(handle.close)
        self.assertFalse(self._child_can_lock())

    def test_lock_is_released_when_the_holder_exits(self):
        # The child takes and drops the lock; this process must then get it.
        self.assertTrue(self._child_can_lock())
        handle = launcher.acquire_single_instance_lock(self.lock_path)
        self.addCleanup(handle.close)
        self.assertIsNotNone(handle)

    def test_lock_file_records_the_holder_pid(self):
        handle = launcher.acquire_single_instance_lock(self.lock_path)
        self.addCleanup(handle.close)
        self.assertEqual(self.lock_path.read_text().strip(), str(__import__("os").getpid()))

    def test_unwritable_location_does_not_block_startup(self):
        # A broken state dir must not leave the user with no launcher at all.
        missing = Path(self.tmpdir.name) / "nope" / "deeper" / "launcher.lock"
        self.assertIsNone(launcher.acquire_single_instance_lock(missing))


if __name__ == "__main__":
    unittest.main()
