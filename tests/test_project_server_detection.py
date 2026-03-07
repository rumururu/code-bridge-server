import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_server_detection import (
    detect_port_for_project,
    get_process_cwd,
    list_listening_processes,
)
from project_models import ProjectType


class ProjectServerDetectionTest(unittest.TestCase):
    def test_list_listening_processes_parses_lsof_output(self):
        fake_stdout = "\n".join(
            [
                "p101",
                "cnode",
                "n*:3000",
                "n127.0.0.1:5173->127.0.0.1:64999",
                "p102",
                "cpython",
                "n*:8000",
                "",
            ]
        )

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=fake_stdout, stderr="")

        listeners = list_listening_processes(run_command=fake_run)

        self.assertEqual(listeners[101]["command"], "node")
        self.assertEqual(listeners[101]["ports"], {3000, 5173})
        self.assertEqual(listeners[102]["command"], "python")
        self.assertEqual(listeners[102]["ports"], {8000})

    def test_list_listening_processes_returns_empty_on_os_error(self):
        def fake_run(*args, **kwargs):
            raise OSError("boom")

        self.assertEqual(list_listening_processes(run_command=fake_run), {})

    def test_get_process_cwd_parses_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd_path = str(Path(tmp_dir).resolve())
            fake_stdout = f"p123\nf20\nn{cwd_path}\n"

            def fake_run(*args, **kwargs):
                return subprocess.CompletedProcess(args=args, returncode=0, stdout=fake_stdout, stderr="")

            result = get_process_cwd(123, run_command=fake_run)

        self.assertEqual(result, Path(cwd_path))

    def test_detect_port_for_project_prefers_best_score(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir).resolve()
            nested = (project_root / "apps" / "web").resolve()
            nested.mkdir(parents=True, exist_ok=True)

            listeners = {
                11: {"command": "node", "ports": {7000, 5173}},
                22: {"command": "node", "ports": {3000}},
            }
            cwd_map = {11: project_root, 22: nested}

            result = detect_port_for_project(
                str(project_root),
                ProjectType.NEXTJS,
                bridge_port=8080,
                listeners=listeners,
                get_cwd=lambda pid: cwd_map.get(pid),
            )

        self.assertEqual(result, 5173)

    def test_detect_port_for_project_filters_bridge_port(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir).resolve()
            listeners = {1: {"command": "node", "ports": {8080}}}

            result = detect_port_for_project(
                str(project_root),
                ProjectType.NEXTJS,
                bridge_port=8080,
                listeners=listeners,
                get_cwd=lambda _pid: project_root,
            )

        self.assertIsNone(result)

    def test_detect_port_for_project_skips_non_candidate_command(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir).resolve()
            listeners = {1: {"command": "python", "ports": {8000}}}

            result = detect_port_for_project(
                str(project_root),
                ProjectType.NEXTJS,
                bridge_port=9999,
                listeners=listeners,
                get_cwd=lambda _pid: project_root,
            )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
