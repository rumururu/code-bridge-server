import json
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_dev_server import (
    build_js_script_command_for_runner,
    guess_js_runner_from_package_manager,
    infer_default_dev_server_command_from_project,
    load_package_json_file,
)
from project_models import ProjectType


class ProjectDevServerHelpersTest(unittest.TestCase):
    def test_load_package_json_file_returns_parsed_dict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            package_json = Path(tmp_dir) / "package.json"
            package_json.write_text(json.dumps({"scripts": {"dev": "vite"}}), encoding="utf-8")

            result = load_package_json_file(package_json)

        self.assertEqual(result, {"scripts": {"dev": "vite"}})

    def test_load_package_json_file_returns_empty_dict_on_parse_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            package_json = Path(tmp_dir) / "package.json"
            package_json.write_text("{invalid}", encoding="utf-8")

            result = load_package_json_file(package_json)

        self.assertEqual(result, {})

    def test_guess_js_runner_from_package_manager(self):
        self.assertEqual(guess_js_runner_from_package_manager("pnpm@9.0.0"), "pnpm")
        self.assertEqual(guess_js_runner_from_package_manager("yarn@4.0.0"), "yarn")
        self.assertEqual(guess_js_runner_from_package_manager("bun@1.0.0"), "bun")
        self.assertEqual(guess_js_runner_from_package_manager("npm@10.0.0"), "npm")
        self.assertEqual(guess_js_runner_from_package_manager(""), "npm")

    def test_build_js_script_command_for_runner(self):
        self.assertEqual(build_js_script_command_for_runner("pnpm", "dev"), "pnpm dev")
        self.assertEqual(build_js_script_command_for_runner("yarn", "start"), "yarn start")
        self.assertEqual(build_js_script_command_for_runner("bun", "dev"), "bun run dev")
        self.assertEqual(build_js_script_command_for_runner("npm", "dev"), "npm run dev")

    def test_infer_default_dev_server_command_returns_none_for_missing_path(self):
        result = infer_default_dev_server_command_from_project(
            "/tmp/this-path-should-not-exist-for-tests",
            ProjectType.NEXTJS,
        )

        self.assertIsNone(result)

    def test_infer_default_dev_server_command_returns_none_for_flutter_project(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir)
            (project_path / "pubspec.yaml").write_text("name: demo", encoding="utf-8")

            result = infer_default_dev_server_command_from_project(
                str(project_path),
                ProjectType.FLUTTER,
            )

        self.assertIsNone(result)

    def test_infer_default_dev_server_command_prefers_dev_script(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir)
            (project_path / "package.json").write_text(
                json.dumps(
                    {
                        "packageManager": "pnpm@9.0.0",
                        "scripts": {
                            "dev": "next dev",
                            "start": "next start",
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = infer_default_dev_server_command_from_project(
                str(project_path),
                ProjectType.NEXTJS,
            )

        self.assertEqual(result, "pnpm dev")

    def test_infer_default_dev_server_command_uses_start_when_dev_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir)
            (project_path / "package.json").write_text(
                json.dumps(
                    {
                        "packageManager": "yarn@4.0.0",
                        "scripts": {"start": "vite preview"},
                    }
                ),
                encoding="utf-8",
            )

            result = infer_default_dev_server_command_from_project(
                str(project_path),
                ProjectType.NEXTJS,
            )

        self.assertEqual(result, "yarn start")

    def test_infer_default_dev_server_command_falls_back_for_nextjs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir)

            result = infer_default_dev_server_command_from_project(
                str(project_path),
                ProjectType.NEXTJS,
            )

        self.assertEqual(result, "npm run dev")


if __name__ == "__main__":
    unittest.main()
