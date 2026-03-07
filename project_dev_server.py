"""Helpers for inferring JS dev-server commands from project metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_package_json_file(package_json_path: Path) -> dict[str, Any]:
    """Safely read and parse package.json."""
    try:
        return json.loads(package_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def guess_js_runner_from_package_manager(package_manager: str) -> str:
    """Guess script runner from packageManager field."""
    if package_manager.startswith("pnpm"):
        return "pnpm"
    if package_manager.startswith("yarn"):
        return "yarn"
    if package_manager.startswith("bun"):
        return "bun"
    return "npm"


def build_js_script_command_for_runner(runner: str, script: str) -> str:
    """Build a script command for the detected JS package manager."""
    if runner == "pnpm":
        return f"pnpm {script}"
    if runner == "yarn":
        return f"yarn {script}"
    if runner == "bun":
        return f"bun run {script}"
    return f"npm run {script}"


def infer_default_dev_server_command_from_project(
    project_path: str,
    project_type: Any,
) -> str | None:
    """Infer a default dev-server command from project files."""
    path = Path(project_path)
    if not path.exists() or not path.is_dir():
        return None

    project_type_value = str(getattr(project_type, "value", project_type)).lower()
    if project_type_value == "flutter" and (path / "pubspec.yaml").exists():
        return None

    package_json_path = path / "package.json"
    if package_json_path.exists():
        package_data = load_package_json_file(package_json_path)
        scripts = package_data.get("scripts", {}) if isinstance(package_data, dict) else {}
        package_manager = (
            str(package_data.get("packageManager", "")).lower()
            if isinstance(package_data, dict)
            else ""
        )
        runner = guess_js_runner_from_package_manager(package_manager)

        if isinstance(scripts, dict):
            if "dev" in scripts:
                return build_js_script_command_for_runner(runner, "dev")
            if "start" in scripts:
                return build_js_script_command_for_runner(runner, "start")

    if project_type_value == "nextjs":
        return "npm run dev"

    return None
