"""Service wrappers for project Git operations."""

from typing import Any

from core.base_result import BaseRouteResult
from files.git_manager import GitResult, get_git_manager
from projects.project_manager import get_project_manager

GitActionResult = BaseRouteResult


def _project_path(name: str) -> str | None:
    project = get_project_manager().get_project(name)
    path = project.get("path") if project else None
    return path if isinstance(path, str) and path else None


def _missing_project(name: str) -> GitActionResult:
    return GitActionResult.error(404, f"Project '{name}' not found")


def _git_result_payload(result: GitResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "output": result.stdout if result.stdout else result.stderr,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "error": result.error,
    }


def _git_result_response(result: GitResult) -> GitActionResult:
    payload = _git_result_payload(result)
    if result.success:
        return GitActionResult.ok(payload)
    return GitActionResult.error(
        400,
        result.error or result.stderr or "Git command failed",
        **payload,
    )


async def get_git_status_for_current_server(name: str) -> GitActionResult:
    path = _project_path(name)
    if path is None:
        return _missing_project(name)
    manager = get_git_manager(path)
    if not await manager.is_git_repo():
        return GitActionResult.error(400, "Project is not a git repository")
    return GitActionResult.ok((await manager.get_status()).to_dict())


async def get_git_diff_for_current_server(
    name: str,
    *,
    staged: bool = False,
    file: str | None = None,
) -> GitActionResult:
    path = _project_path(name)
    if path is None:
        return _missing_project(name)
    diff = await get_git_manager(path).get_diff(staged=staged, file=file)
    return GitActionResult.ok({"diff": diff})


async def get_git_log_for_current_server(
    name: str,
    *,
    limit: int = 20,
    file: str | None = None,
) -> GitActionResult:
    path = _project_path(name)
    if path is None:
        return _missing_project(name)
    commits = await get_git_manager(path).get_log(limit=limit, file=file)
    return GitActionResult.ok({"commits": commits})


async def get_git_branches_for_current_server(name: str) -> GitActionResult:
    path = _project_path(name)
    if path is None:
        return _missing_project(name)
    return GitActionResult.ok(await get_git_manager(path).get_branches())


async def stage_git_file_for_current_server(name: str, path: str) -> GitActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    return _git_result_response(await get_git_manager(project_path).stage_file(path))


async def unstage_git_file_for_current_server(name: str, path: str) -> GitActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    return _git_result_response(await get_git_manager(project_path).unstage_file(path))


async def discard_git_changes_for_current_server(name: str, path: str) -> GitActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    return _git_result_response(await get_git_manager(project_path).discard_changes(path))


async def commit_git_changes_for_current_server(name: str, message: str) -> GitActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    return _git_result_response(await get_git_manager(project_path).commit(message))


async def push_git_for_current_server(
    name: str,
    *,
    remote: str = "origin",
    branch: str | None = None,
) -> GitActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    return _git_result_response(
        await get_git_manager(project_path).push(remote=remote, branch=branch)
    )


async def pull_git_for_current_server(
    name: str,
    *,
    remote: str = "origin",
    branch: str | None = None,
) -> GitActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    return _git_result_response(
        await get_git_manager(project_path).pull(remote=remote, branch=branch)
    )


async def checkout_git_for_current_server(
    name: str,
    *,
    branch: str,
    create: bool = False,
) -> GitActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    return _git_result_response(
        await get_git_manager(project_path).checkout(branch, create=create)
    )

