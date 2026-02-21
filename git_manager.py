"""Git operations manager for project repositories."""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Maximum output size (bytes)
MAX_OUTPUT_SIZE = 512 * 1024  # 512KB


@dataclass
class GitResult:
    """Result of a git operation."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and self.error is None


@dataclass
class GitStatus:
    """Git repository status."""

    branch: str = ""
    is_clean: bool = True
    ahead: int = 0
    behind: int = 0
    staged: list = None
    modified: list = None
    untracked: list = None
    deleted: list = None

    def __post_init__(self):
        if self.staged is None:
            self.staged = []
        if self.modified is None:
            self.modified = []
        if self.untracked is None:
            self.untracked = []
        if self.deleted is None:
            self.deleted = []

    def to_dict(self) -> dict:
        return {
            "branch": self.branch,
            "is_clean": self.is_clean,
            "ahead": self.ahead,
            "behind": self.behind,
            "staged": self.staged,
            "modified": self.modified,
            "untracked": self.untracked,
            "deleted": self.deleted,
        }


class GitManager:
    """Manages git operations for a project."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)

    async def _run_git(self, *args, timeout: int = 30) -> GitResult:
        """Run a git command."""
        cmd = ["git"] + list(args)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Truncate if too large
            if len(stdout_str) > MAX_OUTPUT_SIZE:
                stdout_str = stdout_str[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
            if len(stderr_str) > MAX_OUTPUT_SIZE:
                stderr_str = stderr_str[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"

            return GitResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=process.returncode or 0,
            )

        except asyncio.TimeoutError:
            return GitResult(
                error=f"Git command timed out after {timeout}s",
                exit_code=124,
            )
        except Exception as e:
            return GitResult(
                error=str(e),
                exit_code=1,
            )

    async def is_git_repo(self) -> bool:
        """Check if the project is a git repository."""
        result = await self._run_git("rev-parse", "--git-dir")
        return result.success

    async def get_status(self) -> GitStatus:
        """Get repository status."""
        status = GitStatus()

        # Get current branch
        branch_result = await self._run_git("branch", "--show-current")
        if branch_result.success:
            status.branch = branch_result.stdout.strip()

        # Get ahead/behind
        tracking_result = await self._run_git(
            "rev-list", "--left-right", "--count", f"{status.branch}@{{upstream}}...HEAD"
        )
        if tracking_result.success:
            parts = tracking_result.stdout.strip().split()
            if len(parts) == 2:
                status.behind = int(parts[0])
                status.ahead = int(parts[1])

        # Get file status
        status_result = await self._run_git("status", "--porcelain")
        if status_result.success:
            for line in status_result.stdout.strip().split("\n"):
                if not line:
                    continue

                index_status = line[0] if len(line) > 0 else " "
                work_status = line[1] if len(line) > 1 else " "
                filename = line[3:] if len(line) > 3 else ""

                # Staged changes
                if index_status in "MADRC":
                    status.staged.append(filename)

                # Modified in working tree
                if work_status == "M":
                    status.modified.append(filename)

                # Deleted
                if work_status == "D" or index_status == "D":
                    status.deleted.append(filename)

                # Untracked
                if index_status == "?" and work_status == "?":
                    status.untracked.append(filename)

        status.is_clean = (
            len(status.staged) == 0 and
            len(status.modified) == 0 and
            len(status.untracked) == 0 and
            len(status.deleted) == 0
        )

        return status

    async def get_diff(self, staged: bool = False, file: Optional[str] = None) -> str:
        """Get diff of changes."""
        args = ["diff"]
        if staged:
            args.append("--cached")
        if file:
            args.append("--")
            args.append(file)

        result = await self._run_git(*args, timeout=60)
        if result.success:
            return result.stdout
        return result.error or result.stderr

    async def get_log(self, limit: int = 20, file: Optional[str] = None) -> list:
        """Get commit log."""
        args = [
            "log",
            f"-{limit}",
            "--pretty=format:%H|%h|%s|%an|%ae|%ai",
        ]
        if file:
            args.append("--")
            args.append(file)

        result = await self._run_git(*args, timeout=30)
        if not result.success:
            return []

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 6:
                commits.append({
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "message": parts[2],
                    "author": parts[3],
                    "email": parts[4],
                    "date": parts[5],
                })

        return commits

    async def get_branches(self) -> dict:
        """Get list of branches."""
        result = await self._run_git(
            "branch", "-a", "--format=%(refname:short)|%(objectname:short)|%(upstream:short)"
        )

        if not result.success:
            return {"local": [], "remote": [], "current": ""}

        local = []
        remote = []
        current = ""

        # Get current branch
        current_result = await self._run_git("branch", "--show-current")
        if current_result.success:
            current = current_result.stdout.strip()

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            name = parts[0]
            short_hash = parts[1] if len(parts) > 1 else ""
            upstream = parts[2] if len(parts) > 2 else ""

            if name.startswith("remotes/") or name.startswith("origin/"):
                remote.append({
                    "name": name.replace("remotes/", ""),
                    "hash": short_hash,
                })
            else:
                local.append({
                    "name": name,
                    "hash": short_hash,
                    "upstream": upstream,
                    "is_current": name == current,
                })

        return {"local": local, "remote": remote, "current": current}

    async def stage_file(self, path: str) -> GitResult:
        """Stage a file for commit."""
        return await self._run_git("add", path)

    async def unstage_file(self, path: str) -> GitResult:
        """Unstage a file."""
        return await self._run_git("reset", "HEAD", "--", path)

    async def discard_changes(self, path: str) -> GitResult:
        """Discard changes to a file."""
        return await self._run_git("checkout", "--", path)

    async def commit(self, message: str) -> GitResult:
        """Create a commit."""
        return await self._run_git("commit", "-m", message)

    async def push(self, remote: str = "origin", branch: Optional[str] = None) -> GitResult:
        """Push to remote."""
        args = ["push", remote]
        if branch:
            args.append(branch)
        return await self._run_git(*args, timeout=120)

    async def pull(self, remote: str = "origin", branch: Optional[str] = None) -> GitResult:
        """Pull from remote."""
        args = ["pull", remote]
        if branch:
            args.append(branch)
        return await self._run_git(*args, timeout=120)

    async def fetch(self, remote: str = "origin") -> GitResult:
        """Fetch from remote."""
        return await self._run_git("fetch", remote, timeout=60)

    async def checkout(self, branch: str, create: bool = False) -> GitResult:
        """Checkout a branch."""
        if create:
            return await self._run_git("checkout", "-b", branch)
        return await self._run_git("checkout", branch)

    async def create_branch(self, name: str, start_point: Optional[str] = None) -> GitResult:
        """Create a new branch."""
        args = ["branch", name]
        if start_point:
            args.append(start_point)
        return await self._run_git(*args)

    async def delete_branch(self, name: str, force: bool = False) -> GitResult:
        """Delete a branch."""
        flag = "-D" if force else "-d"
        return await self._run_git("branch", flag, name)


# Cache for git managers
_git_managers: dict[str, GitManager] = {}


def get_git_manager(project_path: str) -> GitManager:
    """Get or create a GitManager for a project."""
    if project_path not in _git_managers:
        _git_managers[project_path] = GitManager(project_path)
    return _git_managers[project_path]
