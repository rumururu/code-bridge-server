"""Async jobs for installing LLM provider CLIs."""

from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from llm.llm_settings import (
    check_llm_provider_cli_available,
    clear_llm_provider_cli_cache,
    get_llm_options_snapshot,
    resolve_llm_provider_install,
)

PROVIDER_CLI_MISSING = "provider_cli_missing"
PROVIDER_INSTALL_TIMEOUT = "provider_install_timeout"
PROVIDER_INSTALL_FAILED = "provider_install_failed"
PROVIDER_CLI_UNAVAILABLE = "provider_cli_unavailable"
UNKNOWN_PROVIDER = "unknown_provider"
INVALID_INSTALL_METHOD = "invalid_install_method"

INSTALL_TIMEOUT_SECONDS = 600
TAIL_CHARS = 4000


@dataclass
class LlmProviderInstallJob:
    job_id: str
    provider_id: str
    method: str
    provider_name: str
    provider_command: str
    command: list[str]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    returncode: int | None = None
    installed: bool | None = None
    error_code: str | None = None
    error_message: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    options: dict[str, Any] | None = None
    cancel_requested: bool = False
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)


class LlmProviderInstallJobManager:
    """Bounded in-memory manager for provider install jobs."""

    def __init__(
        self,
        *,
        max_jobs: int = 100,
        ttl_seconds: int = 3600,
        timeout_seconds: int = INSTALL_TIMEOUT_SECONDS,
    ) -> None:
        self._jobs: dict[str, LlmProviderInstallJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._max_jobs = max_jobs
        self._ttl_seconds = ttl_seconds
        self._timeout_seconds = timeout_seconds

    async def start_install(self, provider_id: str, method: str) -> LlmProviderInstallJob:
        spec = resolve_llm_provider_install(provider_id, method)
        job = LlmProviderInstallJob(
            job_id=uuid.uuid4().hex[:12],
            provider_id=spec["provider_id"],
            method=spec["method"],
            provider_name=spec["provider_name"],
            provider_command=spec["provider_command"],
            command=spec["command"],
        )
        async with self._lock:
            self._purge_locked(time.time())
            self._jobs[job.job_id] = job
            self._tasks[job.job_id] = asyncio.create_task(self._run(job))
        return job

    def get_job(self, job_id: str) -> LlmProviderInstallJob | None:
        self._purge(time.time())
        return self._jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> LlmProviderInstallJob | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in {"completed", "failed", "cancelled"}:
            return job

        job.cancel_requested = True
        task = self._tasks.get(job_id)
        if job.process and job.process.returncode is None:
            job.process.terminate()
        if task and not task.done():
            task.cancel()
        job.status = "cancelled"
        job.updated_at = time.time()
        return job

    def serialize(self, job: LlmProviderInstallJob) -> dict[str, Any]:
        finished = job.status in {"completed", "failed", "cancelled"}
        output = "\n".join(part for part in (job.stdout_tail.strip(), job.stderr_tail.strip()) if part)
        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "provider_id": job.provider_id,
            "provider_name": job.provider_name,
            "method": job.method,
            "command": job.command,
            "status": job.status,
            "finished": finished,
            "returncode": job.returncode,
            "installed": job.installed,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "stdout_tail": job.stdout_tail,
            "stderr_tail": job.stderr_tail,
            "output": output[-TAIL_CHARS:],
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        if job.options is not None:
            payload["options"] = job.options
        return payload

    async def _run(self, job: LlmProviderInstallJob) -> None:
        job.status = "running"
        job.updated_at = time.time()
        executable = job.command[0]
        resolved = shutil.which(executable)
        if not resolved:
            self._fail(job, PROVIDER_CLI_MISSING, f"{executable} is not installed")
            return

        cmd = [resolved, *job.command[1:]]
        try:
            job.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                job.process.communicate(),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            await self._terminate_process(job)
            if job.process is not None:
                job.returncode = job.process.returncode
            self._fail(
                job,
                PROVIDER_INSTALL_TIMEOUT,
                f"Install timed out after {self._timeout_seconds} seconds",
            )
            return
        except FileNotFoundError:
            self._fail(job, PROVIDER_CLI_MISSING, f"{executable} is not installed")
            return
        except asyncio.CancelledError:
            await self._terminate_process(job)
            if job.process is not None:
                job.returncode = job.process.returncode
            job.status = "cancelled"
            job.updated_at = time.time()
            return
        except OSError as exc:
            self._fail(job, PROVIDER_INSTALL_FAILED, str(exc))
            return

        job.returncode = job.process.returncode
        job.stdout_tail = stdout.decode(errors="replace")[-TAIL_CHARS:]
        job.stderr_tail = stderr.decode(errors="replace")[-TAIL_CHARS:]
        if job.returncode != 0:
            message = self._tail_message(job) or f"Install failed with exit code {job.returncode}"
            self._fail(job, PROVIDER_INSTALL_FAILED, message)
            return

        clear_llm_provider_cli_cache(job.provider_id)
        installed, availability_error = check_llm_provider_cli_available(job.provider_id)
        job.installed = installed
        if not installed:
            self._fail(
                job,
                PROVIDER_CLI_UNAVAILABLE,
                availability_error or f"{job.provider_command} CLI is not available after install",
            )
            return

        job.status = "completed"
        job.error_code = None
        job.error_message = None
        job.options = get_llm_options_snapshot()
        job.updated_at = time.time()

    async def _terminate_process(self, job: LlmProviderInstallJob) -> None:
        proc = job.process
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    def _fail(self, job: LlmProviderInstallJob, error_code: str, error_message: str) -> None:
        job.status = "failed"
        job.error_code = error_code
        job.error_message = error_message
        if job.installed is None:
            job.installed = False
        try:
            job.options = get_llm_options_snapshot()
        except Exception:  # noqa: BLE001
            job.options = None
        job.updated_at = time.time()

    def _tail_message(self, job: LlmProviderInstallJob) -> str:
        combined = "\n".join(part for part in (job.stderr_tail.strip(), job.stdout_tail.strip()) if part)
        return combined[-500:]

    def _purge(self, now: float) -> None:
        if self._lock.locked():
            return
        self._purge_locked(now)

    def _purge_locked(self, now: float) -> None:
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {"completed", "failed", "cancelled"}
            and now - job.updated_at > self._ttl_seconds
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
            self._tasks.pop(job_id, None)

        if len(self._jobs) <= self._max_jobs:
            return

        removable = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status in {"completed", "failed", "cancelled"}
            ),
            key=lambda item: item.updated_at,
        )
        for job in removable:
            if len(self._jobs) <= self._max_jobs:
                break
            self._jobs.pop(job.job_id, None)
            self._tasks.pop(job.job_id, None)


_manager = LlmProviderInstallJobManager()


def get_llm_provider_install_job_manager() -> LlmProviderInstallJobManager:
    return _manager
