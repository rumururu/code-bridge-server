"""Async job that puts Chromium on the machine hosting the server.

The product already *knew* the browser runtime was missing — a readiness chip
on the agents page, a `browser_runtime_unavailable` warning on the commit gate,
a copy-paste command in the fix modal. What it could not do was fix it, so the
only route back to a working `browser_action` step was a terminal on the host.
This is that route without the terminal.

Shape borrowed from `llm_provider_install_jobs.py` (same bounded in-memory
manager, same serialize/cancel contract) with three deliberate differences:

*   **One job at a time.** Asking twice returns the job already running rather
    than starting a second 450MB download over the top of the first.
*   **The verdict comes from a re-probe, never from the exit code.** A
    `playwright install` that exits 0 without leaving an executable on disk is
    reported as a failure. Readiness must never turn green on the strength of
    an install having been *started* — or even finished.
*   **The opt-out is honoured here too.** `CODE_BRIDGE_BROWSER=0` is one switch
    for the installers, the sync script and this endpoint.

Nothing here runs at startup or during a run. A server with no browser runtime
starts normally and says what it is missing; this only moves when a person
asks it to.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.browser_action_adapter import (
    get_browser_runtime_readiness,
    reset_browser_readiness_cache,
)
from system.browser_runtime_setup import (
    CHROMIUM_DISK_MB,
    CHROMIUM_DOWNLOAD_MB,
    OPT_OUT_ENV,
    browser_runtime_opt_out,
    chromium_install_argv,
)

BROWSER_INSTALL_DISABLED = "browser_runtime_install_disabled"
BROWSER_INSTALL_TIMEOUT = "browser_runtime_install_timeout"
BROWSER_INSTALL_FAILED = "browser_runtime_install_failed"
BROWSER_RUNTIME_UNAVAILABLE = "browser_runtime_unavailable"

# A 200MB download on a slow line, then an extraction. The provider-CLI jobs
# allow 10 minutes; that is not enough here and a timeout would look like a
# failure while the download was still healthy.
INSTALL_TIMEOUT_SECONDS = 1800
TAIL_CHARS = 4000

_ACTIVE_STATUSES = frozenset({"queued", "running"})
_FINISHED_STATUSES = frozenset({"completed", "failed", "cancelled"})


class BrowserRuntimeInstallDisabled(RuntimeError):
    """Raised when the operator has opted this machine out of Chromium."""


@dataclass
class BrowserRuntimeInstallJob:
    job_id: str
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
    readiness: dict[str, Any] | None = None
    cancel_requested: bool = False
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)


class BrowserRuntimeInstallJobManager:
    """Bounded in-memory manager for the Chromium install job."""

    def __init__(
        self,
        *,
        max_jobs: int = 20,
        ttl_seconds: int = 3600,
        timeout_seconds: int = INSTALL_TIMEOUT_SECONDS,
    ) -> None:
        self._jobs: dict[str, BrowserRuntimeInstallJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._max_jobs = max_jobs
        self._ttl_seconds = ttl_seconds
        self._timeout_seconds = timeout_seconds

    async def start_install(self) -> BrowserRuntimeInstallJob:
        if browser_runtime_opt_out():
            raise BrowserRuntimeInstallDisabled(
                f"Browser runtime installs are switched off on this machine ({OPT_OUT_ENV}=0)."
            )

        async with self._lock:
            self._purge_locked(time.time())
            active = self._active_job_locked()
            if active is not None:
                # Two clicks must not become two downloads.
                return active

            job = BrowserRuntimeInstallJob(
                job_id=uuid.uuid4().hex[:12],
                command=chromium_install_argv(),
            )
            self._jobs[job.job_id] = job
            self._tasks[job.job_id] = asyncio.create_task(self._run(job))
            return job

    def get_job(self, job_id: str) -> BrowserRuntimeInstallJob | None:
        self._purge(time.time())
        return self._jobs.get(job_id)

    def active_job(self) -> BrowserRuntimeInstallJob | None:
        return self._active_job_locked()

    async def cancel_job(self, job_id: str) -> BrowserRuntimeInstallJob | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in _FINISHED_STATUSES:
            return job

        job.cancel_requested = True
        task = self._tasks.get(job_id)
        if job.process and job.process.returncode is None:
            job.process.terminate()
        if task and not task.done():
            task.cancel()
        job.status = "cancelled"
        # A cancelled download leaves nothing usable behind, and saying
        # "installed: unknown" here would let the UI guess in the green
        # direction. It did not install.
        job.installed = False
        job.updated_at = time.time()
        return job

    def serialize(self, job: BrowserRuntimeInstallJob) -> dict[str, Any]:
        output = "\n".join(
            part for part in (job.stdout_tail.strip(), job.stderr_tail.strip()) if part
        )
        return {
            "job_id": job.job_id,
            "command": job.command,
            "status": job.status,
            "finished": job.status in _FINISHED_STATUSES,
            "returncode": job.returncode,
            "installed": job.installed,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "stdout_tail": job.stdout_tail,
            "stderr_tail": job.stderr_tail,
            "output": output[-TAIL_CHARS:],
            "readiness": job.readiness,
            "download_mb": CHROMIUM_DOWNLOAD_MB,
            "disk_mb": CHROMIUM_DISK_MB,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    async def _run(self, job: BrowserRuntimeInstallJob) -> None:
        job.status = "running"
        job.updated_at = time.time()

        try:
            job.process = await asyncio.create_subprocess_exec(
                *job.command,
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
            await self._fail(
                job,
                BROWSER_INSTALL_TIMEOUT,
                f"Chromium download timed out after {self._timeout_seconds} seconds",
            )
            return
        except FileNotFoundError as exc:
            await self._fail(job, BROWSER_INSTALL_FAILED, str(exc))
            return
        except asyncio.CancelledError:
            await self._terminate_process(job)
            if job.process is not None:
                job.returncode = job.process.returncode
            job.status = "cancelled"
            job.installed = False
            job.updated_at = time.time()
            return
        except OSError as exc:
            await self._fail(job, BROWSER_INSTALL_FAILED, str(exc))
            return

        job.returncode = job.process.returncode
        job.stdout_tail = stdout.decode(errors="replace")[-TAIL_CHARS:]
        job.stderr_tail = stderr.decode(errors="replace")[-TAIL_CHARS:]

        if job.returncode != 0:
            message = self._tail_message(job) or (
                f"Chromium download failed with exit code {job.returncode}"
            )
            await self._fail(job, BROWSER_INSTALL_FAILED, message)
            return

        # The exit code is not the answer. Ask the same probe the chip and the
        # commit gate ask, forced past its cache, and let the executable on
        # disk decide.
        readiness = await self._refresh_readiness()
        job.readiness = readiness
        job.installed = bool(readiness.get("ready"))
        if not job.installed:
            await self._fail(
                job,
                BROWSER_RUNTIME_UNAVAILABLE,
                str(readiness.get("message") or "Chromium is still not available after install"),
                readiness=readiness,
            )
            return

        job.status = "completed"
        job.error_code = None
        job.error_message = None
        job.updated_at = time.time()

    async def _refresh_readiness(self) -> dict[str, Any]:
        try:
            reset_browser_readiness_cache()
            return await get_browser_runtime_readiness(force_refresh=True)
        except Exception as exc:  # noqa: BLE001 - an unreadable probe is "not ready"
            return {
                "ready": False,
                "message": f"Readiness probe failed after install: {exc}",
                "diagnostics": [],
            }

    async def _terminate_process(self, job: BrowserRuntimeInstallJob) -> None:
        proc = job.process
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def _fail(
        self,
        job: BrowserRuntimeInstallJob,
        error_code: str,
        error_message: str,
        *,
        readiness: dict[str, Any] | None = None,
    ) -> None:
        job.status = "failed"
        job.error_code = error_code
        job.error_message = error_message
        job.installed = False
        if readiness is not None:
            job.readiness = readiness
        elif job.readiness is None:
            job.readiness = await self._refresh_readiness()
        job.updated_at = time.time()

    def _tail_message(self, job: BrowserRuntimeInstallJob) -> str:
        combined = "\n".join(
            part for part in (job.stderr_tail.strip(), job.stdout_tail.strip()) if part
        )
        return combined[-500:]

    def _active_job_locked(self) -> BrowserRuntimeInstallJob | None:
        for job in self._jobs.values():
            if job.status in _ACTIVE_STATUSES:
                return job
        return None

    def _purge(self, now: float) -> None:
        if self._lock.locked():
            return
        self._purge_locked(now)

    def _purge_locked(self, now: float) -> None:
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in _FINISHED_STATUSES and now - job.updated_at > self._ttl_seconds
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
            self._tasks.pop(job_id, None)

        if len(self._jobs) <= self._max_jobs:
            return

        removable = sorted(
            (job for job in self._jobs.values() if job.status in _FINISHED_STATUSES),
            key=lambda item: item.updated_at,
        )
        for job in removable:
            if len(self._jobs) <= self._max_jobs:
                break
            self._jobs.pop(job.job_id, None)
            self._tasks.pop(job.job_id, None)


_manager = BrowserRuntimeInstallJobManager()


def get_browser_runtime_install_job_manager() -> BrowserRuntimeInstallJobManager:
    return _manager
