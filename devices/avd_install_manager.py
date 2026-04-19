"""AVD install manager.

Owns subprocess orchestration for:
  1. Accepting Android SDK licenses (``sdkmanager --licenses``)
  2. Downloading a system image (``sdkmanager "system-images;...``)
  3. Creating an AVD (``avdmanager create avd``)

Jobs are identified by short hex ids and tracked in-memory. Clients subscribe
via [subscribe] to receive a stream of progress events so they can render a
realtime progress bar and phase label.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class InstallPhase(str, Enum):
    QUEUED = "queued"
    LICENSES = "licenses"
    DOWNLOAD = "download"
    CREATE_AVD = "create_avd"
    DONE = "done"
    ERROR = "error"


@dataclass
class InstallJob:
    job_id: str
    avd_name: str
    image: str
    device_profile: str
    phase: InstallPhase = InstallPhase.QUEUED
    percent: float = 0.0
    message: str = ""
    error: Optional[str] = None
    finished: bool = False
    subscribers: list[asyncio.Queue] = field(default_factory=list)


# Default install target. Chosen for Apple Silicon compatibility and a
# modern-but-stable API level. Bumping this is a one-line change.
DEFAULT_IMAGE = "system-images;android-35;google_apis;arm64-v8a"
DEFAULT_DEVICE_PROFILE = "pixel_7"
DEFAULT_AVD_NAME = "code_bridge_pixel"


class AvdInstallManager:
    def __init__(self) -> None:
        self._jobs: dict[str, InstallJob] = {}
        self._lock = asyncio.Lock()
        self._sdk_root: Optional[str] = self._resolve_sdk_root()

    # ---------- path detection ----------

    def _resolve_sdk_root(self) -> Optional[str]:
        candidates = [
            os.environ.get("ANDROID_HOME"),
            os.environ.get("ANDROID_SDK_ROOT"),
            os.path.expanduser("~/Library/Android/sdk"),
            os.path.expanduser("~/Android/Sdk"),
        ]
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate
        return None

    def _tool_path(self, name: str) -> Optional[str]:
        which = shutil.which(name)
        if which:
            return which
        if self._sdk_root is None:
            return None
        guesses = [
            os.path.join(self._sdk_root, "cmdline-tools", "latest", "bin", name),
            os.path.join(self._sdk_root, "tools", "bin", name),
            os.path.join(self._sdk_root, "emulator", name),
            os.path.join(self._sdk_root, "platform-tools", name),
        ]
        for guess in guesses:
            if os.path.exists(guess):
                return guess
        return None

    def environment_status(self) -> dict[str, Optional[str]]:
        """Report SDK + tool paths so the client can pre-flight the machine."""
        return {
            "android_sdk_root": self._sdk_root,
            "sdkmanager": self._tool_path("sdkmanager"),
            "avdmanager": self._tool_path("avdmanager"),
            "emulator": self._tool_path("emulator"),
        }

    # ---------- job lifecycle ----------

    async def start_install(
        self,
        *,
        avd_name: str = DEFAULT_AVD_NAME,
        image: str = DEFAULT_IMAGE,
        device_profile: str = DEFAULT_DEVICE_PROFILE,
    ) -> InstallJob:
        job_id = uuid.uuid4().hex[:12]
        job = InstallJob(
            job_id=job_id,
            avd_name=avd_name,
            image=image,
            device_profile=device_profile,
        )
        async with self._lock:
            self._jobs[job_id] = job
        asyncio.create_task(self._run(job))
        return job

    def get_job(self, job_id: str) -> Optional[InstallJob]:
        return self._jobs.get(job_id)

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.append(queue)
        await queue.put(self.serialize(job))
        if job.finished:
            await queue.put(None)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        if queue in job.subscribers:
            job.subscribers.remove(queue)

    def serialize(self, job: InstallJob) -> dict:
        return {
            "job_id": job.job_id,
            "phase": job.phase.value,
            "percent": round(job.percent, 1),
            "message": job.message,
            "avd_name": job.avd_name,
            "image": job.image,
            "error": job.error,
            "finished": job.finished,
        }

    async def _emit(self, job: InstallJob) -> None:
        payload = self.serialize(job)
        for queue in list(job.subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("avd install subscriber queue full job=%s", job.job_id)

    async def _finish(self, job: InstallJob) -> None:
        job.finished = True
        await self._emit(job)
        for queue in list(job.subscribers):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    # ---------- install pipeline ----------

    async def _run(self, job: InstallJob) -> None:
        try:
            sdkmanager = self._tool_path("sdkmanager")
            avdmanager = self._tool_path("avdmanager")
            if sdkmanager is None or avdmanager is None:
                job.phase = InstallPhase.ERROR
                job.error = (
                    "Android SDK cmdline-tools not found. "
                    "Set ANDROID_HOME or install to ~/Library/Android/sdk."
                )
                await self._emit(job)
                return

            # 1) Accept licenses. Harmless if all are already accepted.
            job.phase = InstallPhase.LICENSES
            job.message = "Accepting Android SDK licenses"
            await self._emit(job)
            license_ok, license_err = await self._run_licenses(sdkmanager)
            if not license_ok:
                job.phase = InstallPhase.ERROR
                job.error = license_err or "Failed to accept licenses"
                await self._emit(job)
                return

            # 2) Download system image with percent parsing.
            job.phase = InstallPhase.DOWNLOAD
            job.percent = 0.0
            job.message = f"Downloading {job.image}"
            await self._emit(job)
            dl_ok, dl_err = await self._run_download(sdkmanager, job)
            if not dl_ok:
                job.phase = InstallPhase.ERROR
                job.error = dl_err or "Failed to download system image"
                await self._emit(job)
                return

            # 3) Create AVD.
            job.phase = InstallPhase.CREATE_AVD
            job.percent = 100.0
            job.message = f"Creating AVD: {job.avd_name}"
            await self._emit(job)
            create_ok, create_err = await self._run_create_avd(avdmanager, job)
            if not create_ok:
                job.phase = InstallPhase.ERROR
                job.error = create_err or "Failed to create AVD"
                await self._emit(job)
                return

            job.phase = InstallPhase.DONE
            job.message = f"Emulator ready: {job.avd_name}"
            await self._emit(job)
        except Exception as exc:  # noqa: BLE001
            logger.exception("avd install crashed job=%s", job.job_id)
            job.phase = InstallPhase.ERROR
            job.error = f"Unexpected error: {exc}"
            await self._emit(job)
        finally:
            await self._finish(job)

    async def _run_licenses(self, sdkmanager: str) -> tuple[bool, Optional[str]]:
        try:
            proc = await asyncio.create_subprocess_exec(
                sdkmanager,
                "--licenses",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            # sdkmanager prompts once per unaccepted license. Feed a big "y\n"
            # buffer rather than streaming to avoid racing on prompt output.
            assert proc.stdin is not None
            proc.stdin.write(("y\n" * 30).encode())
            try:
                await proc.stdin.drain()
            except BrokenPipeError:
                pass
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                tail = stdout.decode(errors="ignore")[-400:]
                return False, f"sdkmanager --licenses failed (exit {proc.returncode}): {tail}"
            return True, None
        except FileNotFoundError:
            return False, "sdkmanager executable not found"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    # sdkmanager shows percentages with a visually-noisy progress bar like:
    #   [=========                              ] 24% Downloading system-images...
    _PERCENT_RE = re.compile(r"(\d+)%")

    async def _run_download(
        self,
        sdkmanager: str,
        job: InstallJob,
    ) -> tuple[bool, Optional[str]]:
        try:
            proc = await asyncio.create_subprocess_exec(
                sdkmanager,
                "--install",
                job.image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert proc.stdout is not None
            last_tail = ""
            last_emit_percent = -1.0
            buffer = bytearray()
            # sdkmanager refreshes its progress bar with \r, not \n, so
            # readline() blocks forever waiting for a newline that never
            # arrives. Read fixed-size chunks and split on either separator.
            while True:
                chunk = await proc.stdout.read(256)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    nl = buffer.find(b"\n")
                    cr = buffer.find(b"\r")
                    if nl == -1 and cr == -1:
                        break
                    if nl == -1:
                        idx = cr
                    elif cr == -1:
                        idx = nl
                    else:
                        idx = min(nl, cr)
                    line = bytes(buffer[:idx]).decode(errors="ignore").strip()
                    del buffer[: idx + 1]
                    if not line:
                        continue
                    last_tail = line[-200:]
                    match = self._PERCENT_RE.search(line)
                    if match:
                        pct = float(match.group(1))
                        if pct - last_emit_percent >= 1.0 or pct >= 100:
                            job.percent = pct
                            job.message = f"Downloading {job.image} ({pct:.0f}%)"
                            await self._emit(job)
                            last_emit_percent = pct
                    else:
                        job.message = line[:160]
                        await self._emit(job)
            await proc.wait()
            if proc.returncode != 0:
                return False, f"sdkmanager download failed (exit {proc.returncode}): {last_tail}"
            return True, None
        except FileNotFoundError:
            return False, "sdkmanager executable not found"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    async def _run_create_avd(
        self,
        avdmanager: str,
        job: InstallJob,
    ) -> tuple[bool, Optional[str]]:
        try:
            # -f overwrites if an AVD with the same name exists; this keeps the
            # operation idempotent for retries.
            proc = await asyncio.create_subprocess_exec(
                avdmanager,
                "create",
                "avd",
                "-n",
                job.avd_name,
                "-k",
                job.image,
                "-d",
                job.device_profile,
                "--force",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert proc.stdin is not None
            # avdmanager asks about custom hardware profile — accept defaults.
            proc.stdin.write(b"no\n")
            try:
                await proc.stdin.drain()
            except BrokenPipeError:
                pass
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                tail = stdout.decode(errors="ignore")[-400:]
                return False, f"avdmanager failed (exit {proc.returncode}): {tail}"
            return True, None
        except FileNotFoundError:
            return False, "avdmanager executable not found"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)


_manager: Optional[AvdInstallManager] = None


def get_avd_install_manager() -> AvdInstallManager:
    global _manager
    if _manager is None:
        _manager = AvdInstallManager()
    return _manager
