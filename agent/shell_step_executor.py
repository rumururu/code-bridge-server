"""Run a registered script as a workflow step.

This is the deterministic half of a workflow: no model call, no tokens, no
permission prompt — the script was vetted when it was registered. It exists so
work that is already scripted (device farm cycles, backups, sync jobs) can be
scheduled through Code Bridge without an LLM paraphrasing it every run.

The interesting part is the failure path. A script that fails hands its exit
code and output to the next step through the normal step-output evidence, so a
workflow can escalate — ``on_failure: goto_step: diagnose`` — to an LLM step
that looks at the device, works out what changed, and can edit the script
itself. That is the pattern the step type is shaped around.
"""

from __future__ import annotations

import asyncio
import subprocess
import shutil
import os
import time
from dataclasses import dataclass, field
from typing import Any

# Enough to see what happened without dumping a 50k-line log into a prompt.
_MAX_CAPTURED_CHARS = 8000


@dataclass
class ShellStepResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    error: str | None = None
    command: list[str] = field(default_factory=list)
    #: Other processes on this machine that were driving the same device when
    #: the script failed (AGENT_SELF_REPAIR_SPEC §3 — device_contention).
    contention: list[str] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None

    def to_output(self) -> dict[str, Any]:
        return {
            "status": "completed" if self.completed else "failed",
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            **({"error": self.error} if self.error else {}),
            **({"contention": self.contention} if self.contention else {}),
        }


def _tail(text: str) -> str:
    """Keep the end of the output — that is where the failure is."""
    if len(text) <= _MAX_CAPTURED_CHARS:
        return text
    kept = text[-_MAX_CAPTURED_CHARS:]
    return f"…[{len(text) - _MAX_CAPTURED_CHARS} chars truncated]…\n{kept}"


def build_command(script: dict[str, Any], extra_args: list[str] | None = None) -> list[str]:
    interpreter = str(script.get("interpreter") or "bash")
    path = str(script["path"])
    args = [str(item) for item in (script.get("default_args") or [])]
    args.extend(str(item) for item in (extra_args or []))
    if interpreter == "direct":
        return [path, *args]
    return [interpreter, path, *args]


def _connected_device_serials() -> set[str]:
    """Serials `adb devices` lists right now; empty when adb is not around."""
    adb = shutil.which("adb") or os.environ.get("CODEBRIDGE_ADB_PATH")
    if not adb:
        return set()
    try:
        out = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    serials = set()
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] in ("device", "offline", "unauthorized"):
            serials.add(parts[0])
    return serials


def _process_list() -> list[str]:
    try:
        return subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True, timeout=10, check=False).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return []


def device_contention(command: list[str], *, serials: set[str] | None = None, processes: list[str] | None = None) -> list[str]:
    """Other processes driving a device this command names, if any.

    A device cycle that hung for an hour on 2026-09-04 was diagnosed as "an
    unrecognised screen" when a second scheduler on the same Mac (a launchd
    job running the same script) was tapping the same phone. Nothing in the
    run record could show that; this is where it enters the record. Looked
    up only when a script has failed, and only for arguments that are
    connected adb serials, so a script naming no device costs nothing.
    """
    targets = [arg for arg in command[1:] if isinstance(arg, str) and 6 <= len(arg) <= 40 and arg.isalnum()]
    if not targets:
        return []
    known = _connected_device_serials() if serials is None else serials
    targets = [t for t in targets if t in known]
    if not targets:
        return []
    own = str(os.getpid())
    found: list[str] = []
    for line in (_process_list() if processes is None else processes):
        text = line.strip()
        if not text or text.split()[0] == own:
            continue
        if any(t in text for t in targets) and " ps " not in f" {text} " and "grep" not in text:
            found.append(text[:240])
    return found[:8]


class _TailBuffer:
    """The end of a stream, kept while it flows; ``_tail`` keeps the record's shape.

    Bounded at four bytes per captured character so a multi-byte tail still
    fills ``_MAX_CAPTURED_CHARS``; older bytes are dropped as they arrive,
    and ``dropped`` keeps the count so the truncation note stays honest.
    """

    _LIMIT = _MAX_CAPTURED_CHARS * 4

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._size = 0
        self.dropped = 0

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._chunks.append(chunk)
        self._size += len(chunk)
        while self._size > self._LIMIT and len(self._chunks) > 1:
            gone = self._chunks.pop(0)
            self._size -= len(gone)
            self.dropped += len(gone)

    def text(self) -> str:
        text = b"".join(self._chunks).decode("utf-8", "replace")
        if self.dropped and len(text) <= _MAX_CAPTURED_CHARS:
            return f"…[{self.dropped} bytes truncated]…\n{text}"
        return _tail(text)


async def _pump(stream: asyncio.StreamReader | None, into: _TailBuffer) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        into.write(chunk)


async def run_registered_script(
    script: dict[str, Any],
    *,
    extra_args: list[str] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ShellStepResult:
    """Execute ``script`` and capture its result.

    A timeout kills the process rather than letting it hold the task "active"
    forever — an unbounded script would wedge its own schedule.
    """
    command = build_command(script, extra_args)
    timeout = int(script.get("timeout_seconds") or 3600)
    started = time.monotonic()

    process_env = dict(os.environ)
    if env:
        process_env.update({str(k): str(v) for k, v in env.items()})

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or None,
            env=process_env,
        )
    except OSError as exc:
        return ShellStepResult(
            exit_code=None,
            stdout="",
            stderr="",
            duration_ms=int((time.monotonic() - started) * 1000),
            error=f"could not start script: {exc}",
            command=command,
        )

    # Both pipes are drained as they fill, into bounded tails, so a script
    # the timeout kills still leaves the lines it wrote. `communicate()` kept
    # everything in one buffer that was thrown away on a timeout: a device
    # cycle that hung after fifty minutes of logging was diagnosed from an
    # empty record — "no step log tail identifying which screen it stalled
    # on" — when the script had said exactly where it was.
    stdout_tail = _TailBuffer()
    stderr_tail = _TailBuffer()
    pumps = [
        asyncio.create_task(_pump(process.stdout, stdout_tail)),
        asyncio.create_task(_pump(process.stderr, stderr_tail)),
    ]
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        await asyncio.gather(*pumps, return_exceptions=True)
    except asyncio.TimeoutError:
        process.kill()
        # Reap the killed process so it does not linger as a zombie, and let
        # the pumps read what was left in the pipes.
        try:
            await process.wait()
        except Exception:
            pass
        await asyncio.gather(*pumps, return_exceptions=True)
        return ShellStepResult(
            exit_code=None,
            stdout=stdout_tail.text(),
            stderr=stderr_tail.text(),
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=True,
            error=f"script exceeded its {timeout}s timeout and was killed",
            command=command,
            contention=device_contention(command),
        )

    return ShellStepResult(
        exit_code=process.returncode,
        stdout=stdout_tail.text(),
        stderr=stderr_tail.text(),
        contention=device_contention(command) if process.returncode != 0 else [],
        duration_ms=int((time.monotonic() - started) * 1000),
        command=command,
    )


__all__ = ["ShellStepResult", "build_command", "run_registered_script"]
