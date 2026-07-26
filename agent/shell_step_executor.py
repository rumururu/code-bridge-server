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

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        process.kill()
        # Reap the killed process so it does not linger as a zombie.
        try:
            await process.communicate()
        except Exception:
            pass
        return ShellStepResult(
            exit_code=None,
            stdout="",
            stderr="",
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=True,
            error=f"script exceeded its {timeout}s timeout and was killed",
            command=command,
        )

    return ShellStepResult(
        exit_code=process.returncode,
        stdout=_tail((stdout_bytes or b"").decode("utf-8", "replace")),
        stderr=_tail((stderr_bytes or b"").decode("utf-8", "replace")),
        duration_ms=int((time.monotonic() - started) * 1000),
        command=command,
    )


__all__ = ["ShellStepResult", "build_command", "run_registered_script"]
