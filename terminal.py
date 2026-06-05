"""Terminal management for running commands in project directories."""

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Optional, Union

logger = logging.getLogger(__name__)

# Dangerous commands to block
BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "dd if=/dev/zero",
    "mkfs",
    ":(){ :|:& };:",  # Fork bomb
    "> /dev/sda",
    "chmod -R 777 /",
}

# Commands that require extra caution
CAUTION_PATTERNS = [
    "rm -rf",
    "sudo",
    "chmod",
    "chown",
    "kill",
    "pkill",
]

# Maximum command execution time (seconds)
MAX_EXECUTION_TIME = 300

# Maximum output size (bytes)
MAX_OUTPUT_SIZE = 1024 * 1024  # 1MB


@dataclass
class CommandResult:
    """Result of a command execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: Optional[str] = None
    timed_out: bool = False


@dataclass
class TerminalSession:
    """A terminal session for a project."""

    project_path: str
    command_history: list = field(default_factory=list)
    current_process: Optional[asyncio.subprocess.Process] = None

    def _is_blocked(self, command: str) -> bool:
        """Check if command is blocked."""
        cmd_lower = command.lower().strip()
        return any(blocked in cmd_lower for blocked in BLOCKED_COMMANDS)

    def _needs_caution(self, command: str) -> list[str]:
        """Check if command needs caution warnings."""
        warnings = []
        cmd_lower = command.lower()

        for pattern in CAUTION_PATTERNS:
            if pattern in cmd_lower:
                warnings.append(f"Command contains '{pattern}'")

        return warnings

    async def execute(self, command: str, timeout: int = MAX_EXECUTION_TIME) -> CommandResult:
        """Execute a command in the project directory.

        Args:
            command: The command to execute
            timeout: Maximum execution time in seconds

        Returns:
            CommandResult with stdout, stderr, and exit code
        """
        # Security check
        if self._is_blocked(command):
            return CommandResult(
                error="This command is blocked for security reasons",
                exit_code=1
            )

        # Add to history
        self.command_history.append(command)
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]

        try:
            # Create subprocess
            self.current_process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
                env={**os.environ, "TERM": "xterm-256color"},
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    self.current_process.communicate(),
                    timeout=timeout
                )

                # Decode output
                stdout_str = stdout.decode("utf-8", errors="replace")
                stderr_str = stderr.decode("utf-8", errors="replace")

                # Truncate if too large
                if len(stdout_str) > MAX_OUTPUT_SIZE:
                    stdout_str = stdout_str[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
                if len(stderr_str) > MAX_OUTPUT_SIZE:
                    stderr_str = stderr_str[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"

                return CommandResult(
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=self.current_process.returncode or 0,
                )

            except asyncio.TimeoutError:
                # Kill the process
                self.current_process.kill()
                await self.current_process.wait()

                return CommandResult(
                    error=f"Command timed out after {timeout} seconds",
                    exit_code=124,  # Standard timeout exit code
                    timed_out=True,
                )

        except OSError as e:
            return CommandResult(
                error=str(e),
                exit_code=1,
            )
        finally:
            self.current_process = None

    async def execute_with_emitter(
        self,
        command: str,
        *,
        on_chunk: Optional[Callable[[dict], Union[None, Awaitable[None]]]] = None,
        timeout: int = MAX_EXECUTION_TIME,
    ) -> CommandResult:
        """Run ``command`` and push stdout/stderr chunks to ``on_chunk`` as they arrive.

        Returns the same buffered :class:`CommandResult` as :meth:`execute`
        so callers that want the final captured output still get it. Use this
        method when the caller needs both real-time log streaming **and** a
        final result object.

        ``on_chunk`` receives dicts shaped like::

            {"stream": "stdout"|"stderr", "data": "<text>"}

        Optionally async; both sync and async callbacks are supported.
        Exceptions in the callback are logged and swallowed so a broken
        listener never corrupts the underlying command execution.
        """
        if self._is_blocked(command):
            return CommandResult(
                error="This command is blocked for security reasons",
                exit_code=1,
            )

        async def _emit(payload: dict) -> None:
            if on_chunk is None:
                return
            try:
                result = on_chunk(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 — listener errors are non-fatal
                # Intentionally swallowed: emitter problems must not affect
                # the underlying terminal command. We pick up the buffered
                # result regardless.
                pass

        self.command_history.append(command)
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]

        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        stdout_size = 0
        stderr_size = 0
        truncated = False

        try:
            self.current_process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
                env={**os.environ, "TERM": "xterm-256color"},
            )

            async def pump(stream: asyncio.StreamReader, stream_type: str) -> None:
                nonlocal stdout_size, stderr_size, truncated
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        return
                    decoded = chunk.decode("utf-8", errors="replace")
                    if stream_type == "stdout":
                        if stdout_size >= MAX_OUTPUT_SIZE:
                            if not truncated:
                                truncated = True
                                stdout_buf.append("\n... (output truncated)")
                                await _emit({"stream": "stdout", "data": "\n... (output truncated)"})
                            continue
                        stdout_size += len(decoded)
                        stdout_buf.append(decoded)
                    else:
                        if stderr_size >= MAX_OUTPUT_SIZE:
                            if not truncated:
                                truncated = True
                                stderr_buf.append("\n... (output truncated)")
                                await _emit({"stream": "stderr", "data": "\n... (output truncated)"})
                            continue
                        stderr_size += len(decoded)
                        stderr_buf.append(decoded)
                    await _emit({"stream": stream_type, "data": decoded})

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        pump(self.current_process.stdout, "stdout"),
                        pump(self.current_process.stderr, "stderr"),
                    ),
                    timeout=timeout,
                )
                await self.current_process.wait()
            except asyncio.TimeoutError:
                self.current_process.kill()
                await self.current_process.wait()
                return CommandResult(
                    stdout="".join(stdout_buf),
                    stderr="".join(stderr_buf),
                    exit_code=124,
                    error=f"Command timed out after {timeout} seconds",
                    timed_out=True,
                )

            return CommandResult(
                stdout="".join(stdout_buf),
                stderr="".join(stderr_buf),
                exit_code=self.current_process.returncode or 0,
            )
        except OSError as e:
            return CommandResult(error=str(e), exit_code=1)
        finally:
            self.current_process = None

    async def execute_stream(self, command: str, timeout: int = MAX_EXECUTION_TIME) -> AsyncIterator[dict]:
        """Execute a command and stream output.

        Yields:
            Dict with 'type' (stdout/stderr/exit) and 'data'
        """
        # Security check
        if self._is_blocked(command):
            yield {"type": "error", "data": "This command is blocked for security reasons"}
            yield {"type": "exit", "data": 1}
            return

        # Add to history
        self.command_history.append(command)
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]

        try:
            # Create subprocess
            self.current_process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
                env={**os.environ, "TERM": "xterm-256color"},
            )

            async def read_stream(stream, stream_type):
                """Read from a stream and yield chunks."""
                total_size = 0
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break

                    decoded = chunk.decode("utf-8", errors="replace")
                    total_size += len(decoded)

                    if total_size > MAX_OUTPUT_SIZE:
                        yield {"type": stream_type, "data": "\n... (output truncated)"}
                        break

                    yield {"type": stream_type, "data": decoded}

            # Create tasks for reading both streams
            async def read_both():
                stdout_task = asyncio.create_task(self._read_all(self.current_process.stdout, "stdout"))
                stderr_task = asyncio.create_task(self._read_all(self.current_process.stderr, "stderr"))

                done, pending = await asyncio.wait(
                    [stdout_task, stderr_task],
                    timeout=timeout,
                    return_when=asyncio.ALL_COMPLETED
                )

                results = []
                for task in done:
                    results.extend(task.result())

                if pending:
                    # Timeout occurred
                    for task in pending:
                        task.cancel()
                    self.current_process.kill()
                    results.append({"type": "error", "data": f"Command timed out after {timeout} seconds"})
                    results.append({"type": "exit", "data": 124})
                else:
                    await self.current_process.wait()
                    results.append({"type": "exit", "data": self.current_process.returncode or 0})

                return results

            # Read all output
            results = await read_both()
            for result in results:
                yield result

        except OSError:
            # Streamed errors reach the client directly, so log the full
            # detail server-side and surface only a generic notice.
            logger.exception("terminal stream OSError for command")
            yield {"type": "error", "data": "Command failed"}
            yield {"type": "exit", "data": 1}
        finally:
            self.current_process = None

    async def _read_all(self, stream, stream_type) -> list[dict]:
        """Read all from a stream."""
        results = []
        total_size = 0

        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break

            decoded = chunk.decode("utf-8", errors="replace")
            total_size += len(decoded)

            if total_size > MAX_OUTPUT_SIZE:
                results.append({"type": stream_type, "data": "\n... (output truncated)"})
                break

            results.append({"type": stream_type, "data": decoded})

        return results

    async def cancel(self) -> bool:
        """Cancel the current running process."""
        if self.current_process:
            self.current_process.kill()
            await self.current_process.wait()
            return True
        return False


class TerminalManager:
    """Manages terminal sessions for projects."""

    def __init__(self):
        self._sessions: dict[str, TerminalSession] = {}

    def get_session(self, project_name: str, project_path: str) -> TerminalSession:
        """Get or create a terminal session for a project."""
        if project_name not in self._sessions:
            self._sessions[project_name] = TerminalSession(project_path=project_path)
        return self._sessions[project_name]

    def close_session(self, project_name: str) -> bool:
        """Close a terminal session."""
        if project_name in self._sessions:
            del self._sessions[project_name]
            return True
        return False


# Singleton instance
_terminal_manager: Optional[TerminalManager] = None


def get_terminal_manager() -> TerminalManager:
    """Get the terminal manager singleton."""
    global _terminal_manager
    if _terminal_manager is None:
        _terminal_manager = TerminalManager()
    return _terminal_manager
