"""ws-scrcpy process management for Android device mirroring."""

import asyncio
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Optional

# Default ws-scrcpy port
DEFAULT_SCRCPY_PORT = 8000


class ScrcpyManager:
    """Manages ws-scrcpy server process."""

    def __init__(self, scrcpy_path: Optional[str] = None, port: int = DEFAULT_SCRCPY_PORT):
        """Initialize ScrcpyManager.

        Args:
            scrcpy_path: Path to ws-scrcpy installation (default: server/scrcpy)
            port: Port for ws-scrcpy server (default: 8000)
        """
        if scrcpy_path:
            self.scrcpy_path = Path(scrcpy_path)
        else:
            # Default to server/scrcpy directory
            self.scrcpy_path = Path(__file__).parent / "scrcpy"

        self.port = port
        self._process: Optional[asyncio.subprocess.Process] = None
        self._running = False
        self._runtime_config_path: Optional[Path] = None

    @property
    def is_running(self) -> bool:
        """Check if ws-scrcpy is running."""
        return self._running and self._process is not None and self._process.returncode is None

    @property
    def scrcpy_url(self) -> str:
        """Get ws-scrcpy URL."""
        return f"http://localhost:{self.port}"

    def _is_port_in_use(self, port: int) -> bool:
        """Check whether a TCP port is already bound."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def _find_available_port(self, preferred: int, max_tries: int = 100) -> int:
        """Find an available TCP port near the preferred one."""
        for offset in range(max_tries):
            candidate = preferred + offset
            if not self._is_port_in_use(candidate):
                return candidate
        raise RuntimeError(f"No available port found in range {preferred}-{preferred + max_tries - 1}")

    def _cleanup_runtime_config(self) -> None:
        """Remove temporary ws-scrcpy runtime config, if present."""
        if self._runtime_config_path and self._runtime_config_path.exists():
            try:
                self._runtime_config_path.unlink()
            except Exception:
                pass
        self._runtime_config_path = None

    def _create_runtime_config(self, port: int) -> Path:
        """Create a temporary ws-scrcpy config file with a concrete server port."""
        runtime_config = {
            "runGoogTracker": True,
            "runApplTracker": False,
            "server": [{"secure": False, "port": port}],
            "remoteHostList": [],
        }
        fd, temp_path = tempfile.mkstemp(
            prefix="ws_scrcpy_runtime_",
            suffix=".json",
            dir=str(self.scrcpy_path),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(runtime_config, file)
        self._runtime_config_path = Path(temp_path)
        return self._runtime_config_path

    def _extract_start_error(self, raw_output: str) -> str:
        """Extract a meaningful error message from ws-scrcpy startup output."""
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            return "ws-scrcpy failed to start"

        # Node version footer is not actionable for users.
        filtered = [line for line in lines if not line.startswith("Node.js v")]
        if not filtered:
            return "ws-scrcpy failed to start"

        for line in filtered:
            if line.startswith("Error:"):
                return line

        for line in filtered:
            if "EADDRINUSE" in line or "EPERM" in line or "EACCES" in line:
                return line

        return filtered[-1]

    def is_installed(self) -> bool:
        """Check if ws-scrcpy is installed."""
        # Check for built dist directory
        dist_path = self.scrcpy_path / "dist"
        dist_index = dist_path / "index.js"
        dist_node_modules = dist_path / "node_modules"
        return dist_index.exists() and dist_node_modules.exists()

    async def get_devices(self) -> list[dict]:
        """Get list of connected Android devices via ADB.

        Returns:
            List of device dictionaries with id, model, and state
        """
        try:
            result = await asyncio.create_subprocess_exec(
                "adb", "devices", "-l",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            devices = []
            lines = stdout.decode().strip().split("\n")[1:]  # Skip header

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    device_id = parts[0]
                    state = parts[1]

                    # Extract model from device info
                    model = "Unknown"
                    for part in parts[2:]:
                        if part.startswith("model:"):
                            model = part.split(":")[1]
                            break

                    devices.append({
                        "id": device_id,
                        "model": model,
                        "state": state,
                    })

            return devices

        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error getting devices: {e}")
            return []

    async def start(self) -> dict:
        """Start ws-scrcpy server.

        Returns:
            Dictionary with success status and message/error
        """
        if self.is_running:
            return {"success": True, "message": "Already running", "url": self.scrcpy_url}

        if not self.is_installed():
            return {
                "success": False,
                "error": "ws-scrcpy not installed. Run: cd server/scrcpy && npm run dist:dev && cd dist && npm install",
            }

        try:
            last_error = "ws-scrcpy failed to start"
            base_port = self.port

            # Retry a few times on address conflicts.
            for attempt in range(5):
                candidate = self._find_available_port(base_port + attempt)
                self.port = candidate
                self._cleanup_runtime_config()
                runtime_config_path = self._create_runtime_config(self.port)

                # Start ws-scrcpy from dist directory using node directly.
                # ws-scrcpy reads port from WS_SCRCPY_CONFIG (not from PORT).
                dist_path = self.scrcpy_path / "dist"
                self._process = await asyncio.create_subprocess_exec(
                    "node",
                    "index.js",
                    cwd=str(dist_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env={**os.environ, "WS_SCRCPY_CONFIG": str(runtime_config_path)},
                )

                self._running = True
                await asyncio.sleep(2)

                if self._process.returncode is None:
                    self._cleanup_runtime_config()
                    return {
                        "success": True,
                        "message": "ws-scrcpy started",
                        "url": self.scrcpy_url,
                        "port": self.port,
                    }

                self._running = False
                output = b""
                if self._process.stdout is not None:
                    try:
                        output = await asyncio.wait_for(self._process.stdout.read(), timeout=1.0)
                    except Exception:
                        output = b""
                text = output.decode("utf-8", errors="replace")
                last_error = self._extract_start_error(text)

                # Retry only for port conflict cases.
                if "EADDRINUSE" not in text and "address already in use" not in text:
                    break

            self._cleanup_runtime_config()
            self._process = None
            return {"success": False, "error": last_error}

        except Exception as e:
            self._running = False
            self._cleanup_runtime_config()
            return {"success": False, "error": str(e)}

    async def stop(self) -> dict:
        """Stop ws-scrcpy server.

        Returns:
            Dictionary with success status
        """
        if not self.is_running:
            return {"success": True, "message": "Not running"}

        try:
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._process.kill()
            await self._process.wait()
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self._running = False
            self._process = None
            self._cleanup_runtime_config()

        return {"success": True, "message": "ws-scrcpy stopped"}

    def get_status(self) -> dict:
        """Get current status.

        Returns:
            Dictionary with running state and URL
        """
        return {
            "running": self.is_running,
            "installed": self.is_installed(),
            "url": self.scrcpy_url if self.is_running else None,
            "port": self.port,
        }


# Singleton instance
_scrcpy_manager: Optional[ScrcpyManager] = None


def get_scrcpy_manager() -> ScrcpyManager:
    """Get singleton ScrcpyManager instance."""
    global _scrcpy_manager
    if _scrcpy_manager is None:
        _scrcpy_manager = ScrcpyManager()
    return _scrcpy_manager
