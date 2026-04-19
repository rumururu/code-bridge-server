"""Tango scrcpy process management for Android device mirroring."""

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default Tango server port
DEFAULT_SCRCPY_PORT = 8000

# Device cache settings
_DEVICE_CACHE_TTL = 5.0  # Cache devices for 5 seconds


class ScrcpyManager:
    """Manages Tango scrcpy server process."""

    def __init__(self, scrcpy_path: Optional[str] = None, port: int = DEFAULT_SCRCPY_PORT):
        """Initialize ScrcpyManager.

        Args:
            scrcpy_path: Path to scrcpy installation (default: server/scrcpy)
            port: Port for Tango server (default: 8000)
        """
        if scrcpy_path:
            self.scrcpy_path = Path(scrcpy_path)
        else:
            # Default to server/scrcpy directory (parent.parent = server/)
            self.scrcpy_path = Path(__file__).parent.parent / "scrcpy"

        self.port = port
        self._process: Optional[asyncio.subprocess.Process] = None
        self._running = False
        # Device cache for non-blocking dashboard
        self._devices_cache: list[dict] = []
        self._devices_cached_at: float = 0.0
        self._devices_refresh_task: Optional[asyncio.Task] = None
        self._adb_path: Optional[str] = None
        self._emulator_path: Optional[str] = None

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

    def _extract_start_error(self, raw_output: str) -> str:
        """Extract a meaningful error message from Tango startup output."""
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            return "Tango server failed to start"

        # Node version footer is not actionable for users.
        filtered = [line for line in lines if not line.startswith("Node.js v")]
        if not filtered:
            return "Tango server failed to start"

        for line in filtered:
            if line.startswith("Error:"):
                return line

        for line in filtered:
            if "EADDRINUSE" in line or "EPERM" in line or "EACCES" in line:
                return line

        return filtered[-1]

    def _resolve_adb_path(self) -> Optional[str]:
        """Resolve adb executable path."""
        if self._adb_path:
            return self._adb_path

        candidates = [
            shutil.which("adb"),
            str(Path.home() / "Library/Android/sdk/platform-tools/adb"),
            str(Path.home() / "Android/Sdk/platform-tools/adb"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                self._adb_path = candidate
                return candidate
        return None

    def _resolve_emulator_path(self) -> Optional[str]:
        """Resolve emulator executable path."""
        if self._emulator_path:
            return self._emulator_path

        candidates = [
            shutil.which("emulator"),
            str(Path.home() / "Library/Android/sdk/emulator/emulator"),
            str(Path.home() / "Android/Sdk/emulator/emulator"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                self._emulator_path = candidate
                return candidate
        return None

    def is_installed(self) -> bool:
        """Check if Tango server is installed."""
        # Check for tango-server.mjs in dist directory
        dist_path = self.scrcpy_path / "dist"
        tango_server = dist_path / "tango-server.mjs"
        tango_service = dist_path / "src" / "server" / "goog-device" / "tango" / "TangoScrcpyService.mjs"
        return tango_server.exists() and tango_service.exists()

    async def get_devices(self) -> list[dict]:
        """Get list of connected Android devices via ADB.

        Uses caching with background refresh for non-blocking dashboard.
        Returns cached value immediately and refreshes in background if stale.

        Returns:
            List of device dictionaries with id, model, and state
        """
        now = time.time()
        cache_age = now - self._devices_cached_at

        # Return cached value immediately if available
        if self._devices_cache and cache_age < _DEVICE_CACHE_TTL:
            return self._devices_cache

        # If cache is stale, trigger background refresh
        if cache_age >= _DEVICE_CACHE_TTL:
            if self._devices_refresh_task is None or self._devices_refresh_task.done():
                self._devices_refresh_task = asyncio.create_task(self._refresh_devices())

        # If we have cached data, return it while refreshing in background
        if self._devices_cache:
            return self._devices_cache

        # No cache - must wait for first fetch
        await self._refresh_devices()
        return self._devices_cache

    async def _refresh_devices(self) -> None:
        """Refresh device cache from ADB and installed AVDs."""
        adb_path = self._resolve_adb_path()
        if not adb_path:
            self._devices_cache = []
            self._devices_cached_at = time.time()
            return

        try:
            # Get running devices from adb
            result = await asyncio.create_subprocess_exec(
                adb_path, "devices", "-l",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            devices = []
            running_emulator_ids = set()  # Track running emulator IDs
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

                    # Check if this is an emulator
                    is_emulator = device_id.startswith("emulator-")
                    if is_emulator:
                        running_emulator_ids.add(device_id)

                    devices.append({
                        "id": device_id,
                        "model": model,
                        "state": state,
                        "online": state == "device",
                        "is_emulator": is_emulator,
                    })

            # Get installed AVDs (including offline ones)
            avds = await self._get_installed_avds()

            # Map running emulators to their AVD names
            running_avd_names = await self._get_running_avd_names(running_emulator_ids)

            # Add offline AVDs to the list
            for avd_name in avds:
                if avd_name not in running_avd_names.values():
                    devices.append({
                        "id": f"avd:{avd_name}",
                        "model": avd_name,
                        "state": "offline",
                        "online": False,
                        "is_emulator": True,
                        "avd_name": avd_name,
                    })
                else:
                    # Update running emulator with AVD name
                    for device in devices:
                        if device.get("id") in running_avd_names and running_avd_names[device["id"]] == avd_name:
                            device["avd_name"] = avd_name
                            device["model"] = avd_name  # Use AVD name as model

            self._devices_cache = devices
            self._devices_cached_at = time.time()

        except FileNotFoundError:
            # adb not found
            self._devices_cache = []
            self._devices_cached_at = time.time()
        except OSError as e:
            logger.warning("Error getting devices: %s", e)
            # Keep old cache on error, but mark as stale
            if not self._devices_cache:
                self._devices_cache = []
                self._devices_cached_at = time.time()

    async def _get_installed_avds(self) -> list[str]:
        """Get list of installed Android Virtual Devices."""
        emulator_path = self._resolve_emulator_path()
        if not emulator_path:
            return []
        try:
            result = await asyncio.create_subprocess_exec(
                emulator_path, "-list-avds",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            avds = []
            for line in stdout.decode().strip().split("\n"):
                avd_name = line.strip()
                if avd_name:
                    avds.append(avd_name)
            return avds
        except FileNotFoundError:
            # emulator command not found
            return []
        except OSError as e:
            logger.debug("Error getting AVDs: %s", e)
            return []

    async def _get_running_avd_names(self, emulator_ids: set[str]) -> dict[str, str]:
        """Get AVD names for running emulators.

        Returns:
            Dict mapping emulator ID (e.g., 'emulator-5554') to AVD name
        """
        adb_path = self._resolve_adb_path()
        if not adb_path:
            return {}

        result = {}
        for emulator_id in emulator_ids:
            try:
                # Get AVD name via adb
                proc = await asyncio.create_subprocess_exec(
                    adb_path, "-s", emulator_id, "emu", "avd", "name",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
                avd_name = stdout.decode().strip().split("\n")[0].strip()
                if avd_name and avd_name != "OK":
                    result[emulator_id] = avd_name
            except (asyncio.TimeoutError, OSError):
                # Timeout or ADB error - skip this emulator
                pass
        return result

    async def _list_running_emulator_ids(self) -> set[str]:
        """Return currently running emulator serial numbers."""
        adb_path = self._resolve_adb_path()
        if not adb_path:
            return set()

        try:
            result = await asyncio.create_subprocess_exec(
                adb_path,
                "devices",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
        except OSError:
            return set()

        emulator_ids: set[str] = set()
        lines = stdout.decode(errors="replace").strip().split("\n")[1:]
        for line in lines:
            parts = line.split()
            if parts and parts[0].startswith("emulator-"):
                emulator_ids.add(parts[0])
        return emulator_ids

    async def _wait_for_android_device_boot(
        self,
        device_id: str,
        *,
        timeout: float = 180.0,
    ) -> bool:
        """Wait until Android reports boot completed for the given device."""
        adb_path = self._resolve_adb_path()
        if not adb_path:
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                proc = await asyncio.create_subprocess_exec(
                    adb_path,
                    "-s",
                    device_id,
                    "shell",
                    "getprop",
                    "sys.boot_completed",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
                if stdout.decode(errors="replace").strip() == "1":
                    return True
            except (asyncio.TimeoutError, OSError):
                pass

            await asyncio.sleep(2.0)

        return False

    async def _find_running_emulator_for_avd(self, avd_name: str) -> Optional[str]:
        """Resolve running emulator serial for an AVD name."""
        emulator_ids = await self._list_running_emulator_ids()
        if not emulator_ids:
            return None

        running_avd_names = await self._get_running_avd_names(emulator_ids)
        for emulator_id, running_name in running_avd_names.items():
            if running_name == avd_name:
                return emulator_id
        return None

    async def _launch_avd(self, avd_name: str) -> Optional[str]:
        """Launch an Android Virtual Device and return error text on failure."""
        emulator_path = self._resolve_emulator_path()
        if not emulator_path:
            return "Android emulator CLI not found"

        installed_avds = await self._get_installed_avds()
        if avd_name not in installed_avds:
            return f"Android emulator '{avd_name}' is not installed"

        try:
            subprocess.Popen(
                [emulator_path, "-avd", avd_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=os.environ.copy(),
            )
            return None
        except OSError as exc:
            logger.warning("Failed to launch AVD %s: %s", avd_name, exc)
            return f"Failed to launch emulator {avd_name}: {exc}"

    async def ensure_emulator_ready(
        self,
        device_id: str,
        *,
        timeout: float = 180.0,
    ) -> tuple[Optional[str], Optional[str]]:
        """Ensure an Android emulator is running and boot-complete.

        Accepts either a running emulator serial (`emulator-5554`) or an
        installed AVD placeholder id (`avd:Pixel_8_API_35`).
        """
        normalized_device_id = device_id.strip()
        if not normalized_device_id:
            return None, "Device ID is required"

        if normalized_device_id.startswith("emulator-"):
            ready = await self._wait_for_android_device_boot(
                normalized_device_id,
                timeout=timeout,
            )
            if not ready:
                return None, f"Emulator {normalized_device_id} did not finish booting"
            await self._refresh_devices()
            return normalized_device_id, None

        if not normalized_device_id.startswith("avd:"):
            return None, "Web mirror preview currently supports Android emulators only"

        avd_name = normalized_device_id[4:].strip()
        if not avd_name:
            return None, "Android emulator name is missing"

        running_device_id = await self._find_running_emulator_for_avd(avd_name)
        if running_device_id is None:
            launch_error = await self._launch_avd(avd_name)
            if launch_error:
                return None, launch_error

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            running_device_id = await self._find_running_emulator_for_avd(avd_name)
            if running_device_id:
                ready = await self._wait_for_android_device_boot(
                    running_device_id,
                    timeout=max(5.0, deadline - time.monotonic()),
                )
                if ready:
                    await self._refresh_devices()
                    return running_device_id, None
            await asyncio.sleep(2.0)

        return None, f"Timed out waiting for emulator {avd_name} to boot"

    async def configure_emulator_display(
        self,
        device_id: str,
        *,
        width: int | None = None,
        height: int | None = None,
        density: int | None = None,
        reset_to_default: bool = False,
    ) -> Optional[str]:
        """Apply or reset emulator display overrides through adb shell wm."""
        adb_path = self._resolve_adb_path()
        if not adb_path:
            return "adb not found"

        commands: list[list[str]] = []
        if reset_to_default:
            commands.extend(
                [
                    [
                        adb_path,
                        "-s",
                        device_id,
                        "shell",
                        "wm",
                        "size",
                        "reset",
                    ],
                    [
                        adb_path,
                        "-s",
                        device_id,
                        "shell",
                        "wm",
                        "density",
                        "reset",
                    ],
                ]
            )
        else:
            if width is not None and height is not None:
                commands.append(
                    [
                        adb_path,
                        "-s",
                        device_id,
                        "shell",
                        "wm",
                        "size",
                        f"{width}x{height}",
                    ]
                )
            elif width is not None or height is not None:
                return "Both width and height are required to override emulator size"

            if density is not None:
                commands.append(
                    [
                        adb_path,
                        "-s",
                        device_id,
                        "shell",
                        "wm",
                        "density",
                        str(density),
                    ]
                )

        if not commands:
            return None

        had_failure = False
        last_error = f"Failed to configure display for {device_id}"
        for command in commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            except (asyncio.TimeoutError, OSError) as exc:
                had_failure = True
                last_error = f"Failed to configure display for {device_id}: {exc}"
                continue

            if proc.returncode == 0:
                continue

            had_failure = True

            output = "\n".join(
                part.strip()
                for part in (
                    stdout.decode("utf-8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"),
                )
                if part.strip()
            )
            if output:
                last_error = output

        return last_error if had_failure else None

    async def setup_reverse_port(self, device_id: str, port: int) -> Optional[str]:
        """adb reverse TCP forwarding: emulator localhost:<port> -> host:<port>.

        We serve the dev server URL as http://localhost:<port> inside Chrome on
        the emulator (instead of http://10.0.2.2:<port>) because Firebase Auth
        and most OAuth providers auto-authorize `localhost` but reject
        arbitrary host addresses. Without the reverse forward Chrome's
        localhost resolves to the emulator itself (nothing listening); this
        call wires it back to the developer Mac.
        """
        adb_path = self._resolve_adb_path()
        if not adb_path:
            return "adb not found"
        proc = await asyncio.create_subprocess_exec(
            adb_path,
            "-s",
            device_id,
            "reverse",
            f"tcp:{port}",
            f"tcp:{port}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except asyncio.TimeoutError:
            return "adb reverse timed out"
        if proc.returncode != 0:
            return stderr.decode("utf-8", errors="replace").strip() or "adb reverse failed"
        return None

    async def open_url_in_browser(self, device_id: str, url: str) -> Optional[str]:
        """Open a URL on the device, preferring Chrome.

        Before launching Chrome we drop a Chromium command-line file that
        suppresses the Welcome / sign-in / default-browser interstitials so
        the dev server URL renders immediately. This is best-effort — if the
        prepare fails Chrome will still launch (just with its usual UI).
        """
        adb_path = self._resolve_adb_path()
        if not adb_path:
            return "adb not found"

        from .chrome_first_run import (
            ensure_chrome_first_run_skipped,
            _dismiss_notification_prompt,
        )

        await ensure_chrome_first_run_skipped(adb_path, device_id)

        commands = [
            [
                adb_path,
                "-s",
                device_id,
                "shell",
                "am",
                "start",
                "-W",
                "-n",
                "com.android.chrome/com.google.android.apps.chrome.Main",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                url,
            ],
            [
                adb_path,
                "-s",
                device_id,
                "shell",
                "am",
                "start",
                "-W",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                url,
                "com.android.chrome",
            ],
            [
                adb_path,
                "-s",
                device_id,
                "shell",
                "am",
                "start",
                "-W",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                url,
            ],
        ]

        last_error = "Failed to open URL on Android device"
        for command in commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            except (asyncio.TimeoutError, OSError) as exc:
                last_error = f"Failed to open URL on {device_id}: {exc}"
                continue

            if proc.returncode == 0:
                # Chrome launched. Give it a beat to render its first-run UI
                # (the prompt only appears AFTER the activity inflates) and
                # then auto-dismiss every interstitial we know about.
                await asyncio.sleep(4.0)
                await _dismiss_notification_prompt(adb_path, device_id)
                return None

            output = "\n".join(
                part.strip()
                for part in (
                    stdout.decode("utf-8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"),
                )
                if part.strip()
            )
            if output:
                last_error = output.splitlines()[-1]

        return last_error

    async def start(self) -> dict:
        """Start Tango scrcpy server.

        Returns:
            Dictionary with success status and message/error
        """
        if self.is_running:
            return {"success": True, "message": "Already running", "url": self.scrcpy_url}

        if not self.is_installed():
            return {
                "success": False,
                "error": "Tango server not installed. Run: cd server/scrcpy && npm run build",
            }

        try:
            last_error = "Tango server failed to start"
            base_port = self.port

            # Retry a few times on address conflicts.
            for attempt in range(5):
                candidate = self._find_available_port(base_port + attempt)
                self.port = candidate

                # Start Tango server from dist directory
                # Tango reads port from command line argument
                dist_path = self.scrcpy_path / "dist"
                self._process = await asyncio.create_subprocess_exec(
                    "node",
                    "tango-server.mjs",
                    str(self.port),
                    cwd=str(dist_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=os.environ.copy(),
                )

                self._running = True
                await asyncio.sleep(2)

                if self._process.returncode is None:
                    return {
                        "success": True,
                        "message": "Tango server started",
                        "url": self.scrcpy_url,
                        "port": self.port,
                    }

                self._running = False
                output = b""
                if self._process.stdout is not None:
                    try:
                        output = await asyncio.wait_for(self._process.stdout.read(), timeout=1.0)
                    except (asyncio.TimeoutError, OSError):
                        output = b""
                text = output.decode("utf-8", errors="replace")
                last_error = self._extract_start_error(text)

                # Retry only for port conflict cases.
                if "EADDRINUSE" not in text and "address already in use" not in text:
                    break

            self._process = None
            return {"success": False, "error": last_error}

        except OSError as e:
            logger.warning("Failed to start Tango server: %s", e)
            self._running = False
            return {"success": False, "error": str(e)}

    async def stop(self) -> dict:
        """Stop Tango scrcpy server.

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
        except (OSError, ProcessLookupError) as e:
            logger.debug("Error stopping Tango server: %s", e)
            return {"success": False, "error": str(e)}
        finally:
            self._running = False
            self._process = None

        return {"success": True, "message": "Tango server stopped"}

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
