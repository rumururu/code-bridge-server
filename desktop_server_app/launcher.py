#!/usr/bin/env python3
"""Tray/menu-bar launcher for Code Bridge Server."""

from __future__ import annotations

import os
import platform
import plistlib
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from queue import Queue

try:
    import pystray
    from PIL import Image, ImageDraw
except Exception:
    pystray = None
    Image = ImageDraw = None


APP_NAME = "Code Bridge Server"
BUNDLE_IDENTIFIER = "com.mkideabox.codebridge.server"
DEFAULT_DASHBOARD_PORT = 8766
DEFAULT_API_PORT = 8767
MACOS_MENU_BAR_TITLE = "CB"


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _bundle_root() -> Path:
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _server_dir() -> Path:
    if _is_frozen():
        resources_server = Path(sys.executable).resolve().parent.parent / "Resources" / "server"
        if resources_server.exists():
            return resources_server
    bundled = _bundle_root() / "server"
    if (bundled / "server_cli.py").exists():
        return bundled
    # Script install (install.sh / install.ps1): the install dir is itself
    # the server tree, with this launcher living in desktop_server_app/
    # beside it — there is no nested server/ directory.
    if (_bundle_root() / "server_cli.py").exists():
        return _bundle_root()
    return _repo_root() / "server"


def _app_support_dir() -> Path:
    system = platform.system().lower()
    home = Path.home()
    if system == "darwin":
        base = home / "Library" / "Application Support"
    elif system == "windows":
        base = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    path = base / "Code Bridge Server"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _launcher_log_file() -> Path:
    logs = _app_support_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / "launcher.log"


def _write_launcher_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _launcher_log_file().open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] {message}\n")
    except OSError:
        pass


def _bundled_node_path(server_dir: Path) -> Path:
    executable = "node.exe" if platform.system().lower() == "windows" else "node"
    return server_dir / "vendor" / "node" / "bin" / executable


def _bundled_adb_path(server_dir: Path) -> Path:
    executable = "adb.exe" if platform.system().lower() == "windows" else "adb"
    return server_dir / "vendor" / "platform-tools" / executable


def _prepend_path(env: dict[str, str], path: Path) -> None:
    existing = env.get("PATH", "")
    env["PATH"] = str(path) if not existing else os.pathsep.join([str(path), existing])


def _asset_path(*parts: str) -> Path | None:
    candidates = [
        _bundle_root().joinpath(*parts),
        Path(sys.executable).resolve().parent.joinpath(*parts),
        _repo_root().joinpath(*parts),
    ]
    return next((path for path in candidates if path.exists()), None)


def _parse_port_from_uvicorn_line(line: str) -> tuple[str, int] | None:
    marker = "Uvicorn running on http://"
    if marker not in line:
        return None
    address = line.split(marker, 1)[1].split(" ", 1)[0]
    if ":" not in address:
        return None
    host, raw_port = address.rsplit(":", 1)
    try:
        port = int(raw_port.strip("/"))
    except ValueError:
        return None
    return host, port


class AutoStartManager:
    def __init__(self, app_support_dir: Path) -> None:
        self.app_support_dir = app_support_dir

    @property
    def system(self) -> str:
        return platform.system().lower()

    def launch_command(self) -> list[str]:
        if _is_frozen():
            return [str(Path(sys.executable).resolve())]
        return [str(Path(sys.executable).resolve()), str(Path(__file__).resolve())]

    def is_enabled(self) -> bool:
        if self.system == "darwin":
            return self._macos_plist_path().exists()
        if self.system == "windows":
            try:
                import winreg

                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self._windows_run_key()) as key:
                    winreg.QueryValueEx(key, APP_NAME)
                return True
            except OSError:
                return False
        return self._linux_desktop_file_path().exists()

    def set_enabled(self, enabled: bool) -> None:
        if self.system == "darwin":
            self._set_macos_enabled(enabled)
        elif self.system == "windows":
            self._set_windows_enabled(enabled)
        else:
            self._set_linux_enabled(enabled)

    def _macos_plist_path(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{BUNDLE_IDENTIFIER}.plist"

    def _set_macos_enabled(self, enabled: bool) -> None:
        plist_path = self._macos_plist_path()
        if not enabled:
            plist_path.unlink(missing_ok=True)
            return

        logs = self.app_support_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        payload = {
            "Label": BUNDLE_IDENTIFIER,
            "ProgramArguments": self.launch_command(),
            "RunAtLoad": True,
            "KeepAlive": False,
            "StandardOutPath": str(logs / "launchd.out.log"),
            "StandardErrorPath": str(logs / "launchd.err.log"),
        }
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        with plist_path.open("wb") as fh:
            plistlib.dump(payload, fh)

    @staticmethod
    def _windows_run_key() -> str:
        return r"Software\Microsoft\Windows\CurrentVersion\Run"

    def _set_windows_enabled(self, enabled: bool) -> None:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            self._windows_run_key(),
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            if enabled:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, subprocess.list2cmdline(self.launch_command()))
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass

    def _linux_desktop_file_path(self) -> Path:
        return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "autostart" / "code-bridge-server.desktop"

    def _set_linux_enabled(self, enabled: bool) -> None:
        desktop_file = self._linux_desktop_file_path()
        if not enabled:
            desktop_file.unlink(missing_ok=True)
            return

        command = " ".join(subprocess.list2cmdline([part]) for part in self.launch_command())
        desktop_file.parent.mkdir(parents=True, exist_ok=True)
        desktop_file.write_text(
            "\n".join(
                [
                    "[Desktop Entry]",
                    "Type=Application",
                    f"Name={APP_NAME}",
                    f"Exec={command}",
                    "Terminal=false",
                    "X-GNOME-Autostart-enabled=true",
                    "",
                ]
            ),
            encoding="utf-8",
        )


class ServerManager:
    def __init__(self) -> None:
        self.server_dir = _server_dir()
        self.app_support_dir = _app_support_dir()
        self.process: subprocess.Popen[str] | None = None
        self.status = "Stopped"
        self.last_error = ""
        self.dashboard_port = int(os.environ.get("CODEBRIDGE_DASHBOARD_PORT", DEFAULT_DASHBOARD_PORT))
        self.api_port = int(os.environ.get("CODEBRIDGE_API_PORT", DEFAULT_API_PORT))
        self.log_queue: Queue[str] = Queue()
        self._lock = threading.RLock()

    def is_running(self) -> bool:
        with self._lock:
            return self.process is not None and self.process.poll() is None

    def dashboard_url(self) -> str:
        return f"http://localhost:{self.dashboard_port}/dashboard"

    def pair_url(self) -> str:
        return f"http://localhost:{self.dashboard_port}/pair"

    def api_url(self) -> str:
        return f"http://localhost:{self.api_port}"

    def server_command(self) -> list[str]:
        if _is_frozen():
            return [sys.executable, "--server-child"]

        venv_python = self.server_dir / "venv" / "bin" / "python"
        if platform.system().lower() == "windows":
            venv_python = self.server_dir / "venv" / "Scripts" / "python.exe"
        python = str(venv_python if venv_python.exists() else Path(sys.executable))
        return [python, str(self.server_dir / "server_cli.py"), "--restart"]

    def server_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CODEBRIDGE_DESKTOP_APP"] = "1"
        env.setdefault("CODEBRIDGE_DASHBOARD_PORT", str(DEFAULT_DASHBOARD_PORT))
        env.setdefault("CODEBRIDGE_API_PORT", str(DEFAULT_API_PORT))
        env["CODEBRIDGE_APP_SUPPORT_DIR"] = str(self.app_support_dir)

        bundled_node = _bundled_node_path(self.server_dir)
        if bundled_node.exists():
            env["CODEBRIDGE_NODE_PATH"] = str(bundled_node)

        bundled_adb = _bundled_adb_path(self.server_dir)
        if bundled_adb.exists():
            env["CODEBRIDGE_ADB_PATH"] = str(bundled_adb)
            _prepend_path(env, bundled_adb.parent)

        python_path = str(self.server_dir)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = python_path if not existing else os.pathsep.join([python_path, existing])
        return env

    def start(self) -> None:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                return
            command = self.server_command()
            if not _is_frozen() and not Path(command[1]).exists():
                self.status = "Error"
                self.last_error = f"Server entry not found: {command[1]}"
                _write_launcher_log(self.last_error)
                return

            self.status = "Starting"
            self.last_error = ""
            _write_launcher_log("$ " + subprocess.list2cmdline(command))
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=str(self.server_dir),
                    env=self.server_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                self.status = "Error"
                self.last_error = str(exc)
                _write_launcher_log(f"Failed to start server: {exc}")
                return

        threading.Thread(target=self._read_process_output, daemon=True).start()
        threading.Thread(target=self._wait_for_process, daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            process = self.process
            if process is None or process.poll() is not None:
                self.status = "Stopped"
                return
            self.status = "Stopping"

        _write_launcher_log("Stopping server...")
        try:
            if platform.system().lower() == "windows":
                process.terminate()
            else:
                process.send_signal(signal.SIGTERM)
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _write_launcher_log("Server did not stop in time; killing process.")
            process.kill()
        finally:
            with self._lock:
                self.status = "Stopped"

    def _read_process_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            self.log_queue.put(line)
            _write_launcher_log(line)
            parsed = _parse_port_from_uvicorn_line(line)
            if parsed:
                host, port = parsed
                if host in {"127.0.0.1", "localhost"}:
                    self.dashboard_port = port
                elif host in {"0.0.0.0", "*"}:
                    self.api_port = port
                with self._lock:
                    if self.status != "Stopping":
                        self.status = "Running"

    def _wait_for_process(self) -> None:
        process = self.process
        if process is None:
            return
        code = process.wait()
        _write_launcher_log(f"Server process exited with code {code}.")
        with self._lock:
            if self.process is process:
                self.status = "Stopped" if code == 0 else "Error"
                if code != 0:
                    self.last_error = f"Server exited with code {code}"


def _create_icon_image(running: bool):
    if Image is None or ImageDraw is None:
        return None
    system = platform.system().lower()
    if system != "darwin":
        app_icon = _asset_path("assets", "app_icon_512.png") or _asset_path("assets", "app_icon.png")
        if app_icon is not None:
            return Image.open(app_icon).convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)

    scale = 4
    size = 64
    canvas = size * scale
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    if system == "darwin":
        # macOS treats this as a template image; only the alpha shape matters.
        ink = (0, 0, 0, 255)
        draw.arc((11 * scale, 9 * scale, 48 * scale, 55 * scale), 42, 318, fill=ink, width=7 * scale)
        draw.rounded_rectangle((34 * scale, 26 * scale, 51 * scale, 33 * scale), radius=3 * scale, fill=ink)
        for cx, cy, radius in ((48, 20, 3), (56, 27, 2), (48, 40, 2)):
            draw.ellipse(
                ((cx - radius) * scale, (cy - radius) * scale, (cx + radius) * scale, (cy + radius) * scale),
                fill=ink,
            )
    else:
        primary = (16, 145, 255, 255) if running else (113, 124, 137, 255)
        accent = (30, 216, 229, 255) if running else (151, 160, 171, 255)
        warm = (255, 169, 76, 255) if running else (151, 160, 171, 255)
        draw.arc((10 * scale, 8 * scale, 48 * scale, 56 * scale), 42, 318, fill=primary, width=8 * scale)
        draw.rounded_rectangle((34 * scale, 26 * scale, 52 * scale, 34 * scale), radius=3 * scale, fill=accent)
        draw.rounded_rectangle((28 * scale, 39 * scale, 45 * scale, 46 * scale), radius=3 * scale, fill=warm)
        for cx, cy, radius in ((48, 20, 3), (56, 27, 2), (49, 41, 2)):
            draw.ellipse(
                ((cx - radius) * scale, (cy - radius) * scale, (cx + radius) * scale, (cy + radius) * scale),
                fill=accent,
            )

    return image.resize((size, size), Image.Resampling.LANCZOS)


class TrayLauncher:
    def __init__(self) -> None:
        if pystray is None:
            raise RuntimeError("pystray is not available")
        self.server = ServerManager()
        self.autostart = AutoStartManager(self.server.app_support_dir)
        self.icon = pystray.Icon(
            "code-bridge-server",
            _create_icon_image(False),
            APP_NAME,
            self._build_menu(),
        )
        self._closed = threading.Event()

    def _build_menu(self):
        return pystray.Menu(
            pystray.MenuItem(lambda item: f"Status: {self.server.status}", None, enabled=False),
            pystray.MenuItem(lambda item: f"API: {self.server.api_url()}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Start Server", self._start_server, enabled=lambda item: not self.server.is_running()),
            pystray.MenuItem("Stop Server", self._stop_server, enabled=lambda item: self.server.is_running()),
            pystray.MenuItem("Open Dashboard", self._open_dashboard),
            pystray.MenuItem("Show QR Pairing", self._show_qr_pairing),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Start at Login",
                self._toggle_autostart,
                checked=lambda item: self.autostart.is_enabled(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

    def run(self) -> None:
        self._install_signal_handlers()
        self.icon.run(setup=self._setup)

    def _setup(self, icon) -> None:
        self._apply_macos_menu_bar_title()
        self.server.start()
        threading.Thread(target=self._refresh_loop, daemon=True).start()

    def _refresh_loop(self) -> None:
        while not self._closed.is_set():
            self.icon.icon = _create_icon_image(self.server.is_running())
            self._apply_macos_menu_bar_title()
            self.icon.title = f"{APP_NAME} - {self.server.status}"
            self.icon.update_menu()
            time.sleep(1.5)

    def _apply_macos_menu_bar_title(self) -> None:
        if platform.system().lower() != "darwin":
            return
        status_item = getattr(self.icon, "_status_item", None)
        if status_item is None:
            return
        button = status_item.button()
        if button is None:
            return
        button.setImage_(None)
        button.setTitle_(MACOS_MENU_BAR_TITLE)
        button.setToolTip_(f"{APP_NAME} - {self.server.status}")

    def _start_server(self, icon=None, item=None) -> None:
        self.server.start()
        self.icon.update_menu()

    def _stop_server(self, icon=None, item=None) -> None:
        threading.Thread(target=self.server.stop, daemon=True).start()
        self.icon.update_menu()

    def _open_dashboard(self, icon=None, item=None) -> None:
        self._ensure_server_then_open(self.server.dashboard_url)

    def _show_qr_pairing(self, icon=None, item=None) -> None:
        self._ensure_server_then_open(self.server.pair_url)

    def _ensure_server_then_open(self, url_factory) -> None:
        def worker() -> None:
            if not self.server.is_running():
                self.server.start()
                time.sleep(2.0)
            webbrowser.open(url_factory())

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_autostart(self, icon=None, item=None) -> None:
        try:
            self.autostart.set_enabled(not self.autostart.is_enabled())
        except OSError as exc:
            self.server.last_error = str(exc)
            _write_launcher_log(f"Failed to update autostart: {exc}")
        self.icon.update_menu()

    def _quit(self, icon=None, item=None) -> None:
        self._closed.set()
        self.server.stop()
        self.icon.stop()

    def _install_signal_handlers(self) -> None:
        def handle_signal(signum, frame) -> None:
            self._quit()

        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(signum, handle_signal)
            except ValueError:
                pass


def main() -> None:
    if "--server-child" in sys.argv:
        server_dir = _server_dir()
        os.chdir(server_dir)
        sys.path.insert(0, str(server_dir))
        from server_cli import main as server_main

        sys.argv = [sys.argv[0], "--restart"]
        server_main()
        return

    # Headless autostart control, so installers and scripts can register
    # the login item without opening a tray.
    if {"--enable-autostart", "--disable-autostart"} & set(sys.argv):
        enable = "--enable-autostart" in sys.argv
        AutoStartManager(_app_support_dir()).set_enabled(enable)
        print(f"Start at login: {'enabled' if enable else 'disabled'}")
        return

    if "--autostart-status" in sys.argv:
        print("enabled" if AutoStartManager(_app_support_dir()).is_enabled() else "disabled")
        return

    if pystray is None:
        run_headless_launcher()
        return

    TrayLauncher().run()


def run_headless_launcher() -> None:
    server = ServerManager()
    server.start()
    time.sleep(2.5)
    webbrowser.open(server.dashboard_url())
    try:
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
