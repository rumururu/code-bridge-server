"""Project management for Code Bridge."""

import asyncio
import json
import socket
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from database import get_project_db


class BuildStatus(Enum):
    """Web build status."""

    NONE = "none"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"


class ProjectType(Enum):
    """Supported project types."""

    FLUTTER = "flutter"
    NEXTJS = "nextjs"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "ProjectType":
        """Convert string to ProjectType."""
        mapping = {
            "flutter": cls.FLUTTER,
            "nextjs": cls.NEXTJS,
            "next": cls.NEXTJS,
            "next.js": cls.NEXTJS,
        }
        return mapping.get(value.lower(), cls.UNKNOWN)


@dataclass
class DevServerProcess:
    """Running dev server process."""

    process: subprocess.Popen
    port: int
    command: str


@dataclass
class DeviceRunProcess:
    """Running Flutter app process for a physical Android device."""

    process: subprocess.Popen
    device_id: str
    command: list[str]
    log_path: str


@dataclass
class BuildInfo:
    """Web build information."""

    status: BuildStatus
    build_path: str | None = None
    error_message: str | None = None
    project_type: str | None = None  # Track which type was built


@dataclass
class ProjectManager:
    """Manages project lifecycle and dev servers."""

    _running_servers: dict[str, DevServerProcess] = field(default_factory=dict)
    _running_device_runs: dict[str, DeviceRunProcess] = field(default_factory=dict)
    _last_device_run_logs: dict[str, str] = field(default_factory=dict)
    _build_info: dict[str, BuildInfo] = field(default_factory=dict)

    def get_all_projects(self) -> list[dict[str, Any]]:
        """Get all configured projects with their status."""
        db = get_project_db()
        projects = []

        for project in db.get_all():
            name = project.get("name", "")
            detected_port = self.get_server_port(name)
            is_running = detected_port is not None
            dev_server = project.get("dev_server") or {}

            projects.append(
                {
                    "name": name,
                    "path": project.get("path", ""),
                    "type": project.get("type", "unknown"),
                    "dev_server": {
                        "port": detected_port or dev_server.get("port"),
                        "running": is_running,
                    },
                }
            )

        return projects

    def get_project(self, name: str) -> dict[str, Any] | None:
        """Get specific project info."""
        db = get_project_db()
        project = db.get(name)

        if project is None:
            return None

        detected_port = self.get_server_port(name)
        is_running = detected_port is not None
        dev_server = project.get("dev_server") or {}

        return {
            "name": project.get("name", ""),
            "path": project.get("path", ""),
            "type": project.get("type", "unknown"),
            "dev_server": {
                "port": detected_port or dev_server.get("port"),
                "command": dev_server.get("command"),
                "running": is_running,
            },
        }

    async def start_dev_server(self, name: str) -> dict[str, Any]:
        """Start dev server for a project."""
        if name in self._running_servers:
            process = self._running_servers[name].process
            if process.poll() is None:
                return {
                    "success": True,
                    "message": f"Dev server for {name} is already running",
                    "port": self._running_servers[name].port,
                }
            del self._running_servers[name]

        existing_port = self.detect_running_server_port(name)
        if existing_port is not None:
            return {
                "success": True,
                "message": f"Dev server for {name} is already running",
                "port": existing_port,
            }

        db = get_project_db()
        project = db.get(name)

        if project is None:
            return {"success": False, "message": f"Project {name} not found"}

        dev_server = project.get("dev_server") or {}
        command = dev_server.get("command")
        port = dev_server.get("port")
        project_path = project.get("path")
        project_type = ProjectType.from_string(project.get("type", ""))

        if not project_path:
            return {
                "success": False,
                "message": f"Invalid dev server configuration for {name}",
            }

        if not isinstance(command, str) or not command.strip():
            command = self._infer_default_dev_server_command(
                project_path=str(project_path),
                project_type=project_type,
            )
            # Persist inferred command so next start is deterministic.
            if command:
                try:
                    db.update(name, {"dev_server": {"command": command}})
                except Exception:
                    pass

        if not command:
            return {
                "success": False,
                "message": (
                    f"No dev server command configured for {name}. "
                    "Run the project's dev server manually and press refresh, "
                    "or configure a command."
                ),
            }

        if not Path(project_path).exists():
            return {"success": False, "message": f"Project path does not exist: {project_path}"}

        try:
            # Start the dev server process
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            resolved_port: int | None = port if isinstance(port, int) else None
            if resolved_port is None:
                resolved_port = await self._wait_for_project_server_port(
                    project_path,
                    project_type,
                    process=process,
                )

            if resolved_port is None:
                error_hint = self._extract_process_error(process)
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
                message = f"Could not detect dev server port for {name}"
                if error_hint:
                    message = f"{message}: {error_hint}"
                return {
                    "success": False,
                    "message": message,
                }

            self._running_servers[name] = DevServerProcess(
                process=process,
                port=resolved_port,
                command=command,
            )

            return {
                "success": True,
                "message": f"Dev server started for {name}",
                "port": resolved_port,
                "pid": process.pid,
            }

        except Exception as e:
            return {"success": False, "message": f"Failed to start dev server: {str(e)}"}

    def _infer_default_dev_server_command(
        self,
        project_path: str,
        project_type: ProjectType,
    ) -> str | None:
        """Infer a default dev-server command from project files."""
        path = Path(project_path)
        if not path.exists() or not path.is_dir():
            return None

        if project_type == ProjectType.FLUTTER and (path / "pubspec.yaml").exists():
            # Flutter projects are handled via device preview flow, not web dev-server.
            return None

        package_json_path = path / "package.json"
        if package_json_path.exists():
            package_data = self._load_package_json(package_json_path)
            scripts = package_data.get("scripts", {}) if isinstance(package_data, dict) else {}
            package_manager = str(package_data.get("packageManager", "")).lower() if isinstance(package_data, dict) else ""
            runner = self._guess_js_runner(package_manager)

            if isinstance(scripts, dict):
                if "dev" in scripts:
                    return self._build_js_script_command(runner, "dev")
                if "start" in scripts:
                    return self._build_js_script_command(runner, "start")

        if project_type == ProjectType.NEXTJS:
            return "npm run dev"

        return None

    def _load_package_json(self, package_json_path: Path) -> dict[str, Any]:
        """Safely read and parse package.json."""
        try:
            return json.loads(package_json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _guess_js_runner(self, package_manager: str) -> str:
        """Guess script runner from packageManager field."""
        if package_manager.startswith("pnpm"):
            return "pnpm"
        if package_manager.startswith("yarn"):
            return "yarn"
        if package_manager.startswith("bun"):
            return "bun"
        return "npm"

    def _build_js_script_command(self, runner: str, script: str) -> str:
        """Build a script command for the detected JS package manager."""
        if runner == "pnpm":
            return f"pnpm {script}"
        if runner == "yarn":
            return f"yarn {script}"
        if runner == "bun":
            return f"bun run {script}"
        return f"npm run {script}"

    async def stop_dev_server(self, name: str) -> dict[str, Any]:
        """Stop dev server for a project."""
        if name not in self._running_servers:
            return {"success": True, "message": f"Dev server for {name} is not running"}

        try:
            server = self._running_servers[name]
            server.process.terminate()

            # Wait for graceful termination
            try:
                server.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.process.kill()

            del self._running_servers[name]

            return {"success": True, "message": f"Dev server stopped for {name}"}

        except Exception as e:
            return {"success": False, "message": f"Failed to stop dev server: {str(e)}"}

    def _sanitize_log_name(self, value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)
        safe = safe.strip("_")
        return safe or "project"

    def _device_run_log_path(self, project_name: str, device_id: str) -> Path:
        safe_project = self._sanitize_log_name(project_name)
        safe_device = self._sanitize_log_name(device_id)
        return Path("/tmp") / f"code_bridge_device_run_{safe_project}_{safe_device}.log"

    def _read_log_tail(
        self,
        log_path: str | Path,
        max_lines: int = 120,
        max_chars: int = 16000,
    ) -> str:
        path = Path(log_path)
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

        lines = text.splitlines()
        tail = "\n".join(lines[-max_lines:]) if lines else text
        if len(tail) > max_chars:
            tail = tail[-max_chars:]
        return tail

    def _active_device_run(self, name: str) -> DeviceRunProcess | None:
        info = self._running_device_runs.get(name)
        if info is None:
            return None
        if info.process.poll() is None:
            return info
        del self._running_device_runs[name]
        return None

    async def run_project_on_device(
        self,
        name: str,
        device_id: str,
        restart: bool = False,
    ) -> dict[str, Any]:
        """Run a Flutter project on a specific Android device via `flutter run`."""
        normalized_device_id = device_id.strip()
        if not normalized_device_id:
            return {"success": False, "message": "Device ID is required"}

        db = get_project_db()
        project = db.get(name)
        if not project:
            return {"success": False, "message": f"Project {name} not found"}

        project_type = ProjectType.from_string(project.get("type", ""))
        if project_type != ProjectType.FLUTTER:
            return {"success": False, "message": "Only Flutter projects support device run"}

        project_path = project.get("path")
        if not project_path or not Path(project_path).exists():
            return {"success": False, "message": f"Project path does not exist: {project_path}"}

        existing = self._active_device_run(name)
        if existing and existing.device_id == normalized_device_id and not restart:
            return {
                "success": True,
                "message": "Flutter app is already running on selected device",
                "pid": existing.process.pid,
                "device_id": existing.device_id,
                "log_path": existing.log_path,
                "already_running": True,
            }

        if existing is not None:
            try:
                existing.process.terminate()
                existing.process.wait(timeout=5)
            except Exception:
                try:
                    existing.process.kill()
                    existing.process.wait(timeout=2)
                except Exception:
                    pass
            self._running_device_runs.pop(name, None)

        command = [
            "flutter",
            "run",
            "-d",
            normalized_device_id,
            "--machine",
            "--target",
            "lib/main.dart",
        ]
        log_path = self._device_run_log_path(name, normalized_device_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            log_path.write_text("", encoding="utf-8")
        except Exception:
            pass

        try:
            with log_path.open("ab") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=project_path,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
        except FileNotFoundError:
            return {"success": False, "message": "Flutter CLI not found on server"}
        except Exception as exc:
            return {"success": False, "message": f"Failed to start flutter run: {exc}"}

        await asyncio.sleep(2.0)
        if process.poll() is not None:
            tail = self._read_log_tail(log_path, max_lines=80, max_chars=4000)
            lines = [line.strip() for line in tail.splitlines() if line.strip()]
            summary = lines[-1] if lines else "flutter run exited immediately"
            return {
                "success": False,
                "message": summary,
                "log_tail": tail,
            }

        info = DeviceRunProcess(
            process=process,
            device_id=normalized_device_id,
            command=command,
            log_path=str(log_path),
        )
        self._running_device_runs[name] = info
        self._last_device_run_logs[name] = info.log_path

        return {
            "success": True,
            "message": f"Started flutter run on {normalized_device_id}",
            "pid": process.pid,
            "device_id": normalized_device_id,
            "log_path": info.log_path,
        }

    async def stop_project_on_device(self, name: str) -> dict[str, Any]:
        """Stop a running Flutter device process for the project."""
        info = self._active_device_run(name)
        if info is None:
            return {"success": True, "message": "No running device process"}

        try:
            info.process.terminate()
            info.process.wait(timeout=5)
        except Exception:
            try:
                info.process.kill()
                info.process.wait(timeout=2)
            except Exception:
                pass
        finally:
            self._running_device_runs.pop(name, None)

        return {"success": True, "message": "Stopped Flutter device run"}

    def get_device_run_log(self, name: str, lines: int = 120) -> dict[str, Any]:
        """Return latest Flutter device-run log tail for the project."""
        capped_lines = max(10, min(lines, 500))
        active = self._active_device_run(name)
        log_path = active.log_path if active is not None else self._last_device_run_logs.get(name)
        if not log_path:
            return {
                "running": False,
                "device_id": None,
                "log_path": None,
                "log_tail": "",
            }

        return {
            "running": active is not None,
            "device_id": active.device_id if active is not None else None,
            "log_path": log_path,
            "log_tail": self._read_log_tail(log_path, max_lines=capped_lines),
        }

    def is_server_running(self, name: str) -> bool:
        """Check if dev server is running."""
        return name in self._running_servers

    def get_server_port(self, name: str) -> int | None:
        """Get running server port."""
        if name in self._running_servers:
            process = self._running_servers[name].process
            if process.poll() is None:
                return self._running_servers[name].port
            del self._running_servers[name]

        return self.detect_running_server_port(name)

    def detect_running_server_port(self, name: str) -> int | None:
        """Detect port for an externally running project dev server."""
        db = get_project_db()
        project = db.get(name)
        if not project:
            return None

        project_path = project.get("path")
        if not project_path:
            return None

        project_type = ProjectType.from_string(project.get("type", ""))
        detected_port = self._detect_port_for_project(
            project_path=str(project_path),
            project_type=project_type,
        )
        if detected_port is not None:
            return detected_port

        configured_port = (project.get("dev_server") or {}).get("port")
        if isinstance(configured_port, int) and self._is_local_port_open(configured_port):
            return configured_port
        return None

    async def _wait_for_project_server_port(
        self,
        project_path: str,
        project_type: ProjectType,
        process: subprocess.Popen | None = None,
        timeout_seconds: float = 15.0,
    ) -> int | None:
        """Wait until a dev server for the project starts listening."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            if process is not None and process.poll() is not None:
                return None

            detected_port = self._detect_port_for_project(project_path, project_type)
            if detected_port is not None:
                return detected_port
            await asyncio.sleep(0.5)
        return None

    def _extract_process_error(self, process: subprocess.Popen) -> str | None:
        """Extract a concise stderr hint from a failed process."""
        try:
            if process.poll() is None:
                return None

            if process.stderr is None:
                return None

            raw = process.stderr.read()
            if not raw:
                return None

            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                return None

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                return None

            # Prefer the last non-empty line for concise feedback.
            return lines[-1][:240]
        except Exception:
            return None

    def _detect_port_for_project(self, project_path: str, project_type: ProjectType) -> int | None:
        """Detect a listening port for a project by matching process CWD."""
        try:
            target = Path(project_path).resolve()
        except Exception:
            return None

        bridge_port = self._get_bridge_port()
        listeners = self._list_listening_processes()

        best_score = -1
        best_port: int | None = None

        for pid, listener in listeners.items():
            command = listener.get("command", "").lower()
            if not self._is_candidate_command(command, project_type):
                continue

            cwd_path = self._get_process_cwd(pid)
            if cwd_path is None:
                continue

            match_score = self._match_project_path_score(target, cwd_path)
            if match_score < 0:
                continue

            ports = [
                port
                for port in listener.get("ports", set())
                if isinstance(port, int) and port >= 1024 and port != bridge_port
            ]
            if not ports:
                continue

            selected_port = self._select_preferred_port(ports)
            if match_score > best_score:
                best_score = match_score
                best_port = selected_port

        return best_port

    def _get_bridge_port(self) -> int | None:
        try:
            from config import get_config

            return get_config().port
        except Exception:
            return None

    def _list_listening_processes(self) -> dict[int, dict[str, Any]]:
        """List TCP listening processes from lsof."""
        try:
            result = subprocess.run(
                ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fpcn"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}

        listeners: dict[int, dict[str, Any]] = {}
        current_pid: int | None = None

        for raw in result.stdout.splitlines():
            if not raw:
                continue

            field = raw[0]
            value = raw[1:]

            if field == "p":
                if not value.isdigit():
                    current_pid = None
                    continue
                current_pid = int(value)
                listeners.setdefault(current_pid, {"command": "", "ports": set()})
            elif field == "c" and current_pid is not None:
                listeners[current_pid]["command"] = value
            elif field == "n" and current_pid is not None:
                port = self._extract_port(value)
                if port is not None:
                    listeners[current_pid]["ports"].add(port)

        return listeners

    def _get_process_cwd(self, pid: int) -> Path | None:
        """Get process current working directory via lsof."""
        try:
            result = subprocess.run(
                ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        for line in result.stdout.splitlines():
            if line.startswith("n") and len(line) > 1:
                try:
                    return Path(line[1:]).resolve()
                except Exception:
                    return None
        return None

    def _match_project_path_score(self, project_path: Path, cwd_path: Path) -> int:
        """Score CWD matching against project path."""
        if cwd_path == project_path:
            return 3
        if project_path in cwd_path.parents:
            return 2
        if cwd_path in project_path.parents:
            return 1
        return -1

    def _is_candidate_command(self, command: str, project_type: ProjectType) -> bool:
        """Check if command likely hosts a dev server."""
        normalized = command.lower()

        node_like_tokens = ("node", "npm", "pnpm", "yarn", "bun", "next", "vite", "deno")
        flutter_like_tokens = ("flutter", "dart")
        python_like_tokens = ("python", "uvicorn", "gunicorn")

        if project_type == ProjectType.FLUTTER:
            return any(token in normalized for token in flutter_like_tokens)
        if project_type == ProjectType.NEXTJS:
            return any(token in normalized for token in node_like_tokens)

        return (
            any(token in normalized for token in node_like_tokens)
            or any(token in normalized for token in flutter_like_tokens)
            or any(token in normalized for token in python_like_tokens)
        )

    def _select_preferred_port(self, ports: list[int]) -> int:
        """Select the most likely app port from candidates."""
        preferred_ports = (3000, 5173, 4200, 4173, 8081, 8082, 8000, 5000)
        unique_ports = sorted(set(ports))
        for preferred in preferred_ports:
            if preferred in unique_ports:
                return preferred
        return unique_ports[0]

    def _extract_port(self, endpoint: str) -> int | None:
        """Extract local port from lsof endpoint field."""
        if ":" not in endpoint:
            return None

        local = endpoint.split("->", 1)[0]
        candidate = local.rsplit(":", 1)[-1]
        return int(candidate) if candidate.isdigit() else None

    def _is_local_port_open(self, port: int) -> bool:
        """Check if localhost TCP port is accepting connections."""
        for host in ("127.0.0.1", "::1"):
            try:
                family = socket.AF_INET6 if host == "::1" else socket.AF_INET
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.2)
                    if sock.connect_ex((host, port)) == 0:
                        return True
            except OSError:
                continue
        return False

    async def build_flutter_web(self, name: str) -> dict[str, Any]:
        """Build web app (Flutter or Next.js).

        Supports:
        - Flutter: `flutter build web --release` → build/web/
        - Next.js: `npm run build` → .next/standalone/ or out/
        """
        project = self.get_project(name)
        if not project:
            return {"success": False, "message": f"Project {name} not found"}

        project_type = ProjectType.from_string(project.get("type", ""))
        project_path = project.get("path")

        if not project_path or not Path(project_path).exists():
            return {"success": False, "message": f"Project path does not exist: {project_path}"}

        # Mark as building
        self._build_info[name] = BuildInfo(status=BuildStatus.BUILDING, project_type=project_type.value)

        try:
            if project_type == ProjectType.FLUTTER:
                return await self._build_flutter(name, project_path)
            elif project_type == ProjectType.NEXTJS:
                return await self._build_nextjs(name, project_path)
            else:
                self._build_info[name] = BuildInfo(
                    status=BuildStatus.ERROR,
                    error_message=f"Unsupported project type: {project.get('type')}",
                )
                return {"success": False, "message": f"Unsupported project type: {project.get('type')}"}

        except Exception as e:
            error_msg = str(e)
            self._build_info[name] = BuildInfo(
                status=BuildStatus.ERROR,
                error_message=error_msg,
            )
            return {"success": False, "message": f"Build failed: {error_msg}"}

    async def _build_flutter(self, name: str, project_path: str) -> dict[str, Any]:
        """Build Flutter web app."""
        try:
            process = await asyncio.create_subprocess_exec(
                "flutter",
                "build",
                "web",
                "--release",
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                self._build_info[name] = BuildInfo(
                    status=BuildStatus.ERROR,
                    error_message=error_msg or "Build failed",
                    project_type=ProjectType.FLUTTER.value,
                )
                return {"success": False, "message": error_msg or "Build failed"}

            build_path = str(Path(project_path) / "build" / "web")
            self._build_info[name] = BuildInfo(
                status=BuildStatus.READY,
                build_path=build_path,
                project_type=ProjectType.FLUTTER.value,
            )

            return {
                "success": True,
                "message": "Build completed",
                "build_path": build_path,
            }

        except FileNotFoundError:
            error_msg = "Flutter CLI not found. Is Flutter installed?"
            self._build_info[name] = BuildInfo(
                status=BuildStatus.ERROR,
                error_message=error_msg,
                project_type=ProjectType.FLUTTER.value,
            )
            return {"success": False, "message": error_msg}

    async def _build_nextjs(self, name: str, project_path: str) -> dict[str, Any]:
        """Build Next.js app.

        Runs `npm run build`. Output depends on next.config.js:
        - Static export (output: 'export'): out/
        - Standalone (output: 'standalone'): .next/standalone/
        - Default: .next/ (requires next start)
        """
        try:
            # Run npm run build
            process = await asyncio.create_subprocess_exec(
                "npm",
                "run",
                "build",
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                # Also include stdout as Next.js often puts errors there
                if not error_msg:
                    error_msg = stdout.decode("utf-8", errors="replace").strip()
                self._build_info[name] = BuildInfo(
                    status=BuildStatus.ERROR,
                    error_message=error_msg or "Build failed",
                    project_type=ProjectType.NEXTJS.value,
                )
                return {"success": False, "message": error_msg or "Build failed"}

            # Determine build output path
            # Priority: out/ (static export) > .next/standalone/ > .next/
            out_path = Path(project_path) / "out"
            standalone_path = Path(project_path) / ".next" / "standalone"
            next_path = Path(project_path) / ".next"

            if out_path.exists() and (out_path / "index.html").exists():
                build_path = str(out_path)
            elif standalone_path.exists():
                build_path = str(standalone_path)
            elif next_path.exists():
                # For non-static Next.js, we'll need to serve via next start
                # For now, mark as ready but note it needs server
                build_path = str(next_path)
            else:
                self._build_info[name] = BuildInfo(
                    status=BuildStatus.ERROR,
                    error_message="Build completed but no output directory found",
                    project_type=ProjectType.NEXTJS.value,
                )
                return {"success": False, "message": "Build completed but no output directory found"}

            self._build_info[name] = BuildInfo(
                status=BuildStatus.READY,
                build_path=build_path,
                project_type=ProjectType.NEXTJS.value,
            )

            return {
                "success": True,
                "message": "Build completed",
                "build_path": build_path,
            }

        except FileNotFoundError:
            error_msg = "npm not found. Is Node.js installed?"
            self._build_info[name] = BuildInfo(
                status=BuildStatus.ERROR,
                error_message=error_msg,
                project_type=ProjectType.NEXTJS.value,
            )
            return {"success": False, "message": error_msg}

    def get_build_status(self, name: str) -> dict[str, Any]:
        """Get build status for a project."""
        info = self._build_info.get(name)

        if not info:
            return {
                "status": BuildStatus.NONE.value,
                "build_path": None,
                "error_message": None,
                "project_type": None,
            }

        return {
            "status": info.status.value,
            "build_path": info.build_path,
            "error_message": info.error_message,
            "project_type": info.project_type,
        }

    def get_build_path(self, name: str) -> str | None:
        """Get build path for a project if ready."""
        info = self._build_info.get(name)
        if info and info.status == BuildStatus.READY:
            return info.build_path
        return None


# Global project manager instance
_project_manager: ProjectManager | None = None


def get_project_manager() -> ProjectManager:
    """Get global project manager instance."""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager
