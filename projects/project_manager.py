"""Project management for Code Bridge."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from core.database import get_project_db
from devices.scrcpy_manager import get_scrcpy_manager
from .project_dev_server_start import resolve_dev_server_start_plan, spawn_dev_server_process
from .project_device_logs import device_run_log_path, read_log_tail
from .project_device_run_plan import resolve_device_run_plan
from .project_device_run_service import (
    extract_vm_service_uri_from_log,
    start_flutter_run_process,
    summarize_flutter_run_exit,
)
from .project_models import DevServerProcess, DeviceRunProcess, ProjectType
from .project_process_utils import is_process_running, terminate_process_safely
from .project_query_service import (
    build_project_list_view,
    build_single_project_view,
    detect_project_running_server_port,
)
from .project_server_detection import (
    detect_port_for_project,
    list_listening_processes,
    list_process_cwds,
)
from .project_server_process import extract_process_error, wait_for_project_server_port


@dataclass
class ProjectManager:
    """Manages project lifecycle and dev servers."""

    _project_db_factory: Callable[[], Any] = get_project_db
    _running_servers: dict[str, DevServerProcess] = field(default_factory=dict)
    _running_device_runs: dict[str, DeviceRunProcess] = field(default_factory=dict)
    _last_device_run_logs: dict[str, str] = field(default_factory=dict)

    def _project_db(self) -> Any:
        return self._project_db_factory()

    def get_all_projects(self) -> list[dict[str, Any]]:
        """Get all configured projects with their status.

        Optimized to call lsof commands once for all projects instead of per-project.
        """
        db = self._project_db()
        projects = db.get_all()

        # Cache listeners and CWDs once for all projects (expensive lsof calls)
        cached_listeners = list_listening_processes()
        cached_cwds = list_process_cwds(list(cached_listeners.keys()))

        def get_server_port_cached(name: str) -> int | None:
            """Get server port using cached listeners and CWD maps."""
            if name in self._running_servers:
                process = self._running_servers[name].process
                if is_process_running(process):
                    return self._running_servers[name].port
                del self._running_servers[name]
            # Use cached listeners and CWDs for detection
            project = db.get(name)
            if not project:
                return None
            project_path = project.get("path")
            if not project_path:
                return None
            project_type = ProjectType.from_string(project.get("type", ""))
            return detect_port_for_project(
                str(project_path),
                project_type,
                listeners=cached_listeners,
                cwd_map=cached_cwds,
            )

        return build_project_list_view(
            projects,
            get_server_port=get_server_port_cached,
            is_managed_server=self.is_server_running,
        )

    def get_project(self, name: str) -> dict[str, Any] | None:
        """Get specific project info."""
        db = self._project_db()
        return build_single_project_view(
            name,
            db.get(name),
            get_server_port=self.get_server_port,
            is_managed_server=self.is_server_running,
        )

    async def start_dev_server(self, name: str) -> dict[str, Any]:
        """Start dev server for a project."""
        if name in self._running_servers:
            process = self._running_servers[name].process
            if is_process_running(process):
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

        db = self._project_db()
        project = db.get(name)

        if project is None:
            return {"success": False, "message": f"Project {name} not found"}

        plan_result = resolve_dev_server_start_plan(name, project, project_db=db)
        if not plan_result.success or plan_result.plan is None:
            return {"success": False, "message": plan_result.error_message or "Invalid dev server configuration"}
        plan = plan_result.plan

        try:
            process = spawn_dev_server_process(plan.command, plan.project_path)

            resolved_port: int | None = plan.port_hint
            if resolved_port is None:
                resolved_port = await wait_for_project_server_port(
                    plan.project_path,
                    plan.project_type,
                    detect_port=self._detect_port_for_project,
                    process=process,
                )

            if resolved_port is None:
                error_hint = extract_process_error(process)
                terminate_process_safely(process, terminate_timeout=5.0, kill_timeout=2.0)
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
                command=plan.command,
            )

            return {
                "success": True,
                "message": f"Dev server started for {name}",
                "port": resolved_port,
                "pid": process.pid,
            }

        except OSError as e:
            return {"success": False, "message": f"Failed to start dev server: {str(e)}"}

    async def stop_dev_server(self, name: str) -> dict[str, Any]:
        """Stop dev server for a project."""
        if name not in self._running_servers:
            return {"success": True, "message": f"Dev server for {name} is not running"}

        try:
            server = self._running_servers[name]
            terminate_process_safely(server.process, terminate_timeout=5.0, kill_timeout=2.0)

            del self._running_servers[name]

            return {"success": True, "message": f"Dev server stopped for {name}"}

        except OSError as e:
            return {"success": False, "message": f"Failed to stop dev server: {str(e)}"}

    def _active_device_run(self, name: str) -> DeviceRunProcess | None:
        info = self._running_device_runs.get(name)
        if info is None:
            return None
        if is_process_running(info.process):
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
        db = self._project_db()
        project = db.get(name)
        plan_result = resolve_device_run_plan(name, device_id, project)
        if not plan_result.success or plan_result.plan is None:
            return {"success": False, "message": plan_result.error_message or "Invalid device run request"}
        plan = plan_result.plan

        existing = self._active_device_run(name)
        if existing and existing.device_id == plan.device_id and not restart:
            return {
                "success": True,
                "message": "Flutter app is already running on selected device",
                "pid": existing.process.pid,
                "device_id": existing.device_id,
                "log_path": existing.log_path,
                "already_running": True,
            }

        if existing is not None:
            terminate_process_safely(existing.process, terminate_timeout=5.0, kill_timeout=2.0)
            self._running_device_runs.pop(name, None)

        log_path = device_run_log_path(name, plan.device_id)
        start_result = start_flutter_run_process(
            plan.project_path,
            device_id=plan.device_id,
            log_path=log_path,
        )
        if not start_result.success or start_result.process is None:
            return {"success": False, "message": start_result.error_message or "Failed to start flutter run"}
        process = start_result.process

        await asyncio.sleep(2.0)
        if process.poll() is not None:
            summary, tail = summarize_flutter_run_exit(log_path)
            return {
                "success": False,
                "message": summary,
                "log_tail": tail,
            }

        # Try to extract VM Service URI from log (may need more time)
        vm_service_uri = extract_vm_service_uri_from_log(log_path)

        info = DeviceRunProcess(
            process=process,
            device_id=plan.device_id,
            command=start_result.command,
            log_path=str(log_path),
            vm_service_uri=vm_service_uri,
        )
        self._running_device_runs[name] = info
        self._last_device_run_logs[name] = info.log_path

        return {
            "success": True,
            "message": f"Started flutter run on {plan.device_id}",
            "pid": process.pid,
            "device_id": plan.device_id,
            "log_path": info.log_path,
            "vm_service_uri": vm_service_uri,
        }

    async def open_web_preview_on_device(
        self,
        name: str,
        device_id: str,
        *,
        width: int | None = None,
        height: int | None = None,
        density: int | None = None,
        reset_to_default: bool = False,
    ) -> dict[str, Any]:
        """Open a web project's running dev server in Chrome on an Android emulator."""
        project = self.get_project(name)
        if not project:
            return {"success": False, "message": f"Project {name} not found"}

        project_type = ProjectType.from_string(project.get("type", ""))
        if project_type == ProjectType.FLUTTER:
            return {
                "success": False,
                "message": "Flutter projects already use direct device mirroring",
            }

        port = self.get_server_port(name)
        if port is None:
            start_result = await self.start_dev_server(name)
            if not start_result.get("success"):
                return start_result
            port = start_result.get("port")

        if port is None:
            return {
                "success": False,
                "message": f"Dev server is not running for {name}",
            }

        scrcpy_manager = get_scrcpy_manager()
        resolved_device_id, ensure_error = await scrcpy_manager.ensure_emulator_ready(
            device_id,
        )
        if ensure_error:
            return {"success": False, "message": ensure_error}
        if not resolved_device_id:
            return {
                "success": False,
                "message": "Could not resolve Android emulator device",
            }

        display_override_applied = (
            reset_to_default
            or width is not None
            or height is not None
            or density is not None
        )
        if display_override_applied:
            display_error = await scrcpy_manager.configure_emulator_display(
                resolved_device_id,
                width=width,
                height=height,
                density=density,
                reset_to_default=reset_to_default,
            )
            if display_error:
                return {"success": False, "message": display_error}
            await asyncio.sleep(1.0)

        # Use localhost + adb reverse instead of 10.0.2.2. Firebase Auth and
        # other OAuth providers auto-authorize "localhost" but refuse arbitrary
        # hosts like 10.0.2.2, so this lets the dev server's auth flow run in
        # real Chrome on the emulator without touching the project config.
        preview_url = f"http://localhost:{port}"
        await scrcpy_manager.setup_reverse_port(resolved_device_id, port)
        open_error = await scrcpy_manager.open_url_in_browser(
            resolved_device_id,
            preview_url,
        )
        if open_error:
            return {"success": False, "message": open_error}

        return {
            "success": True,
            "message": f"Opened {name} preview on {resolved_device_id}",
            "device_id": resolved_device_id,
            "preview_url": preview_url,
            "port": port,
            "display_override_applied": display_override_applied,
            "display_config": {
                "width": width,
                "height": height,
                "density": density,
                "reset_to_default": reset_to_default,
            },
        }

    async def stop_project_on_device(self, name: str) -> dict[str, Any]:
        """Stop a running Flutter device process for the project."""
        info = self._active_device_run(name)
        if info is None:
            return {"success": True, "message": "No running device process"}

        try:
            terminate_process_safely(info.process, terminate_timeout=5.0, kill_timeout=2.0)
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
            "log_tail": read_log_tail(log_path, max_lines=capped_lines),
        }

    def get_vm_service_uri(self, name: str) -> str | None:
        """Get VM Service URI for a running Flutter app.

        If URI is not yet available, try to extract from log file.
        """
        active = self._active_device_run(name)
        if active is None:
            return None

        # Return cached URI if available
        if active.vm_service_uri:
            return active.vm_service_uri

        # Try to extract from log file (app may have started after initial check)
        uri = extract_vm_service_uri_from_log(active.log_path)
        if uri:
            # Update cached value
            active_new = DeviceRunProcess(
                process=active.process,
                device_id=active.device_id,
                command=active.command,
                log_path=active.log_path,
                vm_service_uri=uri,
            )
            self._running_device_runs[name] = active_new
            return uri

        return None

    def get_device_run_status(self, name: str) -> dict[str, Any]:
        """Get detailed status of device run including VM Service URI."""
        active = self._active_device_run(name)
        if active is None:
            return {
                "running": False,
                "device_id": None,
                "vm_service_uri": None,
                "log_path": self._last_device_run_logs.get(name),
            }

        # Try to get/update VM Service URI
        vm_uri = self.get_vm_service_uri(name)

        return {
            "running": True,
            "device_id": active.device_id,
            "vm_service_uri": vm_uri,
            "log_path": active.log_path,
            "pid": active.process.pid,
        }

    def is_server_running(self, name: str) -> bool:
        """Check if dev server is running (managed by Code Bridge)."""
        if name not in self._running_servers:
            return False
        process = self._running_servers[name].process
        if is_process_running(process):
            return True
        # Process died externally - clean up
        del self._running_servers[name]
        return False

    def get_server_port(self, name: str) -> int | None:
        """Get running server port."""
        if name in self._running_servers:
            process = self._running_servers[name].process
            if is_process_running(process):
                return self._running_servers[name].port
            del self._running_servers[name]

        return self.detect_running_server_port(name)

    def detect_running_server_port(self, name: str) -> int | None:
        """Detect port for an externally running project dev server."""
        db = self._project_db()
        return detect_project_running_server_port(
            db.get(name),
            detect_port_for_project=self._detect_port_for_project,
        )

    def _detect_port_for_project(self, project_path: str, project_type: ProjectType) -> int | None:
        return detect_port_for_project(project_path, project_type)

# Global project manager instance
_project_manager: ProjectManager | None = None


def get_project_manager() -> ProjectManager:
    """Get global project manager instance."""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager
