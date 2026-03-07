"""Project management for Code Bridge."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from database import get_project_db
from project_build_service import build_flutter_web_project, build_nextjs_project
from project_build_state import (
    build_status_payload,
    mark_build_error,
    mark_build_ready,
    mark_building,
    ready_build_path,
)
from project_dev_server_start import resolve_dev_server_start_plan, spawn_dev_server_process
from project_device_logs import device_run_log_path, read_log_tail
from project_device_run_plan import resolve_device_run_plan
from project_device_run_service import start_flutter_run_process, summarize_flutter_run_exit
from project_models import BuildInfo, DevServerProcess, DeviceRunProcess, ProjectType
from project_process_utils import is_process_running, terminate_process_safely
from project_query_service import (
    build_project_list_view,
    build_single_project_view,
    detect_project_running_server_port,
)
from project_server_detection import (
    detect_port_for_project,
    list_listening_processes,
    list_process_cwds,
)
from project_server_process import extract_process_error, wait_for_project_server_port


@dataclass
class ProjectManager:
    """Manages project lifecycle and dev servers."""

    _project_db_factory: Callable[[], Any] = get_project_db
    _running_servers: dict[str, DevServerProcess] = field(default_factory=dict)
    _running_device_runs: dict[str, DeviceRunProcess] = field(default_factory=dict)
    _last_device_run_logs: dict[str, str] = field(default_factory=dict)
    _build_info: dict[str, BuildInfo] = field(default_factory=dict)

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

        return build_project_list_view(projects, get_server_port=get_server_port_cached)

    def get_project(self, name: str) -> dict[str, Any] | None:
        """Get specific project info."""
        db = self._project_db()
        return build_single_project_view(name, db.get(name), get_server_port=self.get_server_port)

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

        except Exception as e:
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

        except Exception as e:
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

        info = DeviceRunProcess(
            process=process,
            device_id=plan.device_id,
            command=start_result.command,
            log_path=str(log_path),
        )
        self._running_device_runs[name] = info
        self._last_device_run_logs[name] = info.log_path

        return {
            "success": True,
            "message": f"Started flutter run on {plan.device_id}",
            "pid": process.pid,
            "device_id": plan.device_id,
            "log_path": info.log_path,
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

    def is_server_running(self, name: str) -> bool:
        """Check if dev server is running."""
        return name in self._running_servers

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
        mark_building(self._build_info, name, project_type.value)

        try:
            if project_type == ProjectType.FLUTTER:
                return await self._build_flutter(name, project_path)
            elif project_type == ProjectType.NEXTJS:
                return await self._build_nextjs(name, project_path)
            else:
                mark_build_error(
                    self._build_info,
                    name,
                    error_message=f"Unsupported project type: {project.get('type')}",
                )
                return {"success": False, "message": f"Unsupported project type: {project.get('type')}"}

        except Exception as e:
            error_msg = str(e)
            mark_build_error(self._build_info, name, error_message=error_msg)
            return {"success": False, "message": f"Build failed: {error_msg}"}

    async def _build_flutter(self, name: str, project_path: str) -> dict[str, Any]:
        """Build Flutter web app."""
        result = await build_flutter_web_project(project_path)
        if not result.success:
            error_msg = result.message or "Build failed"
            mark_build_error(
                self._build_info,
                name,
                error_message=error_msg,
                project_type=ProjectType.FLUTTER.value,
            )
            return {"success": False, "message": error_msg}

        mark_build_ready(
            self._build_info,
            name,
            build_path=result.build_path,
            project_type=ProjectType.FLUTTER.value,
        )
        return {
            "success": True,
            "message": result.message,
            "build_path": result.build_path,
        }

    async def _build_nextjs(self, name: str, project_path: str) -> dict[str, Any]:
        """Build Next.js app.

        Runs `npm run build`. Output depends on next.config.js:
        - Static export (output: 'export'): out/
        - Standalone (output: 'standalone'): .next/standalone/
        - Default: .next/ (requires next start)
        """
        result = await build_nextjs_project(project_path)
        if not result.success:
            error_msg = result.message or "Build failed"
            mark_build_error(
                self._build_info,
                name,
                error_message=error_msg,
                project_type=ProjectType.NEXTJS.value,
            )
            return {"success": False, "message": error_msg}

        mark_build_ready(
            self._build_info,
            name,
            build_path=result.build_path,
            project_type=ProjectType.NEXTJS.value,
        )
        return {
            "success": True,
            "message": result.message,
            "build_path": result.build_path,
        }

    def get_build_status(self, name: str) -> dict[str, Any]:
        """Get build status for a project."""
        return build_status_payload(self._build_info.get(name))

    def get_build_path(self, name: str) -> str | None:
        """Get build path for a project if ready."""
        return ready_build_path(self._build_info.get(name))


# Global project manager instance
_project_manager: ProjectManager | None = None


def get_project_manager() -> ProjectManager:
    """Get global project manager instance."""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager
