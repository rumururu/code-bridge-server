"""Async build execution helpers for supported project types."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from project_build_helpers import decode_build_error, resolve_nextjs_build_output_path


@dataclass(frozen=True)
class BuildExecutionResult:
    """Normalized result for one build execution."""

    success: bool
    message: str
    build_path: str | None = None


async def build_flutter_web_project(project_path: str) -> BuildExecutionResult:
    """Run `flutter build web --release` for a project path."""
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
            error_msg = decode_build_error(stderr, stdout) or "Build failed"
            return BuildExecutionResult(success=False, message=error_msg)

        return BuildExecutionResult(
            success=True,
            message="Build completed",
            build_path=str(Path(project_path) / "build" / "web"),
        )
    except FileNotFoundError:
        return BuildExecutionResult(
            success=False,
            message="Flutter CLI not found. Is Flutter installed?",
        )


async def build_nextjs_project(project_path: str) -> BuildExecutionResult:
    """Run `npm run build` and resolve Next.js output path."""
    try:
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
            error_msg = decode_build_error(stderr, stdout) or "Build failed"
            return BuildExecutionResult(success=False, message=error_msg)

        build_path = resolve_nextjs_build_output_path(project_path)
        if build_path is None:
            return BuildExecutionResult(
                success=False,
                message="Build completed but no output directory found",
            )

        return BuildExecutionResult(
            success=True,
            message="Build completed",
            build_path=build_path,
        )
    except FileNotFoundError:
        return BuildExecutionResult(
            success=False,
            message="npm not found. Is Node.js installed?",
        )
