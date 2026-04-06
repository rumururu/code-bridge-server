"""Helpers for project build output resolution and error parsing."""

from __future__ import annotations

from pathlib import Path


def decode_build_error(stderr: bytes, stdout: bytes) -> str:
    """Return a concise build error string preferring stderr over stdout."""
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    if stderr_text:
        return stderr_text
    return stdout.decode("utf-8", errors="replace").strip()


def resolve_nextjs_build_output_path(project_path: str | Path) -> str | None:
    """Resolve Next.js build output path by known output conventions."""
    base = Path(project_path)
    out_path = base / "out"
    standalone_path = base / ".next" / "standalone"
    next_path = base / ".next"

    if out_path.exists() and (out_path / "index.html").exists():
        return str(out_path)
    if standalone_path.exists():
        return str(standalone_path)
    if next_path.exists():
        return str(next_path)
    return None
