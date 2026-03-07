"""Helpers for collecting and normalizing Claude usage metrics."""

from __future__ import annotations

import asyncio
import json
import os
import re
import select
import subprocess
import time
from typing import Any

CLAUDE_USAGE_CACHE_TTL_SECONDS = 300
CLAUDE_USAGE_UNSUPPORTED_TTL_SECONDS = 300

_claude_usage_cache: dict[str, Any] = {
    "expires_at": 0.0,
    "payload": None,
}
_claude_usage_probe_lock = asyncio.Lock()
_ansi_escape_pattern = re.compile(
    r"\x1B\[[0-?]*[ -/]*[@-~]|\x1B\][^\x07]*(?:\x07|\x1B\\)|\x1B[@-_]"
)


def _strip_ansi(text: str) -> str:
    """Strip ANSI control sequences for easier text parsing."""
    return _ansi_escape_pattern.sub("", text)


def _extract_usage_percent_from_text(text: str) -> float | None:
    """Extract first likely usage percentage from /usage text output."""
    if not text.strip():
        return None

    normalized = _strip_ansi(text)
    prioritized_patterns = (
        r"(?:used|usage|weekly|limit)[^\n]{0,80}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"(\d{1,3}(?:\.\d+)?)\s*%[^\n]{0,80}?(?:weekly|limit|usage)",
    )
    for pattern in prioritized_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match is not None:
            try:
                return round(float(match.group(1)), 2)
            except (TypeError, ValueError):
                pass

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    focus_lines = [
        line
        for line in lines
        if any(token in line.lower() for token in ("usage", "limit", "week", "weekly"))
    ]
    candidates = focus_lines or lines

    for line in candidates:
        for match in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%", line):
            try:
                return round(float(match.group(1)), 2)
            except (TypeError, ValueError):
                continue
    return None


def _probe_claude_usage_percent_via_tui(timeout_seconds: float = 10.0) -> float | None:
    """Launch interactive Claude in a PTY, run /usage, and parse % from screen output."""
    master_fd: int | None = None
    slave_fd: int | None = None
    process: subprocess.Popen[Any] | None = None
    collected_chunks: list[str] = []
    sent_usage = False
    sent_exit = False

    try:
        master_fd, slave_fd = os.openpty()
        process = subprocess.Popen(
            ["claude"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
            env=os.environ.copy(),
        )
        os.close(slave_fd)
        slave_fd = None

        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout_seconds:
            if not sent_usage and (time.monotonic() - start_time) > 1.2:
                os.write(master_fd, b"/usage\n")
                sent_usage = True

            readable, _, _ = select.select([master_fd], [], [], 0.2)
            if readable:
                chunk = os.read(master_fd, 8192)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                collected_chunks.append(text)
                if len(collected_chunks) > 120:
                    collected_chunks = collected_chunks[-120:]

                parsed_percent = _extract_usage_percent_from_text("".join(collected_chunks))
                if parsed_percent is not None:
                    return parsed_percent

            if sent_usage and not sent_exit and (time.monotonic() - start_time) > 4.5:
                os.write(master_fd, b"/exit\n")
                sent_exit = True

        return None
    except Exception:
        return None
    finally:
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=0.5)
                except Exception:
                    pass
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass
        if slave_fd is not None:
            try:
                os.close(slave_fd)
            except OSError:
                pass


async def fetch_claude_usage_snapshot(
    force_refresh: bool = False,
    allow_refresh: bool = True,
) -> dict[str, Any]:
    """Get Claude weekly usage percentage via /usage when available."""
    now = time.monotonic()
    cached_payload = _claude_usage_cache.get("payload")
    expires_at = float(_claude_usage_cache.get("expires_at") or 0.0)
    if not force_refresh and cached_payload is not None and now < expires_at:
        return dict(cached_payload)

    if not allow_refresh:
        if isinstance(cached_payload, dict):
            return dict(cached_payload)
        return {
            "claude_usage_supported": None,
            "claude_usage_percent": None,
            "claude_usage_error": "unavailable",
        }

    payload: dict[str, Any] = {
        "claude_usage_supported": None,
        "claude_usage_percent": None,
        "claude_usage_error": None,
    }
    ttl_seconds = CLAUDE_USAGE_CACHE_TTL_SECONDS

    try:
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "-p",
            "/usage",
            "--output-format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )

        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(proc.communicate(), timeout=8.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            payload["claude_usage_error"] = "timeout"
            ttl_seconds = 30
        else:
            stdout_text = stdout_raw.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_raw.decode("utf-8", errors="replace").strip()

            if not stdout_text:
                payload["claude_usage_error"] = "empty_output"
                ttl_seconds = 30
            else:
                try:
                    parsed = json.loads(stdout_text)
                except json.JSONDecodeError:
                    result_text = stdout_text
                else:
                    result_text = ""
                    if isinstance(parsed, dict):
                        raw_result = parsed.get("result")
                        if isinstance(raw_result, str):
                            result_text = raw_result

                lowered = result_text.lower()
                if "unknown skill: usage" in lowered:
                    payload["claude_usage_error"] = "unsupported"
                    # Fallback: probe interactive TUI output and parse percentage.
                    async with _claude_usage_probe_lock:
                        tui_percent = await asyncio.to_thread(
                            _probe_claude_usage_percent_via_tui,
                            10.0,
                        )
                    if tui_percent is not None:
                        payload["claude_usage_supported"] = True
                        payload["claude_usage_percent"] = tui_percent
                        payload["claude_usage_error"] = None
                        ttl_seconds = CLAUDE_USAGE_CACHE_TTL_SECONDS
                    else:
                        payload["claude_usage_supported"] = False
                        payload["claude_usage_error"] = "unsupported"
                        ttl_seconds = CLAUDE_USAGE_UNSUPPORTED_TTL_SECONDS
                else:
                    usage_percent = _extract_usage_percent_from_text(result_text)
                    payload["claude_usage_supported"] = True
                    payload["claude_usage_percent"] = usage_percent
                    if usage_percent is None:
                        payload["claude_usage_error"] = "percent_not_found"
                        ttl_seconds = 30

                if stderr_text and payload["claude_usage_error"] is None:
                    payload["claude_usage_error"] = f"stderr: {stderr_text[:120]}"
    except FileNotFoundError:
        payload["claude_usage_supported"] = False
        payload["claude_usage_error"] = "claude_not_found"
        ttl_seconds = CLAUDE_USAGE_UNSUPPORTED_TTL_SECONDS
    except Exception as exc:
        payload["claude_usage_error"] = f"exec_error: {exc}"
        ttl_seconds = 30

    _claude_usage_cache["payload"] = dict(payload)
    _claude_usage_cache["expires_at"] = now + float(ttl_seconds)
    return dict(payload)


def merge_usage_for_display(
    weekly_summary: dict[str, Any],
    claude_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Attach display percent. Only Claude /usage is used for UI display."""
    merged = {**weekly_summary, **claude_snapshot}

    claude_percent = merged.get("claude_usage_percent")
    if isinstance(claude_percent, (int, float)):
        merged["display_usage_percent"] = round(float(claude_percent), 2)
        merged["display_usage_source"] = "claude_usage"
    else:
        merged["display_usage_percent"] = None
        merged["display_usage_source"] = "unavailable"

    return merged
