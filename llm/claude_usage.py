"""Helpers for collecting and normalizing Claude usage metrics."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import select
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

CLAUDE_USAGE_CACHE_TTL_SECONDS = 300
CLAUDE_USAGE_UNSUPPORTED_TTL_SECONDS = 300
CLAUDE_USAGE_ERROR_RETRY_TTL_SECONDS = 30

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
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Error probing Claude usage: %s", exc)
        return None
    except (ValueError, UnicodeDecodeError, RuntimeError) as exc:
        logger.warning("Unexpected error probing Claude usage: %s", exc)
        return None
    finally:
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    process.kill()
                    process.wait(timeout=0.5)
                except (subprocess.TimeoutExpired, OSError, ProcessLookupError):
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
            ttl_seconds = CLAUDE_USAGE_ERROR_RETRY_TTL_SECONDS
        else:
            stdout_text = stdout_raw.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_raw.decode("utf-8", errors="replace").strip()

            if not stdout_text:
                payload["claude_usage_error"] = "empty_output"
                ttl_seconds = CLAUDE_USAGE_ERROR_RETRY_TTL_SECONDS
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
                        ttl_seconds = CLAUDE_USAGE_ERROR_RETRY_TTL_SECONDS

                if stderr_text and payload["claude_usage_error"] is None:
                    payload["claude_usage_error"] = f"stderr: {stderr_text[:120]}"
    except FileNotFoundError:
        payload["claude_usage_supported"] = False
        payload["claude_usage_error"] = "claude_not_found"
        ttl_seconds = CLAUDE_USAGE_UNSUPPORTED_TTL_SECONDS
    except (OSError, subprocess.SubprocessError, ValueError, UnicodeDecodeError) as exc:
        payload["claude_usage_error"] = f"exec_error: {exc}"
        ttl_seconds = CLAUDE_USAGE_ERROR_RETRY_TTL_SECONDS

    _claude_usage_cache["payload"] = dict(payload)
    _claude_usage_cache["expires_at"] = now + float(ttl_seconds)
    return dict(payload)


_CLI_STATS_CACHE_TTL_SECONDS = 300
_CLI_STATS_ERROR_RETRY_TTL_SECONDS = 30
_cli_stats_cache: dict[str, Any] = {"expires_at": 0.0, "payload": None}
_cli_stats_probe_lock = asyncio.Lock()

_BOX_CHARS_PATTERN = re.compile(r"[\u2500-\u257F\u2580-\u259F│┃║░▒▓█·]")
_USED_PERCENT_PATTERN = re.compile(
    r"\b(\d{1,3}(?:\.\d+)?)\s*%\s*used\b", flags=re.IGNORECASE
)
_RESETS_PATTERN = re.compile(r"\bResets\s+(.+)$", flags=re.IGNORECASE)


def _clean_terminal_lines(output: str) -> list[str]:
    """Strip ANSI + box drawing chars and split into clean trimmed lines."""
    stripped = _strip_ansi(output).replace("\r", "\n")
    stripped = _BOX_CHARS_PATTERN.sub(" ", stripped)
    lines: list[str] = []
    for raw in stripped.split("\n"):
        compacted = re.sub(r"\s+", " ", raw).strip()
        if compacted:
            lines.append(compacted)
    return lines


def _compact_for_search(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _matches_tokens(line: str, tokens: list[str]) -> bool:
    compact = _compact_for_search(line)
    return all(_compact_for_search(token) in compact for token in tokens)


def _reset_label_from_line(line: str) -> str | None:
    match = _RESETS_PATTERN.search(line)
    if match is None:
        return None
    candidate = match.group(1).strip()
    candidate = re.sub(
        r"\d{1,3}(?:\.\d+)?\s*%\s*used.*$", "", candidate, flags=re.IGNORECASE
    )
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate or None


def _parse_usage_window(
    lines: list[str], *, required_tokens: list[str]
) -> dict[str, Any] | None:
    """Locate a `current week | <tokens>` block and pull used_percent + reset label."""
    for i, line in enumerate(lines):
        if not _matches_tokens(line, required_tokens):
            continue
        used_percent: float | None = None
        reset_label: str | None = None
        end = min(i + 12, len(lines))
        for j in range(i, end):
            candidate = lines[j]
            if (
                j > i
                and _matches_tokens(candidate, ["current week"])
                and not _matches_tokens(candidate, required_tokens)
            ):
                break
            if reset_label is None:
                reset_label = _reset_label_from_line(candidate)
            if used_percent is None:
                percent_match = _USED_PERCENT_PATTERN.search(candidate)
                if percent_match is not None:
                    try:
                        used_percent = max(0.0, min(100.0, float(percent_match.group(1))))
                    except (TypeError, ValueError):
                        used_percent = None
        if used_percent is not None:
            return {
                "used_percent": round(used_percent, 2),
                "reset_label": reset_label,
            }
    return None


def _probe_claude_cli_stats_via_tui(timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Launch interactive Claude in a PTY, run /stats, parse usage windows."""
    master_fd: int | None = None
    slave_fd: int | None = None
    process: subprocess.Popen[Any] | None = None
    collected_chunks: list[str] = []
    sent_stats = False
    sent_detail = False
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
            env={
                **os.environ.copy(),
                "TERM": "xterm-256color",
                "COLUMNS": "160",
                "LINES": "50",
            },
        )
        os.close(slave_fd)
        slave_fd = None

        start = time.monotonic()
        while (time.monotonic() - start) < timeout_seconds:
            elapsed = time.monotonic() - start
            if not sent_stats and elapsed > 1.4:
                os.write(master_fd, b"/stats\r")
                sent_stats = True
            elif sent_stats and not sent_detail and elapsed > 3.0:
                # Shift+Tab toggles to detailed view in Claude CLI TUI.
                os.write(master_fd, b"\x1B[Z")
                sent_detail = True
            elif sent_detail and not sent_exit and elapsed > 4.8:
                os.write(master_fd, b"\x03")
                sent_exit = True

            readable, _, _ = select.select([master_fd], [], [], 0.2)
            if readable:
                chunk = os.read(master_fd, 8192)
                if not chunk:
                    break
                collected_chunks.append(chunk.decode("utf-8", errors="replace"))
                if len(collected_chunks) > 200:
                    collected_chunks = collected_chunks[-200:]

        lines = _clean_terminal_lines("".join(collected_chunks))
        all_models = _parse_usage_window(
            lines, required_tokens=["current week", "all models"]
        )
        sonnet_only = _parse_usage_window(
            lines, required_tokens=["current week", "sonnet only"]
        )
        if all_models is None and sonnet_only is None:
            return {"available": False, "error": "parse_failed"}
        return {
            "available": True,
            "all_models": all_models,
            "sonnet_only": sonnet_only,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Error probing Claude /stats: %s", exc)
        return {"available": False, "error": f"probe_failed:{exc}"}
    finally:
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    process.kill()
                    process.wait(timeout=0.5)
                except (subprocess.TimeoutExpired, OSError, ProcessLookupError):
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


async def fetch_claude_cli_stats(force_refresh: bool = False) -> dict[str, Any]:
    """Return cached AIA-style /stats snapshot or probe Claude TUI for fresh data."""
    now = time.monotonic()
    cached = _cli_stats_cache.get("payload")
    expires_at = float(_cli_stats_cache.get("expires_at") or 0.0)
    if not force_refresh and isinstance(cached, dict) and now < expires_at:
        return dict(cached)

    async with _cli_stats_probe_lock:
        # Double-check after acquiring lock (another request may have refreshed).
        cached = _cli_stats_cache.get("payload")
        expires_at = float(_cli_stats_cache.get("expires_at") or 0.0)
        if not force_refresh and isinstance(cached, dict) and now < expires_at:
            return dict(cached)
        payload = await asyncio.to_thread(_probe_claude_cli_stats_via_tui, 10.0)
        ttl = (
            _CLI_STATS_CACHE_TTL_SECONDS
            if payload.get("available")
            else _CLI_STATS_ERROR_RETRY_TTL_SECONDS
        )
        _cli_stats_cache["payload"] = dict(payload)
        _cli_stats_cache["expires_at"] = now + float(ttl)
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
