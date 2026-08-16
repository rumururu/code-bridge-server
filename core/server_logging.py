"""Server logging setup utilities."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.env_bootstrap import emit_recorded_bootstrap_log
from core.runtime_paths import runtime_path

LOG_PATH_ENV_KEYS = ("CODE_BRIDGE_SERVER_LOG_PATH", "CODEBRIDGE_SERVER_LOG_PATH")
MAX_LOG_BYTES = 2_000_000
LOG_BACKUP_COUNT = 3

_SERVER_LOG_RELATIVE = "logs/server.log"


def default_server_log_path() -> Path:
    """Where the log lands when no env override is set.

    Resolved lazily through :func:`core.runtime_paths.runtime_path` — never a
    module constant — so ``CODEBRIDGE_APP_SUPPORT_DIR`` (packaged desktop
    launches, and the test suite's conftest) redirects the log per call
    instead of freezing the operator's real ``~/.code-bridge`` path at import
    time.
    """
    return runtime_path(
        _SERVER_LOG_RELATIVE,
        Path("~/.code-bridge").expanduser() / _SERVER_LOG_RELATIVE,
    )


def resolve_server_log_path() -> Path:
    """Resolve server log path from env or default location."""
    for key in LOG_PATH_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return Path(value).expanduser()
    return default_server_log_path()


def _has_file_handler(logger: logging.Logger, log_path: Path) -> bool:
    """Check whether logger already has a file handler for the same path."""
    resolved = log_path.resolve()
    for handler in logger.handlers:
        filename = getattr(handler, "baseFilename", None)
        if not filename:
            continue
        try:
            if Path(filename).resolve() == resolved:
                return True
        except OSError:
            continue
    return False


def configure_server_logging(log_level: str = "info") -> Path:
    """Attach rotating file handler to root logger and forward uvicorn logs."""
    log_path = resolve_server_log_path()
    # Create parent dir with restrictive mode (POSIX only honors the mode bits).
    try:
        log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except TypeError:
        # Fallback for any Path stub without the mode kwarg.
        log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("")
    if not _has_file_handler(root_logger, log_path):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        # Tighten permissions on POSIX so logs aren't world-readable.
        if os.name == "posix":
            try:
                os.chmod(log_path, 0o600)
            except OSError:
                pass

    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(level)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.propagate = True
        if logger.level == logging.NOTSET:
            logger.setLevel(level)

    # The PATH bootstrap runs at server_cli.main() top — before any handler
    # exists — so its summary would otherwise vanish. Now that a file handler
    # is attached, replay the recorded one-liner (a no-op if bootstrap never
    # ran, and emitted at most once even though the dual-server entry point
    # configures logging twice).
    emit_recorded_bootstrap_log()

    return log_path
