"""Backward compatibility stub - import from file_manager instead."""
from file_manager import (
    EXCLUDED_PATTERNS,
    LANGUAGE_MAP,
    MAX_FILE_SIZE,
    FileManager,
    get_file_manager,
)

__all__ = [
    "EXCLUDED_PATTERNS",
    "LANGUAGE_MAP",
    "MAX_FILE_SIZE",
    "FileManager",
    "get_file_manager",
]
