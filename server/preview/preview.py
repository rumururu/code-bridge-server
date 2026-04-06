"""Backward compatibility stub - import from preview_service instead."""
from .preview_service import (
    STRIP_RESPONSE_HEADERS,
    PreviewProxy,
    get_preview_proxy,
)

__all__ = [
    "STRIP_RESPONSE_HEADERS",
    "PreviewProxy",
    "get_preview_proxy",
]
