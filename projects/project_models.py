"""Shared project manager enums and dataclasses."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum


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
    vm_service_uri: str | None = None
