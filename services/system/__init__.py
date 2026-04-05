"""System services package.

Provides system inspection and settings management.
"""

from system.system_inspect_service import get_system_inspect_service, SystemInspectService
from system.system_settings_service import get_system_settings_service, SystemSettingsService

__all__ = [
    "SystemInspectService",
    "get_system_inspect_service",
    "SystemSettingsService",
    "get_system_settings_service",
]
