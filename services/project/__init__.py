"""Project management service package.

Provides project CRUD operations, build management, and query services.
"""

from project_manager import (
    ProjectManager,
    get_project_manager,
)

__all__ = [
    "ProjectManager",
    "get_project_manager",
]
