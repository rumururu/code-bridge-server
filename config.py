"""Configuration loader for Code Bridge server."""

from pathlib import Path
from typing import Any

import yaml

# Server version
VERSION = "1.0.0"


class Config:
    """Server configuration from config.yaml."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            # Look for config.yaml in parent directory
            config_path = Path(__file__).parent.parent / "config.yaml"

        self._config = self._load_config(config_path)
        self._migration_done = False
        self._runtime_port: int | None = None  # CLI override

    def _load_config(self, path: Path | str) -> dict[str, Any]:
        """Load configuration from YAML file, creating default if missing."""
        path = Path(path)
        if not path.exists():
            self._create_default_config(path)
            print(f"Created default config: {path}")

        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _create_default_config(self, path: Path) -> None:
        """Create default config.yaml with sensible defaults."""
        default_config = """\
server:
  host: "0.0.0.0"
  port: 8766
  server_name: "Code Bridge"
  debug: true
  log_level: "info"
  cors_origins: ["*"]

  # LLM usage tracking
  weekly_budget_usd: 100.0
  usage_window_days: 7

  # Heartbeat interval for presence updates (minutes)
  heartbeat_interval_minutes: 15
"""
        path.write_text(default_config)

    def _get_server_value(self, key: str, default: Any) -> Any:
        """Get server section value with fallback."""
        return self._config.get("server", {}).get(key, default)

    @staticmethod
    def _parse_int(value: Any, default: int, *, minimum: int | None = None) -> int:
        """Parse integer config value with optional lower bound."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        if minimum is not None:
            parsed = max(parsed, minimum)
        return parsed

    @staticmethod
    def _parse_float(value: Any, default: float, *, minimum: float | None = None) -> float:
        """Parse float config value with optional lower bound."""
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        if minimum is not None:
            parsed = max(parsed, minimum)
        return parsed

    @property
    def host(self) -> str:
        return self._get_server_value("host", "0.0.0.0")

    @property
    def port(self) -> int:
        """Legacy port property. Returns dashboard_port for backward compatibility."""
        return self.dashboard_port

    @property
    def dashboard_port(self) -> int:
        """Dashboard port (localhost only, not exposed via tunnel).

        Default: 8766
        """
        import os
        env_port = os.environ.get("CODEBRIDGE_DASHBOARD_PORT")
        if env_port:
            return int(env_port)
        if self._runtime_port is not None:
            return self._runtime_port
        return self._get_server_value("dashboard_port", 8766)

    @property
    def api_port(self) -> int:
        """API port (exposed via tunnel for external access).

        Default: dashboard_port + 1
        """
        import os
        env_port = os.environ.get("CODEBRIDGE_API_PORT")
        if env_port:
            return int(env_port)
        return self._get_server_value("api_port", self.dashboard_port + 1)

    def set_runtime_port(self, port: int) -> None:
        """Set runtime dashboard port override (from CLI --port argument)."""
        import os
        os.environ["CODEBRIDGE_DASHBOARD_PORT"] = str(port)
        self._runtime_port = port

    @property
    def api_key(self) -> str:
        # API key auth is intentionally disabled in current app settings flow.
        # Keep returning empty so verify_api_key treats all requests as local/dev.
        return ""

    @property
    def cors_origins(self) -> list[str]:
        """Get CORS allowed origins. Defaults to ["*"] for development."""
        return self._get_server_value("cors_origins", ["*"])

    @property
    def debug(self) -> bool:
        """Get debug mode. Defaults to True."""
        return self._get_server_value("debug", True)

    @property
    def log_level(self) -> str:
        """Get log level. Defaults to 'info'."""
        return self._get_server_value("log_level", "info")

    @property
    def weekly_budget_usd(self) -> float:
        """Weekly budget in USD used for usage percentage display."""
        raw_value = self._get_server_value("weekly_budget_usd", 100.0)
        return self._parse_float(raw_value, 100.0, minimum=0.0)

    @property
    def usage_window_days(self) -> int:
        """Rolling usage window in days."""
        raw_value = self._get_server_value("usage_window_days", 7)
        return self._parse_int(raw_value, 7, minimum=1)

    @property
    def server_name(self) -> str:
        """Human-readable server name for discovery. Defaults to 'Code Bridge'."""
        return self._get_server_value("server_name", "Code Bridge")

    @property
    def heartbeat_interval_minutes(self) -> int:
        """Heartbeat interval in minutes for Firebase presence updates.

        Defaults to 15 minutes, minimum 5 minutes.
        """
        raw_value = self._get_server_value("heartbeat_interval_minutes", 15)
        return self._parse_int(raw_value, 15, minimum=5)  # Minimum 5 minutes

    @property
    def remote_access_enabled(self) -> bool:
        """Whether remote access via Cloudflare Tunnel is enabled. Defaults to True."""
        return self._get_server_value("remote_access_enabled", True)

    @property
    def firebase_enabled(self) -> bool:
        """Whether Firebase authentication is enabled. Defaults to False."""
        return self._get_server_value("firebase_enabled", False)

    @property
    def accessible_folders(self) -> list[str]:
        """Get list of accessible folder paths from database.

        These directories and their subdirectories are accessible via
        filesystem API. This is SEPARATE from the projects list.

        Falls back to home directory if not configured.
        """
        # Import here to avoid circular import
        from database import get_accessible_folder_db

        db = get_accessible_folder_db()
        folders = db.get_all()

        if not folders:
            return [str(Path.home())]

        # Filter to only existing paths
        paths = []
        for folder in folders:
            resolved = Path(folder).resolve()
            if resolved.exists():
                paths.append(str(resolved))

        return sorted(paths) if paths else [str(Path.home())]

    @property
    def config_projects(self) -> list[dict[str, Any]]:
        """Get projects from config.yaml (for migration only)."""
        return self._config.get("projects", [])

    def migrate_projects_to_db(self) -> int:
        """Migrate projects from config.yaml to SQLite database."""
        if self._migration_done:
            return 0

        from database import get_project_db

        db = get_project_db()
        count = db.migrate_from_config(self.config_projects)

        if count > 0:
            print(f"Migrated {count} projects from config.yaml to database")

        self._migration_done = True
        return count

    def get_project(self, name: str) -> dict[str, Any] | None:
        """Get project configuration by name from database."""
        from database import get_project_db

        db = get_project_db()
        return db.get(name)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
