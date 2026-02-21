"""Configuration loader for Code Bridge server."""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Server configuration from config.yaml."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            # Look for config.yaml in parent directory
            config_path = Path(__file__).parent.parent / "config.yaml"

        self._config = self._load_config(config_path)
        self._migration_done = False

    def _load_config(self, path: Path | str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            return yaml.safe_load(f)

    @property
    def host(self) -> str:
        return self._config.get("server", {}).get("host", "0.0.0.0")

    @property
    def port(self) -> int:
        return self._config.get("server", {}).get("port", 8080)

    @property
    def api_key(self) -> str:
        # API key auth is intentionally disabled in current app settings flow.
        # Keep returning empty so verify_api_key treats all requests as local/dev.
        return ""

    @property
    def cors_origins(self) -> list[str]:
        """Get CORS allowed origins. Defaults to ["*"] for development."""
        return self._config.get("server", {}).get("cors_origins", ["*"])

    @property
    def debug(self) -> bool:
        """Get debug mode. Defaults to True."""
        return self._config.get("server", {}).get("debug", True)

    @property
    def log_level(self) -> str:
        """Get log level. Defaults to 'info'."""
        return self._config.get("server", {}).get("log_level", "info")

    @property
    def weekly_budget_usd(self) -> float:
        """Weekly budget in USD used for usage percentage display."""
        raw_value = self._config.get("server", {}).get("weekly_budget_usd", 100.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 100.0
        return max(value, 0.0)

    @property
    def usage_window_days(self) -> int:
        """Rolling usage window in days."""
        raw_value = self._config.get("server", {}).get("usage_window_days", 7)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = 7
        return max(value, 1)

    @property
    def mdns_enabled(self) -> bool:
        """Whether mDNS service discovery is enabled. Defaults to True."""
        return self._config.get("server", {}).get("mdns_enabled", True)

    @property
    def server_name(self) -> str:
        """Human-readable server name for discovery. Defaults to 'Code Bridge'."""
        return self._config.get("server", {}).get("server_name", "Code Bridge")

    @property
    def remote_access_enabled(self) -> bool:
        """Whether remote access via Cloudflare Tunnel is enabled. Defaults to False."""
        return self._config.get("server", {}).get("remote_access_enabled", False)

    @property
    def firebase_enabled(self) -> bool:
        """Whether Firebase authentication is enabled. Defaults to False."""
        return self._config.get("server", {}).get("firebase_enabled", False)

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
