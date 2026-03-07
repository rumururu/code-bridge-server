"""Preview token/session access control helpers."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi import Request

PREVIEW_TOKEN_TTL_MINUTES = 15


@dataclass
class PreviewToken:
    token: str
    project_name: str
    created_at: datetime
    expires_at: datetime


class PreviewAccessManager:
    """Manage preview tokens, remote sessions, and last active project."""

    def __init__(self, ttl_minutes: int = PREVIEW_TOKEN_TTL_MINUTES) -> None:
        self._ttl_minutes = ttl_minutes
        self._preview_tokens: dict[str, PreviewToken] = {}
        self._preview_sessions: dict[str, str] = {}
        self._last_previewed_project: str | None = None

    @property
    def ttl_minutes(self) -> int:
        return self._ttl_minutes

    def _cleanup_expired_tokens(self) -> None:
        now = datetime.now()
        expired = [
            token
            for token, token_data in self._preview_tokens.items()
            if token_data.expires_at < now
        ]
        for token in expired:
            del self._preview_tokens[token]

    def generate_preview_token(self, project_name: str) -> str:
        self._cleanup_expired_tokens()
        token = secrets.token_urlsafe(32)
        now = datetime.now()
        self._preview_tokens[token] = PreviewToken(
            token=token,
            project_name=project_name,
            created_at=now,
            expires_at=now + timedelta(minutes=self._ttl_minutes),
        )
        return token

    def validate_preview_token(self, token: str, project_name: str) -> bool:
        self._cleanup_expired_tokens()
        token_data = self._preview_tokens.get(token)
        if token_data is None:
            return False
        return token_data.project_name == project_name and token_data.expires_at > datetime.now()

    def get_client_ip(self, request: Request) -> str:
        # Check forwarded headers first.
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        return request.client.host if request.client else "unknown"

    def is_local_request(self, request: Request) -> bool:
        client_ip = self.get_client_ip(request)
        local_prefixes = (
            "127.",
            "::1",
            "192.168.",
            "10.",
            "172.16.",
            "172.17.",
            "172.18.",
            "172.19.",
            "172.20.",
            "172.21.",
            "172.22.",
            "172.23.",
            "172.24.",
            "172.25.",
            "172.26.",
            "172.27.",
            "172.28.",
            "172.29.",
            "172.30.",
            "172.31.",
            "localhost",
        )
        return any(client_ip.startswith(prefix) for prefix in local_prefixes)

    def bind_remote_session(self, request: Request, project_name: str) -> None:
        self._preview_sessions[self.get_client_ip(request)] = project_name

    def has_remote_session(self, request: Request, project_name: str) -> bool:
        return self._preview_sessions.get(self.get_client_ip(request)) == project_name

    def set_last_previewed_project(self, project_name: str) -> None:
        self._last_previewed_project = project_name

    def get_last_previewed_project(self) -> str | None:
        return self._last_previewed_project


_preview_access_manager = PreviewAccessManager()


def get_preview_access_manager() -> PreviewAccessManager:
    return _preview_access_manager
