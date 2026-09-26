"""Short-lived, scope-limited tokens for the browser workflow canvas.

Why a *separate* credential at all
----------------------------------
The canvas is browser JavaScript. Whatever credential that page holds is
readable by every line of script running in it, so the question is never "can
the page keep a secret" — it cannot — but **"what is that credential worth
when it leaks"**.

A paired API key is worth the whole server: ``/api/agent/*`` starts runs,
executes ``shell`` steps, reads the secret store and browses the filesystem.
A canvas token is worth *one agent's workflow graph, for fifteen minutes*, and
nothing else. It cannot start a run, cannot read a secret, cannot name a file.
That difference is the entire security design of the canvas: the leak is not
prevented, its blast radius is.

Modelled on :mod:`preview.preview_access` (``secrets.token_urlsafe(32)``, a
15-minute TTL, an expiry sweep on every touch, the issuing API key retained).
Two deliberate differences:

* **No ambient session.** ``PreviewAccessManager.bind_remote_session`` keys a
  session by client IP and ``get_client_ip`` trusts ``X-Forwarded-For`` — a
  header the client writes. Behind a tunnel an IP is not an identity, so the
  canvas presents its token on *every* request and this module has no
  IP-keyed state at all.
* **Scoped to one agent.** The token names the agent it was issued for. A
  request for a different agent's graph is refused even with a live token.

Nothing here ever logs a token. :func:`token_fingerprint` exists so an audit
row can be correlated with a session without the token value being writable to
disk in the first place.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta

CANVAS_TOKEN_TTL_MINUTES = 15

# The token opens exactly these two capabilities, and ``routes/canvas_api.py``
# publishes no route outside them. Kept as data (not prose) so the scope the
# app is handed is the same object the router is written against.
SCOPE_GRAPH_READ = "graph:read"
SCOPE_GRAPH_WRITE = "graph:write"
CANVAS_SCOPE: tuple[str, ...] = (SCOPE_GRAPH_READ, SCOPE_GRAPH_WRITE)

# Reasons a lookup can fail. Audited; never shown a token value.
REASON_MISSING = "missing"
REASON_UNKNOWN = "unknown"
REASON_EXPIRED = "expired"


def token_fingerprint(token: str) -> str:
    """A short one-way handle for a token, safe to write to an audit row.

    SHA-256 truncated to 12 hex chars: enough to tell two live sessions apart
    in a log, not enough (and not reversible enough) to replay. Deliberately
    *not* named ``*_token`` so it reads as what it is — and so the audit
    redactor's ``*_token`` key rule stays a second line of defence rather than
    the thing this depends on.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class CanvasSession:
    """One issued canvas token and the narrow authority it carries."""

    token: str
    agent_id: str
    created_at: datetime
    expires_at: datetime
    # The paired API key that asked for this token. Retained for the same
    # reason ``PreviewToken.api_key`` is: so a revoked pairing can be traced
    # to the sessions it spawned. Never travels to the browser.
    api_key: str | None = None
    scope: tuple[str, ...] = field(default=CANVAS_SCOPE)

    @property
    def fingerprint(self) -> str:
        return token_fingerprint(self.token)

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.expires_at <= (now or datetime.now())

    def allows(self, scope: str) -> bool:
        return scope in self.scope


@dataclass(frozen=True)
class CanvasTokenLookup:
    """Result of resolving a presented token.

    Carries a ``reason`` on failure so the caller can audit *why* a request
    was refused (an expired token and an invented one are different events)
    without the route having to re-derive it.
    """

    session: CanvasSession | None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.session is not None


class CanvasSessionManager:
    """In-memory store of live canvas tokens.

    In-memory on purpose: a canvas token must not outlive the process that
    issued it. There is no revocation list to keep in sync because a restart
    revokes everything, and fifteen minutes is short enough that this is a
    feature rather than a gap.
    """

    def __init__(self, ttl_minutes: int = CANVAS_TOKEN_TTL_MINUTES) -> None:
        self._ttl_minutes = ttl_minutes
        self._sessions: dict[str, CanvasSession] = {}

    @property
    def ttl_minutes(self) -> int:
        return self._ttl_minutes

    def _sweep_expired(self, now: datetime | None = None) -> list[CanvasSession]:
        """Drop expired sessions; return them so the caller can audit expiry."""
        now = now or datetime.now()
        expired = [
            session for session in self._sessions.values() if session.is_expired(now)
        ]
        for session in expired:
            self._sessions.pop(session.token, None)
        return expired

    def issue(self, agent_id: str, *, api_key: str | None = None) -> CanvasSession:
        self._sweep_expired()
        now = datetime.now()
        session = CanvasSession(
            token=secrets.token_urlsafe(32),
            agent_id=agent_id,
            created_at=now,
            expires_at=now + timedelta(minutes=self._ttl_minutes),
            api_key=api_key,
            scope=CANVAS_SCOPE,
        )
        self._sessions[session.token] = session
        return session

    def resolve(self, token: str | None) -> CanvasTokenLookup:
        if not token:
            return CanvasTokenLookup(None, REASON_MISSING)
        session = self._sessions.get(token)
        if session is None:
            # Sweep only after a miss: the common path (a live token) should
            # not pay for a full scan on every request.
            self._sweep_expired()
            return CanvasTokenLookup(None, REASON_UNKNOWN)
        if session.is_expired():
            self._sessions.pop(token, None)
            return CanvasTokenLookup(None, REASON_EXPIRED)
        return CanvasTokenLookup(session)

    def revoke(self, token: str) -> bool:
        return self._sessions.pop(token, None) is not None

    def active_count(self) -> int:
        self._sweep_expired()
        return len(self._sessions)

    def clear(self) -> None:
        """Drop every session. Used by tests; also the honest reset on logout."""
        self._sessions.clear()


_canvas_session_manager = CanvasSessionManager()


def get_canvas_session_manager() -> CanvasSessionManager:
    return _canvas_session_manager
