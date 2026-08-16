"""Push an agent's notification to the phone while the app is closed.

`notification_store` already makes a notify step's message durable — it
survives the run and shows up the next time the inbox is opened. But "next
time the inbox is opened" is a pull, and the whole point of an unattended
agent is that nobody is watching. This module is the push half: it hands the
same message to Firebase Cloud Messaging so Android can show it in the
system tray even with the app fully closed.

Two failure modes must be told apart, because they mean different things to
the person running the server:

* No service-account key on disk, or the `firebase-admin` package missing —
  push is simply turned off. The operator never finished (or never wanted)
  the optional Firebase setup. This must never raise; every notify step
  would otherwise fail for a reason that has nothing to do with the step.
* A key that exists but is malformed, or a send that FCM rejects — logged
  once/per-call, never raised. The notification is already durably stored
  by the time this module is called, so losing the push is losing a ring,
  not losing the message.

A token FCM reports as unregistered (app uninstalled, data cleared, token
rotated away) is dropped by the caller via the `dropped` list this module
returns — retrying it forever would just be a wasted network call on every
future notify.

Android only for now. iOS needs an APNs key from a separate Apple developer
account; nothing here assumes a platform, so adding it later is wiring a
second credential path, not rewriting this module.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from core.runtime_paths import runtime_path

logger = logging.getLogger(__name__)

# Overridable so tests (and unusual deployments) don't have to touch the
# operator's real key on disk.
SERVICE_ACCOUNT_ENV = "CODEBRIDGE_FCM_SERVICE_ACCOUNT"
_SERVICE_ACCOUNT_FILENAME = "firebase_service_account.json"

_APP_NAME = "codebridge-push"

_lock = threading.Lock()
_app: Any = None
_init_attempted = False
_warned: set[str] = set()


def default_service_account_path() -> Path:
    """Where the key lives when no env override is set.

    Resolved lazily through :func:`core.runtime_paths.runtime_path` — never a
    module constant — because ``CODEBRIDGE_APP_SUPPORT_DIR`` (set by the
    packaged desktop launcher, and by the test suite's conftest) decides
    whether this is the operator's real ``~/.code-bridge`` key or an isolated
    copy, and that decision must be made per call, not frozen at import time.
    """
    return runtime_path(
        _SERVICE_ACCOUNT_FILENAME,
        Path.home() / ".code-bridge" / _SERVICE_ACCOUNT_FILENAME,
    )


def _service_account_path() -> Path:
    raw = os.environ.get(SERVICE_ACCOUNT_ENV)
    return Path(raw).expanduser() if raw else default_service_account_path()


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message)


def reset_for_tests() -> None:
    """Drop cached init state so each test starts from "push is off"."""
    global _app, _init_attempted
    with _lock:
        _app = None
        _init_attempted = False
        _warned.clear()


def is_configured() -> bool:
    """Whether a service-account key exists at the resolved path.

    Cheap existence check — does not attempt to initialize the SDK or
    validate the file's contents. Used by callers that want to skip a push
    attempt entirely without paying for a warning log.
    """
    return _service_account_path().exists()


def _get_app() -> Any:
    """Lazily initialize the Admin SDK app, or return None if push is off."""
    global _app, _init_attempted
    if _app is not None:
        return _app
    with _lock:
        if _app is not None or _init_attempted:
            return _app
        _init_attempted = True

        path = _service_account_path()
        if not path.exists():
            _warn_once(
                "no_key",
                f"Push notifications disabled: no service account key at {path}. "
                "Notifications are still stored and visible in the inbox.",
            )
            return None

        try:
            import firebase_admin
            from firebase_admin import credentials
        except ImportError:
            _warn_once(
                "no_package",
                "Push notifications disabled: the firebase-admin package is not "
                "installed. Notifications are still stored and visible in the inbox.",
            )
            return None

        try:
            cred = credentials.Certificate(str(path))
            _app = firebase_admin.initialize_app(cred, name=_APP_NAME)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _warn_once(
                "bad_key",
                f"Push notifications disabled: service account key at {path} is "
                f"invalid ({exc}). Notifications are still stored and visible in "
                "the inbox.",
            )
            _app = None
        return _app


def send_to_tokens(
    tokens: list[str],
    *,
    title: str,
    body: str | None = None,
    notification_id: str | None = None,
    level: str | None = None,
) -> dict[str, list[str]]:
    """Best-effort push of one notification to a set of device tokens.

    Never raises. Returns ``{"delivered": [...], "dropped": [...]}`` —
    ``dropped`` is every token FCM reported as unregistered; the caller
    (`pairing.PairingService.remove_push_token`) should stop tracking them
    so a dead install doesn't cost a network round-trip on every future
    notify.
    """
    result: dict[str, list[str]] = {"delivered": [], "dropped": []}
    if not tokens:
        return result

    app = _get_app()
    if app is None:
        return result

    try:
        from firebase_admin import exceptions, messaging
    except ImportError:
        return result

    data = {"notification_id": notification_id or ""}
    if level:
        data["level"] = level

    for token in tokens:
        if not token:
            continue
        try:
            message = messaging.Message(
                token=token,
                notification=messaging.Notification(title=title, body=body),
                data=data,
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        title=title,
                        body=body,
                    ),
                ),
            )
            messaging.send(message, app=app)
            result["delivered"].append(token)
        except messaging.UnregisteredError:
            logger.info("Push token no longer registered; dropping it.")
            result["dropped"].append(token)
        except (exceptions.FirebaseError, ValueError) as exc:
            # Any other FCM failure (quota, auth, malformed token, transient
            # network error) is a lost ring, not a lost message — the
            # notification is already durably stored. Log and move on.
            logger.warning("Push send failed for a token: %s", exc)

    return result


__all__ = [
    "SERVICE_ACCOUNT_ENV",
    "default_service_account_path",
    "is_configured",
    "reset_for_tests",
    "send_to_tokens",
]
