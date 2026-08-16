"""Which channel an approval decision physically arrived on.

``desktop_only`` is how the policy engine says "a person at this machine has
to decide this". It is what ``file.read`` of ``~/.ssh/id_rsa``, a workspace
``.env``, ``firebase_service_account.json`` or an ``AuthKey_*.p8`` escalates
to (see ``policy/path_guard.py`` and ``policy/secret_classifier.py``), and the
whole premise is that a tap on a remote-paired phone is *not* enough.

It used to be enough. The enforcement in :func:`approvals.approval_service.
decide_approval` read ``approver["type"]`` — a string out of the request body
— and the phone sends ``{'type': 'desktop_app'}``. A self-asserted claim
cannot enforce anything, so the control only ever held against a client that
chose to be honest.

The server does not need to be told. It already knows which door the request
came through:

* :func:`routes.deps.verify_api_key` gates ``/api/approvals/*`` — a paired
  client, phone or otherwise, possibly across a tunnel. That is
  :data:`CHANNEL_REMOTE`.
* :func:`routes.deps.require_local_access` gates ``/api/dashboard/agent/*``,
  and that router is registered only in ``routes.__init__._DASHBOARD_ONLY_ROUTERS``
  — the localhost-bound dashboard app, never the tunnel-exposed API app. That
  is :data:`CHANNEL_DESKTOP`.

So the channel is derived at the route boundary and passed down as an
argument. Client-supplied ``approver`` metadata is still carried into the
audit trail, because "which device answered" is worth recording — it just
cannot *promote* itself. :func:`stamp_approver` is the seam: it keeps the
metadata, moves the client's own claim to ``claimed_type`` (so an attempted
promotion is visible to a reviewer rather than erased), and overwrites
``type`` and ``channel`` with the server-observed truth.

Fail closed: anything that is not exactly :data:`CHANNEL_DESKTOP` is remote.
"""

from __future__ import annotations

from typing import Any

#: Decision made on the localhost-only dashboard listener — someone at the
#: machine. The only channel that may answer a ``desktop_only`` approval.
CHANNEL_DESKTOP = "desktop"

#: Decision made by a paired client over the API listener (the phone), or by
#: an in-process caller that did not state a channel. Never desktop.
CHANNEL_REMOTE = "remote"

#: Keys :func:`stamp_approver` owns. A client that sends any of them has them
#: discarded, so no request body can shape the authoritative fields.
_SERVER_OWNED_KEYS = ("type", "channel", "claimed_type")

_CHANNEL_APPROVER_TYPES = {
    CHANNEL_DESKTOP: "desktop_app",
    CHANNEL_REMOTE: "remote_client",
}


def normalize_channel(channel: str | None) -> str:
    """The channel as one of the two constants, defaulting to remote.

    Unknown, empty and ``None`` all collapse to :data:`CHANNEL_REMOTE`: a
    channel we cannot identify is not the desktop.
    """
    if isinstance(channel, str) and channel.strip().lower() == CHANNEL_DESKTOP:
        return CHANNEL_DESKTOP
    return CHANNEL_REMOTE


def is_desktop_channel(channel: str | None) -> bool:
    """Whether ``channel`` may answer a ``desktop_only`` approval."""
    return normalize_channel(channel) == CHANNEL_DESKTOP


def stamp_approver(
    approver: dict[str, Any] | None,
    *,
    channel: str,
) -> dict[str, Any]:
    """Client approver metadata with the server-observed channel stamped on.

    Everything the client sent survives except the keys the server owns:
    device names, ids and user ids still reach the audit event. The client's
    ``type`` is preserved as ``claimed_type`` — that is what makes a phone
    claiming ``desktop_app`` legible in the audit log instead of silently
    normalised away.
    """
    normalized = normalize_channel(channel)
    stamped: dict[str, Any] = {}
    claimed_type: Any = None

    if isinstance(approver, dict):
        for key, value in approver.items():
            if key in _SERVER_OWNED_KEYS:
                if key == "type":
                    claimed_type = value
                continue
            stamped[key] = value

    if isinstance(claimed_type, str) and claimed_type.strip():
        stamped["claimed_type"] = claimed_type.strip()

    stamped["type"] = _CHANNEL_APPROVER_TYPES[normalized]
    stamped["channel"] = normalized
    return stamped


__all__ = [
    "CHANNEL_DESKTOP",
    "CHANNEL_REMOTE",
    "is_desktop_channel",
    "normalize_channel",
    "stamp_approver",
]
