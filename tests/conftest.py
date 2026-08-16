"""Keep the test suite off the operator's live runtime.

Two runtime locations are resolved from ``$HOME``, not from anything the test
process controls:

* ``agent/push_notifier.py`` falls back to
  ``~/.code-bridge/firebase_service_account.json`` — the *production* FCM
  key — whenever ``CODEBRIDGE_FCM_SERVICE_ACCOUNT`` is unset.
* ``core/server_logging.py`` falls back to ``~/.code-bridge/logs/server.log``.

So on a machine that also runs the server, ``pytest tests/`` sent real push
notifications to the operator's paired phones and interleaved its output with
the live server log. That is exactly what happened on 2026-08-16: fixture
agents from ``test_cli_agent_sweep.py`` (named ``reviewer``) pushed
"reviewer lost its definition file" to a real device, repeatedly, because
several suite runs happened in parallel.

Pointing both at paths that cannot collide with the real ones makes the safe
behaviour the default. Tests that care about push specifically still set
``push_notifier.SERVICE_ACCOUNT_ENV`` themselves with ``mock.patch.dict``
(see ``test_notify_step_push.py``); this only changes what an *unset*
environment means, so those overrides keep working unchanged.

The key path deliberately points at a file that does not exist — a missing
key is how ``push_notifier`` turns push off (``_get_app`` returns ``None``),
and a missing key cannot be mistaken for a working one.

This runs at import time rather than in a fixture on purpose. pytest imports
``conftest.py`` before it collects any test module, whereas even a
session-scoped autouse fixture runs *after* collection — and a test module
that builds an app at import time would already have called
``configure_server_logging`` against the real path by then.

Since the structural fix (F3), both modules resolve their *defaults* lazily
through ``core.runtime_paths.runtime_path``, so ``CODEBRIDGE_APP_SUPPORT_DIR``
alone already redirects them off the operator's ``~/.code-bridge``. The two
explicit variables above stay anyway as defence in depth: they keep the suite
isolated even if a future edit reintroduces a direct ``$HOME`` fallback.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.gettempdir()) / "code-bridge-tests"
_SANDBOX.mkdir(parents=True, exist_ok=True)

# Must not exist: a missing key is what disables push.
os.environ.setdefault(
    "CODEBRIDGE_FCM_SERVICE_ACCOUNT",
    str(_SANDBOX / "no-such-service-account.json"),
)
os.environ.setdefault(
    "CODE_BRIDGE_SERVER_LOG_PATH",
    str(_SANDBOX / "server.log"),
)

# Everything resolved through ``core.runtime_paths.runtime_path`` — chiefly
# ``paired_accounts.json``, which holds real Firebase refresh tokens. Without
# this, ``test_remote_access_service.py`` rewrote the operator's live file on
# every run and left a fixture account (``user-1``, whose id_token is the
# literal string ``"token"``) in it; the server then failed that account's
# Firestore write with 401 UNAUTHENTICATED on every startup, which reads
# exactly like a product bug and cost real time to chase.
#
# Redirecting the whole support dir is deliberately broader than patching the
# one path that was caught: ``runtime_path`` is the shared resolver, so any
# future runtime file is isolated by construction rather than by somebody
# remembering to add it here.
os.environ.setdefault(
    "CODEBRIDGE_APP_SUPPORT_DIR",
    str(_SANDBOX / "app-support"),
)
