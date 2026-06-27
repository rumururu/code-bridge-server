"""Code Bridge secret store package.

The package owns mutations to ``~/.code-bridge/.env``. The boot loader at
``server.main._load_env_file_once`` only *reads* this file; it stays
untouched. From now on, all writes (upsert / delete) go through
:mod:`secret_store_pkg.secret_store` so the file format, file permissions
and ``os.environ`` propagation stay consistent.

Module name note: the package is intentionally **not** named
``secrets`` because that name shadows the Python standard library
``secrets`` module (used by ``server/dashboard``, ``server/preview`` and
``server/pairing``). Importing ``server/secrets/__init__.py`` would
silently replace stdlib ``secrets`` and break ``secrets.token_urlsafe``
everywhere else in the server.
"""

from . import secret_models, secret_store

__all__ = ["secret_models", "secret_store"]
