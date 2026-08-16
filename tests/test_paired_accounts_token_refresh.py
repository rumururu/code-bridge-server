"""Paired accounts must not write to Firestore with a stale ID token.

Regression cover for the Firestore failure diagnosed as TRACK_D D4. Firebase
ID tokens last one hour. ``update_url_for_all_accounts`` used to send
``account.id_token`` verbatim, however old it was, so on the live server every
paired account failed once it had been paired for more than an hour:

    401 UNAUTHENTICATED  "Missing or invalid authentication."   (recent expiry)
    403 PERMISSION_DENIED "Missing or insufficient permissions." (old expiry)

Both are the same defect. Which one Google returns depends only on whether the
token's signing key is still published in Google's JWKS: while it is, the token
is recognised and rejected (401); once it has rotated out the signature cannot
be checked at all, the request is treated as anonymous, and the security rules
deny it (403).

Every test here passes an explicit ``storage_path`` inside a temporary
directory. The module default is ``~/.code-bridge/paired_accounts.json`` — the
operator's real paired accounts — and a test that touched it would corrupt
live pairings.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import jwt

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from firebase import paired_accounts as pa  # noqa: E402
from firebase.token_manager import TOKEN_REFRESH_THRESHOLD_SECONDS  # noqa: E402

PROJECT = "code-bridge-test"
SERVER_ID = "server-abc"
API_KEY = "test-api-key"


def token_expiring_in(seconds: float, uid: str = "uid-1") -> str:
    """Build a decodable ID token. Only the ``exp`` claim is ever read."""
    exp = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return jwt.encode({"sub": uid, "exp": int(exp.timestamp())}, "unused-secret-of-sufficient-length-for-hmac-sha256")


class PairedAccountTokenRefreshTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage = Path(self._tmp.name) / "paired_accounts.json"

    def make_manager(self, id_token, refresh_token="refresh-1") -> pa.PairedAccountsManager:
        manager = pa.PairedAccountsManager(
            storage_path=self.storage,
            firebase_config={"projectId": PROJECT, "apiKey": API_KEY},
        )
        manager.add_or_update_account(
            user_id="uid-1",
            email="user@example.com",
            id_token=id_token,
            refresh_token=refresh_token,
        )
        return manager

    async def update(self, manager) -> dict:
        return await manager.update_url_for_all_accounts(
            project_id=PROJECT,
            server_id=SERVER_ID,
            api_key=API_KEY,
            tunnel_url="https://tunnel.example",
            local_url="http://192.168.0.2:8767",
        )

    # -- the token that is still good ------------------------------------

    async def test_fresh_token_is_used_as_is_without_refreshing(self):
        good = token_expiring_in(3600)
        manager = self.make_manager(good)
        manager.refresh_account_token = AsyncMock(return_value="should-not-be-used")

        with patch.object(pa.device_registration, "register_device", new=AsyncMock(return_value=True)) as reg:
            results = await self.update(manager)

        self.assertEqual(results, {"uid-1": True})
        manager.refresh_account_token.assert_not_awaited()
        self.assertEqual(reg.await_args.kwargs["id_token"], good)

    # -- the defect D4 identified ----------------------------------------

    async def test_expired_token_is_refreshed_before_the_write(self):
        """The live failure: a token that expired hours ago was sent anyway."""
        stale = token_expiring_in(-13 * 3600)  # mkideabox@, 13.7h past expiry
        manager = self.make_manager(stale)
        manager.refresh_account_token = AsyncMock(return_value="fresh-token")

        with patch.object(pa.device_registration, "register_device", new=AsyncMock(return_value=True)) as reg:
            results = await self.update(manager)

        self.assertEqual(results, {"uid-1": True})
        manager.refresh_account_token.assert_awaited_once_with("uid-1", API_KEY)
        self.assertEqual(reg.await_count, 1)
        self.assertEqual(reg.await_args.kwargs["id_token"], "fresh-token")
        self.assertNotEqual(reg.await_args.kwargs["id_token"], stale)

    async def test_token_inside_the_refresh_threshold_is_refreshed(self):
        """Do not send a token that will expire mid-flight."""
        nearly = token_expiring_in(TOKEN_REFRESH_THRESHOLD_SECONDS - 30)
        manager = self.make_manager(nearly)
        manager.refresh_account_token = AsyncMock(return_value="fresh-token")

        with patch.object(pa.device_registration, "register_device", new=AsyncMock(return_value=True)) as reg:
            await self.update(manager)

        manager.refresh_account_token.assert_awaited_once()
        self.assertEqual(reg.await_args.kwargs["id_token"], "fresh-token")

    async def test_token_just_outside_the_threshold_is_kept(self):
        good = token_expiring_in(TOKEN_REFRESH_THRESHOLD_SECONDS + 120)
        manager = self.make_manager(good)
        manager.refresh_account_token = AsyncMock(return_value="fresh-token")

        with patch.object(pa.device_registration, "register_device", new=AsyncMock(return_value=True)) as reg:
            await self.update(manager)

        manager.refresh_account_token.assert_not_awaited()
        self.assertEqual(reg.await_args.kwargs["id_token"], good)

    async def test_non_jwt_token_is_refreshed_rather_than_sent(self):
        """A stored value that is not a JWT earns 401 ACCESS_TOKEN_TYPE_UNSUPPORTED.

        The live paired_accounts.json held exactly this: an account whose
        id_token was the 5-character string a test fixture had written.
        """
        manager = self.make_manager("token")
        manager.refresh_account_token = AsyncMock(return_value="fresh-token")

        with patch.object(pa.device_registration, "register_device", new=AsyncMock(return_value=True)) as reg:
            await self.update(manager)

        manager.refresh_account_token.assert_awaited_once()
        self.assertEqual(reg.await_args.kwargs["id_token"], "fresh-token")

    async def test_missing_token_is_refreshed(self):
        manager = self.make_manager(None)
        manager.refresh_account_token = AsyncMock(return_value="fresh-token")

        with patch.object(pa.device_registration, "register_device", new=AsyncMock(return_value=True)) as reg:
            await self.update(manager)

        manager.refresh_account_token.assert_awaited_once()
        self.assertEqual(reg.await_args.kwargs["id_token"], "fresh-token")

    # -- refresh-and-retry -----------------------------------------------

    async def test_rejected_valid_token_is_refreshed_and_retried_once(self):
        """Locally valid but rejected: revoked session, clock skew, key rotation."""
        good = token_expiring_in(3600)
        manager = self.make_manager(good)
        manager.refresh_account_token = AsyncMock(return_value="fresh-token")

        register = AsyncMock(side_effect=[False, True])
        with patch.object(pa.device_registration, "register_device", new=register):
            results = await self.update(manager)

        self.assertEqual(results, {"uid-1": True})
        self.assertEqual(register.await_count, 2)
        self.assertEqual(register.await_args_list[0].kwargs["id_token"], good)
        self.assertEqual(register.await_args_list[1].kwargs["id_token"], "fresh-token")
        manager.refresh_account_token.assert_awaited_once()

    async def test_no_second_refresh_when_the_token_was_just_refreshed(self):
        """A token minted seconds ago that is still rejected is not a token problem."""
        manager = self.make_manager(token_expiring_in(-3600))
        manager.refresh_account_token = AsyncMock(return_value="fresh-token")

        register = AsyncMock(return_value=False)
        with patch.object(pa.device_registration, "register_device", new=register):
            results = await self.update(manager)

        self.assertEqual(results, {"uid-1": False})
        self.assertEqual(register.await_count, 1, "must not retry after a proactive refresh")
        manager.refresh_account_token.assert_awaited_once()

    async def test_retry_is_skipped_when_the_refresh_fails(self):
        good = token_expiring_in(3600)
        manager = self.make_manager(good)
        manager.refresh_account_token = AsyncMock(return_value=None)

        register = AsyncMock(return_value=False)
        with patch.object(pa.device_registration, "register_device", new=register):
            results = await self.update(manager)

        self.assertEqual(results, {"uid-1": False})
        self.assertEqual(register.await_count, 1)

    async def test_deleted_firebase_user_is_skipped_not_retried(self):
        """USER_NOT_FOUND from securetoken: refresh returns None, so give up.

        One live account is in this state — its Firebase user no longer
        exists, so no amount of refreshing will help.
        """
        manager = self.make_manager(token_expiring_in(-1738 * 3600))
        manager.refresh_account_token = AsyncMock(return_value=None)

        register = AsyncMock(return_value=True)
        with patch.object(pa.device_registration, "register_device", new=register):
            results = await self.update(manager)

        self.assertEqual(results, {"uid-1": False})
        register.assert_not_awaited()

    # -- multi-account behaviour -----------------------------------------

    async def test_one_bad_account_does_not_block_the_others(self):
        manager = pa.PairedAccountsManager(
            storage_path=self.storage,
            firebase_config={"projectId": PROJECT, "apiKey": API_KEY},
        )
        manager.add_or_update_account("uid-good", "a@example.com", token_expiring_in(3600), "r1")
        manager.add_or_update_account("uid-dead", "b@example.com", token_expiring_in(-9999), "r2")

        async def refresh(user_id, api_key):
            return None if user_id == "uid-dead" else "fresh-token"

        manager.refresh_account_token = AsyncMock(side_effect=refresh)

        with patch.object(pa.device_registration, "register_device", new=AsyncMock(return_value=True)):
            results = await self.update(manager)

        self.assertEqual(results, {"uid-good": True, "uid-dead": False})

    async def test_refreshing_mid_loop_does_not_disturb_iteration(self):
        """refresh_account_token replaces the stored account object."""
        manager = pa.PairedAccountsManager(
            storage_path=self.storage,
            firebase_config={"projectId": PROJECT, "apiKey": API_KEY},
        )
        for i in range(3):
            manager.add_or_update_account(f"uid-{i}", f"{i}@example.com", token_expiring_in(-3600), f"r{i}")

        # Use the genuine refresh path's side effect on _accounts.
        async def refresh(user_id, api_key):
            acc = manager._accounts[user_id]
            manager._accounts[user_id] = acc.with_updated_token(id_token=token_expiring_in(3600))
            return manager._accounts[user_id].id_token

        manager.refresh_account_token = AsyncMock(side_effect=refresh)

        with patch.object(pa.device_registration, "register_device", new=AsyncMock(return_value=True)):
            results = await self.update(manager)

        self.assertEqual(results, {"uid-0": True, "uid-1": True, "uid-2": True})

    # -- the helper in isolation -----------------------------------------

    async def test_ensure_valid_token_reports_whether_it_refreshed(self):
        good = token_expiring_in(3600)
        manager = self.make_manager(good)
        manager.refresh_account_token = AsyncMock(return_value="fresh-token")

        self.assertEqual(await manager.ensure_valid_token("uid-1", API_KEY), (good, False))

        manager.add_or_update_account("uid-1", id_token=token_expiring_in(-60))
        self.assertEqual(await manager.ensure_valid_token("uid-1", API_KEY), ("fresh-token", True))

    async def test_ensure_valid_token_on_unknown_account(self):
        manager = self.make_manager(token_expiring_in(3600))
        self.assertEqual(await manager.ensure_valid_token("nobody", API_KEY), (None, False))

    async def test_storage_path_is_the_temporary_one(self):
        """Guard the guard: these tests must never touch ~/.code-bridge."""
        manager = self.make_manager(token_expiring_in(3600))
        self.assertEqual(manager._storage_path, self.storage)
        self.assertTrue(self.storage.exists())
        self.assertNotIn(".code-bridge", str(self.storage))


if __name__ == "__main__":
    unittest.main()
