"""An empty key store must not silently replace real pairings.

`_load_api_keys` empties the in-memory store on any read failure, and every
ordinary save then writes that emptiness to disk. One unreadable moment was
enough to unpair every device at once — and the phones cannot tell a wiped
server from a rejected key. They just stop working and offer a retry that can
never succeed; recovery is re-pairing each device by QR.

Revoking the last client is a genuinely empty store, so that path says so.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from pairing.pairing_service import PairingService


class KeyStoreGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.file = self.dir / "api_keys.json"
        self.service = PairingService(config_dir=self.dir)

    def _seed(self, count: int = 2) -> None:
        keys = {
            f"client_{i}": {"device_name": f"Phone {i}", "api_key_sha256": f"hash{i}"}
            for i in range(count)
        }
        self.file.write_text(json.dumps(keys))
        self.service._api_keys = dict(keys)

    def test_normal_save_writes_through(self):
        self._seed()
        self.service._save_api_keys()
        self.assertEqual(len(json.loads(self.file.read_text())), 2)

    def test_empty_store_does_not_wipe_the_file(self):
        self._seed()
        self.service._api_keys = {}  # what a failed load leaves behind
        self.service._save_api_keys()
        self.assertEqual(
            len(json.loads(self.file.read_text())),
            2,
            "paired clients were destroyed by an empty in-memory store",
        )

    def test_the_recovered_clients_come_back_into_memory(self):
        # Otherwise the process keeps rejecting every phone until it restarts.
        self._seed()
        self.service._api_keys = {}
        self.service._save_api_keys()
        self.assertEqual(len(self.service._api_keys), 2)

    def test_a_backup_is_left_behind(self):
        self._seed()
        self.service._api_keys = {}
        self.service._save_api_keys()
        backups = list(self.dir.glob("api_keys.*.recovered.json"))
        self.assertTrue(backups, "no recovery copy was written")
        self.assertEqual(len(json.loads(backups[0].read_text())), 2)

    def test_revoking_the_last_client_is_allowed_to_empty_it(self):
        self._seed(count=1)
        result = self.service.revoke_client("client_0")
        self.assertTrue(result.success)
        self.assertEqual(json.loads(self.file.read_text()), {})

    def test_empty_store_over_an_already_empty_file_is_fine(self):
        self.file.write_text("{}")
        self.service._api_keys = {}
        self.service._save_api_keys()
        self.assertEqual(json.loads(self.file.read_text()), {})

    def test_first_ever_save_still_works(self):
        self.service._api_keys = {}
        self.service._save_api_keys()
        self.assertTrue(self.file.exists())


if __name__ == "__main__":
    unittest.main()
