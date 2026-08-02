"""Paired and connected are different facts, and the payload must say both.

A stored API key means a phone was paired once. Whether it is reachable now is
`last_used` against the heartbeat window. The dashboard spine used to judge
both from the key count, so a phone that was switched off, uninstalled, or out
of range still read as connected — the one state anyone opens the dashboard to
check. The client list beside it, reading `is_connected`, disagreed.

These pin the field the UI now depends on, so the two can never drift apart
again without a test saying so.
"""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from pairing.pairing_models import PairingClientStatus, PairingStatus, _now_ts
from remote.heartbeat_settings import get_heartbeat_interval


def _client(*, seconds_ago: float | None, name: str = "SM-F966N") -> PairingClientStatus:
    return PairingClientStatus(
        client_id="client_1",
        device_name=name,
        paired_at=_now_ts() - 86400,
        last_used=None if seconds_ago is None else _now_ts() - seconds_ago,
    )


class IsConnectedTest(unittest.TestCase):
    def test_a_phone_that_just_talked_is_connected(self):
        self.assertTrue(_client(seconds_ago=5).is_connected())

    def test_a_phone_silent_since_yesterday_is_not(self):
        # The case that shipped: paired months ago, last seen 19 hours back,
        # still drawn as connected.
        self.assertFalse(_client(seconds_ago=19.5 * 3600).is_connected())

    def test_the_window_is_the_heartbeat_interval_plus_a_buffer(self):
        window = (get_heartbeat_interval() + 5) * 60
        self.assertTrue(_client(seconds_ago=window - 30).is_connected())
        self.assertFalse(_client(seconds_ago=window + 30).is_connected())

    def test_a_client_that_never_reported_is_not_connected(self):
        self.assertFalse(_client(seconds_ago=None).is_connected())

    def test_an_explicit_threshold_wins(self):
        client = _client(seconds_ago=600)
        self.assertTrue(client.is_connected(threshold_seconds=900))
        self.assertFalse(client.is_connected(threshold_seconds=300))


class PayloadTest(unittest.TestCase):
    def test_the_response_carries_is_connected(self):
        # The dashboard reads this key by name; renaming it would silently turn
        # every phone grey.
        fields = _client(seconds_ago=5).as_response_fields()
        self.assertIn("is_connected", fields)
        self.assertIs(fields["is_connected"], True)

    def test_paired_count_is_not_a_connected_count(self):
        """`active_clients` counts stored keys, which is not liveness.

        Nothing stops a caller using it as one — that is exactly the mistake
        this guards — so the two are asserted to disagree here on purpose.
        """
        offline = _client(seconds_ago=19.5 * 3600)
        status = PairingStatus(
            server_id="server_1",
            active_clients=1,
            pending_tokens=0,
            clients=[offline],
        )
        payload = status.as_response_fields()

        self.assertEqual(payload["active_clients"], 1)
        self.assertEqual(len(payload["clients"]), 1)
        self.assertFalse(payload["clients"][0]["is_connected"])

    def test_a_mixed_fleet_reports_each_phone_on_its_own(self):
        status = PairingStatus(
            server_id="server_1",
            active_clients=2,
            pending_tokens=0,
            clients=[
                _client(seconds_ago=5, name="SM-F966N"),
                _client(seconds_ago=19.5 * 3600, name="SM-M205N"),
            ],
        )
        connected = [c for c in status.as_response_fields()["clients"] if c["is_connected"]]
        self.assertEqual([c["device_name"] for c in connected], ["SM-F966N"])


if __name__ == "__main__":
    unittest.main()
