"""AP-3 — audit redaction categories.

Records an event whose payload contains an AWS access key and an email
address, then reads the persisted row back and asserts the
``redacted_categories`` column captures both category strings.
"""

import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from audit import audit_store
from audit.audit_redactor import (
    CATEGORY_AWS_KEY,
    CATEGORY_EMAIL,
    CATEGORY_GENERIC_SECRET,
    CATEGORY_JWT,
    CATEGORY_PRIVATE_KEY,
    redact_payload,
)
from core import database


class AuditRedactionCategoriesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = (
            Path(self._tmp.name) / "code_bridge_redaction_test.db"
        )
        audit_store._audit_store = None

    def tearDown(self) -> None:
        audit_store._audit_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # ---- core requirement: AWS key + email round-trip through the store ----

    def test_record_event_persists_aws_and_email_categories(self) -> None:
        store = audit_store.get_audit_store()
        event = store.record_event(
            operation="agent.test",
            payload={
                "command": "aws s3 ls",
                "stderr": "no creds for AKIAABCDEFGHIJKLMNOP",
                "note": "contact ops@example.com for rotation",
            },
        )

        categories = event.get("redacted_categories")
        self.assertIsInstance(categories, list)
        self.assertIn(CATEGORY_AWS_KEY, categories)
        self.assertIn(CATEGORY_EMAIL, categories)

        # Payload should have the sensitive substrings replaced.
        stderr = event["payload"]["stderr"]
        note = event["payload"]["note"]
        self.assertNotIn("AKIAABCDEFGHIJKLMNOP", stderr)
        self.assertIn(f"[redacted:{CATEGORY_AWS_KEY}]", stderr)
        self.assertNotIn("ops@example.com", note)
        self.assertIn(f"[redacted:{CATEGORY_EMAIL}]", note)

        # And a second read via list_events sees the same column.
        rows = store.list_events()
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            sorted(rows[0]["redacted_categories"]),
            sorted([CATEGORY_AWS_KEY, CATEGORY_EMAIL]),
        )

    def test_record_event_without_secrets_stores_empty_categories(self) -> None:
        store = audit_store.get_audit_store()
        event = store.record_event(
            operation="agent.test",
            payload={"command": "ls", "cwd": "/tmp"},
        )
        # No matches → column stored as NULL, surfaced as empty list.
        self.assertEqual(event.get("redacted_categories"), [])

    # ---- unit-level sanity on the redactor itself ----

    def test_redact_payload_detects_field_name_secrets(self) -> None:
        redacted, categories = redact_payload(
            {"api_key": "abcd1234", "nested": {"refresh_token": "xyz"}}
        )
        self.assertEqual(redacted["api_key"], "[redacted]")
        self.assertEqual(redacted["nested"]["refresh_token"], "[redacted]")
        self.assertIn(CATEGORY_GENERIC_SECRET, categories)

    def test_redact_payload_detects_jwt_and_private_key(self) -> None:
        jwt_value = (
            "eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiIxMjMifQ."
            "abcDEF123_signature"
        )
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA...\n"
            "-----END RSA PRIVATE KEY-----"
        )
        _, categories = redact_payload({"log": jwt_value, "key_pem": pem})
        self.assertIn(CATEGORY_JWT, categories)
        self.assertIn(CATEGORY_PRIVATE_KEY, categories)


if __name__ == "__main__":
    unittest.main()
