import json
import sys
import unittest
from pathlib import Path

from fastapi.responses import JSONResponse

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.result_response import as_flagged_response


class ResultResponseHelpersTest(unittest.TestCase):
    def test_as_flagged_response_returns_payload_on_success(self):
        payload = {"success": True, "message": "ok"}

        result = as_flagged_response(payload)

        self.assertEqual(result, payload)

    def test_as_flagged_response_returns_json_response_on_failure(self):
        payload = {"success": False, "error": "boom"}

        result = as_flagged_response(payload, error_status_code=422)

        self.assertIsInstance(result, JSONResponse)
        assert isinstance(result, JSONResponse)
        self.assertEqual(result.status_code, 422)
        self.assertEqual(json.loads(result.body), payload)

    def test_as_flagged_response_treats_missing_success_key_as_failure(self):
        payload = {"error": "missing success flag"}

        result = as_flagged_response(payload)

        self.assertIsInstance(result, JSONResponse)
        assert isinstance(result, JSONResponse)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(json.loads(result.body), payload)
