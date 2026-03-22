import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import system_status_service


class SystemStatusServiceTest(unittest.TestCase):
    def test_get_health_status_for_current_server(self):
        result = system_status_service.get_health_status_for_current_server()
        self.assertEqual(result, {"status": "ok", "service": "claude-bridge"})

    def test_get_debug_port_snapshot_for_current_server(self):
        fake_config = SimpleNamespace(dashboard_port=8080, api_port=8081, _runtime_port=9090)

        result = system_status_service.get_debug_port_snapshot_for_current_server(
            config=fake_config,
            env={"CODEBRIDGE_DASHBOARD_PORT": "7000", "CODEBRIDGE_API_PORT": "7001"},
        )

        self.assertEqual(
            result,
            {
                "dashboard_port": 8080,
                "api_port": 8081,
                "env_dashboard_port": "7000",
                "env_api_port": "7001",
                "runtime_port": 9090,
            },
        )


if __name__ == "__main__":
    unittest.main()
