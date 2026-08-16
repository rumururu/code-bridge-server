import asyncio
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import agent.browser_action_adapter as browser_action_adapter  # noqa: E402
from agent.browser_action_adapter import PlaywrightBrowserActionAdapter  # noqa: E402


def test_playwright_browser_action_adapter_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(browser_action_adapter, "ARTIFACT_ROOT", tmp_path)
    # Keep the run out of the dev tree's real agent profile. `runtime_dir`
    # honours this, so the persistent profile lands under tmp_path — which also
    # pins that the profile really is where `dedicated_profile_dir` says.
    monkeypatch.setenv("CODEBRIDGE_APP_SUPPORT_DIR", str(tmp_path / "support"))
    html = """
    <!doctype html>
    <html>
      <head><title>Code Bridge Browser Smoke</title></head>
      <body>
        <main>
          <h1>Browser adapter smoke page</h1>
          <p id="status">Adapter ready</p>
          <section data-testid="payload">Extracted deterministic payload</section>
        </main>
      </body>
    </html>
    """
    url = f"data:text/html;charset=utf-8,{quote(html)}"
    actions = [
        {"type": "navigate", "url": url},
        {"type": "assert", "kind": "text_visible", "value": "Adapter ready"},
        {"type": "extract", "selector": "[data-testid='payload']"},
        {"type": "screenshot"},
    ]

    result = asyncio.run(
        PlaywrightBrowserActionAdapter().run_actions(
            actions,
            context={
                "run_id": "run_browser_smoke",
                "step_id": "step_browser_smoke",
                "browser_context_dir": str(tmp_path / "browser_context"),
            },
        )
    )

    if result.wait_reason == "browser_adapter_unavailable":
        detail = (result.error or {}).get("message") or "Playwright/Chromium unavailable"
        pytest.skip(f"Playwright/Chromium unavailable: {detail}")

    assert result.status == "completed", result.to_output()
    assert result.message == "Browser actions completed."
    assert len(result.observations) == len(actions)
    assert result.extracted == [
        {
            "action_index": 3,
            "selector": "[data-testid='payload']",
            "text": "Extracted deterministic payload",
        }
    ]
    assert len(result.screenshots) == 1
    screenshot = Path(result.screenshots[0])
    assert screenshot.exists()
    assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.storage_state_path
    storage_state = Path(result.storage_state_path)
    assert storage_state.exists()
    assert storage_state.name == "storage_state.json"

    # The profile the run used is a real one on disk, and it is the agent's own
    # — not the operator's Chrome. A browser with no profile is why a daily
    # "check the board" agent could never be logged in.
    profile_dir = tmp_path / "support" / "browser_profile"
    assert profile_dir.is_dir()
    assert any(profile_dir.iterdir()), "the browser wrote nothing into its profile"
