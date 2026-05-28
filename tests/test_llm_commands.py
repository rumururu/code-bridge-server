import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm import llm_commands


class LlmCommandsTest(unittest.TestCase):
    def tearDown(self):
        llm_commands.clear_llm_command_cache()

    def test_global_scope_returns_project_client_action(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ):
            snapshot = llm_commands.get_llm_command_snapshot(scope="global", refresh=True)

        project = next(item for item in snapshot["commands"] if item["name"] == "/project")
        self.assertEqual(project["execution"], "client_action")
        self.assertEqual(project["client_action"], "project_picker_tutorial")
        self.assertTrue(project["enabled"])

    def test_discovers_provider_slash_commands_from_help_but_marks_non_executable(self):
        completed = Mock(
            stdout="Commands:\n  /custom reset context\n",
            stderr="",
            returncode=0,
        )
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/claude"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=completed,
        ):
            snapshot = llm_commands.get_llm_command_snapshot(scope="project", refresh=True)

        provider_names = {
            item["name"]: item
            for item in snapshot["commands"]
            if item.get("source") == "cli"
        }
        self.assertIn("/custom", provider_names)
        self.assertFalse(provider_names["/custom"]["enabled"])
        self.assertEqual(provider_names["/custom"]["execution"], "provider_slash")
        self.assertFalse(snapshot["capabilities"]["slash_commands_executable"])

    def test_code_bridge_commands_have_action_metadata(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value=None):
            snapshot = llm_commands.get_llm_command_snapshot(scope="project", refresh=True)

        commands = {item["name"]: item for item in snapshot["commands"]}
        self.assertEqual(commands["/resume"]["execution"], "client_action")
        self.assertEqual(commands["/resume"]["client_action"], "session_picker")
        self.assertEqual(commands["/clear"]["execution"], "server_action")
        self.assertEqual(commands["/clear"]["server_action"], "clear")
        self.assertEqual(commands["/fix"]["execution"], "prompt_action")
        self.assertEqual(commands["/fix"]["prompt_action"], "insert_template")

    def test_execute_status_returns_structured_payload(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={
                "selected": {"company_id": "anthropic", "model": "sonnet"},
                "companies": [
                    {
                        "id": "anthropic",
                        "name": "Claude",
                        "command": "claude",
                        "connected": True,
                        "selectable": True,
                    }
                ],
            },
        ), patch("llm.llm_commands.shutil.which", return_value=None):
            result = llm_commands.execute_llm_command(
                name="/status",
                scope="project",
                project_name="demo",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution"], "server_action")
        self.assertEqual(result["server_action"], "status")
        self.assertEqual(result["payload"]["project"]["name"], "demo")

    def test_resume_requires_project_and_returns_session_picker_action(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value=None):
            missing_project = llm_commands.execute_llm_command(name="/resume", scope="project")
            result = llm_commands.execute_llm_command(
                name="/resume",
                scope="project",
                project_name="demo",
            )

        self.assertFalse(missing_project["success"])
        self.assertEqual(missing_project["execution"], "disabled")
        self.assertTrue(result["success"])
        self.assertEqual(result["client_action"], "session_picker")
        self.assertEqual(result["payload"]["project_name"], "demo")

    def test_does_not_parse_inline_slash_mentions_as_commands(self):
        completed = Mock(
            stdout=(
                "Options:\n"
                "  -r, --resume [value] Resume a session, or open /resume picker\n"
                "  Skills still resolve via /skill-name.\n"
                "  See docs at https://example.test/docs/cli.\n"
                "  Project path: /tmp/code_bridge\n"
            ),
            stderr="",
            returncode=0,
        )
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/claude"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=completed,
        ):
            snapshot = llm_commands.get_llm_command_snapshot(scope="project", refresh=True)

        self.assertFalse(any(item.get("source") == "cli" for item in snapshot["commands"]))
        self.assertEqual(snapshot["source"], "code_bridge")

    def test_parses_only_standalone_slash_command_help_rows(self):
        parsed = llm_commands._parse_slash_commands(
            "\x1b[32m/clear\x1b[0m reset the current conversation\n"
            "- /model choose a model\n"
            "  /compact      summarize and continue\n"
            "Use /inline in prose only.\n"
            "Path: /usr/local/bin/tool\n"
        )

        self.assertEqual(
            parsed,
            [
                ("/clear", "reset the current conversation"),
                ("/compact", "summarize and continue"),
                ("/model", "choose a model"),
            ],
        )

    def test_cache_is_scoped_by_provider_and_can_be_refreshed(self):
        first = Mock(stdout="/clear reset context\n", stderr="", returncode=0)
        second = Mock(stdout="/model choose model\n", stderr="", returncode=0)
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/claude"), patch(
            "llm.llm_commands.subprocess.run",
            side_effect=[first, second],
        ) as run:
            cached = llm_commands.get_llm_command_snapshot(scope="project", refresh=True)
            repeated = llm_commands.get_llm_command_snapshot(scope="project")
            refreshed = llm_commands.get_llm_command_snapshot(scope="project", refresh=True)

        self.assertEqual(run.call_count, 2)
        self.assertIn("/clear", {item["name"] for item in cached["commands"]})
        self.assertIn("/clear", {item["name"] for item in repeated["commands"]})
        self.assertIn("/model", {item["name"] for item in refreshed["commands"]})

    def test_execute_status_returns_markdown_message(self):
        snapshot = {
            "selected": {"company_id": "anthropic", "model": "sonnet"},
            "companies": [
                {
                    "id": "anthropic",
                    "name": "Anthropic",
                    "command": "claude",
                    "connected": True,
                    "selectable": True,
                }
            ],
        }
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value=snapshot,
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/claude"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=Mock(stdout="", stderr="", returncode=0),
        ):
            result = llm_commands.execute_llm_command(
                name="/status",
                provider_id="anthropic",
                model="sonnet",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["display"], "markdown")
        self.assertIn("Code Bridge status", result["message"])

    def test_execute_resume_returns_client_action(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/claude"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=Mock(stdout="", stderr="", returncode=0),
        ):
            result = llm_commands.execute_llm_command(
                name="/resume",
                provider_id="anthropic",
                model="sonnet",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["client_action"], "session_picker")
        self.assertEqual(result["payload"]["project_name"], "CodeBridge")

    def test_openai_project_scope_includes_codex_specific_mappings(self):
        options = {
            "selected": {"company_id": "openai", "model": "gpt-5"},
            "companies": [
                {
                    "id": "openai",
                    "name": "OpenAI (Codex)",
                    "command": "codex",
                    "connected": True,
                    "selectable": True,
                    "settings": {"sandbox_mode": "workspace-write", "sandbox_modes": []},
                }
            ],
        }
        with patch("llm.llm_commands.get_llm_options_snapshot", return_value=options), patch(
            "llm.llm_commands.shutil.which",
            return_value="/usr/local/bin/codex",
        ), patch("llm.llm_commands.subprocess.run", return_value=Mock(stdout="", stderr="", returncode=0)):
            snapshot = llm_commands.get_llm_command_snapshot(
                provider_id="openai",
                scope="project",
                refresh=True,
            )

        commands = {item["name"]: item for item in snapshot["commands"]}
        self.assertEqual(commands["/review"]["execution"], "prompt_action")
        self.assertEqual(commands["/search"]["server_action"], "codex_search")
        self.assertEqual(commands["/sandbox"]["server_action"], "permissions")
        self.assertEqual(commands["/approval"]["server_action"], "permissions")
        self.assertEqual(commands["/features"]["server_action"], "codex_features")
        self.assertEqual(commands["/mcp"]["server_action"], "codex_mcp")
        self.assertEqual(commands["/plugins"]["server_action"], "codex_plugins")
        self.assertFalse(commands["/fork"]["enabled"])
        self.assertFalse(commands["/apply"]["enabled"])
        self.assertFalse(commands["/cloud"]["enabled"])

    def test_openai_codex_specific_server_actions_return_guidance(self):
        options = {
            "selected": {"company_id": "openai", "model": "gpt-5"},
            "companies": [
                {
                    "id": "openai",
                    "name": "OpenAI (Codex)",
                    "command": "codex",
                    "connected": True,
                    "selectable": True,
                    "settings": {"sandbox_mode": "workspace-write", "sandbox_modes": []},
                }
            ],
        }
        with patch("llm.llm_commands.get_llm_options_snapshot", return_value=options), patch(
            "llm.llm_commands.shutil.which",
            return_value="/usr/local/bin/codex",
        ), patch("llm.llm_commands.subprocess.run", return_value=Mock(stdout="", stderr="", returncode=0)):
            sandbox = llm_commands.execute_llm_command(
                name="/sandbox",
                provider_id="openai",
                model="gpt-5",
                scope="project",
                project_name="CodeBridge",
            )
            search = llm_commands.execute_llm_command(
                name="/search",
                provider_id="openai",
                model="gpt-5",
                scope="project",
                project_name="CodeBridge",
            )
            features = llm_commands.execute_llm_command(
                name="/features",
                provider_id="openai",
                model="gpt-5",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(sandbox["success"])
        self.assertEqual(sandbox["server_action"], "permissions")
        self.assertEqual(sandbox["payload"]["sandbox_mode"], "workspace-write")
        self.assertIn("Codex web search", search["message"])
        self.assertEqual(search["payload"]["launch_flag"], "--search")
        self.assertEqual(features["server_action"], "codex_features")

    def test_openai_unsupported_codex_mappings_are_disabled(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "openai", "model": "gpt-5"}},
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/codex"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=Mock(stdout="", stderr="", returncode=0),
        ):
            result = llm_commands.execute_llm_command(
                name="/apply",
                provider_id="openai",
                model="gpt-5",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["execution"], "disabled")
        self.assertIn("destructive", result["message"])

    def test_claude_specific_commands_are_registered_for_anthropic_only(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value=None):
            anthropic = llm_commands.get_llm_command_snapshot(
                provider_id="anthropic",
                scope="project",
                refresh=True,
            )

        llm_commands.clear_llm_command_cache()
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "openai", "model": "o3"}},
        ), patch("llm.llm_commands.shutil.which", return_value=None):
            openai = llm_commands.get_llm_command_snapshot(
                provider_id="openai",
                scope="project",
                refresh=True,
            )

        anthropic_commands = {item["name"]: item for item in anthropic["commands"]}
        openai_names = {item["name"] for item in openai["commands"]}
        self.assertEqual(anthropic_commands["/agents"]["server_action"], "claude_agents")
        self.assertEqual(anthropic_commands["/auth"]["server_action"], "claude_auth")
        self.assertEqual(anthropic_commands["/plugins"]["server_action"], "claude_plugins")
        self.assertEqual(anthropic_commands["/auto-mode"]["server_action"], "claude_auto_mode")
        self.assertEqual(anthropic_commands["/continue"]["client_action"], "session_picker")
        self.assertFalse(anthropic_commands["/mcp"]["enabled"])
        self.assertFalse(anthropic_commands["/skills"]["enabled"])
        self.assertNotIn("/agents", openai_names)
        self.assertNotIn("/auth", openai_names)

    def test_execute_claude_readonly_agents_command(self):
        completed = Mock(stdout="reviewer\nplanner\n", stderr="", returncode=0)
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/claude"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=completed,
        ) as run:
            result = llm_commands.execute_llm_command(
                name="/agents",
                provider_id="anthropic",
                model="sonnet",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["server_action"], "claude_agents")
        self.assertIn("reviewer", result["message"])
        run.assert_called_with(
            ["claude", "agents"],
            stdout=llm_commands.subprocess.PIPE,
            stderr=llm_commands.subprocess.PIPE,
            text=True,
            timeout=6,
            check=False,
        )

    def test_execute_claude_auth_reports_cli_failure(self):
        completed = Mock(stdout="", stderr="not logged in", returncode=1)
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/claude"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=completed,
        ):
            result = llm_commands.execute_llm_command(
                name="/auth",
                provider_id="anthropic",
                model="sonnet",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["server_action"], "claude_auth")
        self.assertIn("not logged in", result["message"])

    def test_execute_disabled_claude_mcp_returns_reason_without_running_cli(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/claude"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=Mock(stdout="", stderr="", returncode=0),
        ) as run:
            result = llm_commands.execute_llm_command(
                name="/mcp",
                provider_id="anthropic",
                model="sonnet",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["execution"], "disabled")
        self.assertIn("MCP stdio servers", result["message"])
        run.assert_called_once()

    def test_execute_claude_continue_returns_session_picker_payload(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "anthropic", "model": "sonnet"}},
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/claude"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=Mock(stdout="", stderr="", returncode=0),
        ):
            result = llm_commands.execute_llm_command(
                name="/continue",
                provider_id="anthropic",
                model="sonnet",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["client_action"], "session_picker")
        self.assertEqual(result["payload"]["project_name"], "CodeBridge")

    def test_google_project_scope_includes_gemini_specific_mappings(self):
        options = {
            "selected": {"company_id": "google", "model": "gemini-2.5-pro"},
            "companies": [
                {
                    "id": "google",
                    "name": "Google (Gemini)",
                    "command": "gemini",
                    "connected": True,
                    "selectable": True,
                }
            ],
        }
        with patch("llm.llm_commands.get_llm_options_snapshot", return_value=options), patch(
            "llm.llm_commands.shutil.which",
            return_value="/usr/local/bin/gemini",
        ), patch("llm.llm_commands.subprocess.run", return_value=Mock(stdout="", stderr="", returncode=0)):
            snapshot = llm_commands.get_llm_command_snapshot(
                provider_id="google",
                scope="project",
                refresh=True,
            )

        commands = {item["name"]: item for item in snapshot["commands"]}
        self.assertFalse(commands["/resume"]["enabled"])
        self.assertEqual(commands["/mcp"]["server_action"], "gemini_mcp_list")
        self.assertEqual(commands["/skills"]["server_action"], "gemini_skills_list")
        self.assertEqual(commands["/extensions"]["server_action"], "gemini_extensions_list")
        self.assertEqual(commands["/gemma"]["server_action"], "gemini_gemma_status")
        self.assertEqual(commands["/approval"]["server_action"], "gemini_approval")
        self.assertEqual(commands["/policy"]["server_action"], "gemini_policy")
        self.assertFalse(commands["/sessions"]["enabled"])
        self.assertFalse(commands["/hooks"]["enabled"])

    def test_execute_google_mcp_uses_gemini_readonly_command(self):
        completed = Mock(stdout="No MCP servers configured\n", stderr="", returncode=0)
        options = {
            "selected": {"company_id": "google", "model": "gemini-2.5-pro"},
            "companies": [
                {
                    "id": "google",
                    "name": "Google (Gemini)",
                    "command": "gemini",
                    "connected": True,
                    "selectable": True,
                }
            ],
        }
        with patch("llm.llm_commands.get_llm_options_snapshot", return_value=options), patch(
            "llm.llm_commands.shutil.which",
            return_value="/usr/local/bin/gemini",
        ), patch("llm.llm_commands.subprocess.run", return_value=completed) as run:
            result = llm_commands.execute_llm_command(
                name="/mcp",
                provider_id="google",
                model="gemini-2.5-pro",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["server_action"], "gemini_mcp_list")
        self.assertIn("No MCP servers configured", result["message"])
        run.assert_called_with(
            ["gemini", "mcp", "list"],
            stdout=llm_commands.subprocess.PIPE,
            stderr=llm_commands.subprocess.PIPE,
            text=True,
            timeout=6,
            check=False,
        )

    def test_execute_google_approval_returns_gemini_modes(self):
        options = {
            "selected": {"company_id": "google", "model": "gemini-2.5-pro"},
            "companies": [
                {
                    "id": "google",
                    "name": "Google (Gemini)",
                    "command": "gemini",
                    "connected": True,
                    "selectable": True,
                }
            ],
        }
        with patch("llm.llm_commands.get_llm_options_snapshot", return_value=options), patch(
            "llm.llm_commands.shutil.which",
            return_value="/usr/local/bin/gemini",
        ), patch("llm.llm_commands.subprocess.run", return_value=Mock(stdout="", stderr="", returncode=0)):
            result = llm_commands.execute_llm_command(
                name="/approval",
                provider_id="google",
                model="gemini-2.5-pro",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["server_action"], "gemini_approval")
        self.assertIn("plan", result["payload"]["modes"])

    def test_gemini_commands_include_readonly_mappings_and_disabled_launch_features(self):
        with patch(
            "llm.llm_commands.get_llm_options_snapshot",
            return_value={"selected": {"company_id": "google", "model": "gemini-2.5-pro"}},
        ), patch(
            "llm.llm_commands._discover_provider_commands",
            return_value=([], llm_commands._base_capabilities(False, "No slash commands were found in CLI help output.")),
        ):
            snapshot = llm_commands.get_llm_command_snapshot(
                provider_id="google",
                model="gemini-2.5-pro",
                scope="project",
                refresh=True,
            )

        commands = {item["name"]: item for item in snapshot["commands"]}
        self.assertFalse(commands["/resume"]["enabled"])
        self.assertIn("project-path-safe Gemini session adapter", commands["/resume"]["disabled_reason"])
        self.assertEqual(commands["/mcp"]["server_action"], "gemini_mcp_list")
        self.assertEqual(commands["/skills"]["server_action"], "gemini_skills_list")
        self.assertEqual(commands["/extensions"]["server_action"], "gemini_extensions_list")
        self.assertEqual(commands["/gemma"]["server_action"], "gemini_gemma_status")
        self.assertEqual(commands["/approval"]["server_action"], "gemini_approval")
        self.assertEqual(commands["/policy"]["server_action"], "gemini_policy")
        self.assertFalse(commands["/sessions"]["enabled"])
        self.assertFalse(commands["/worktree"]["enabled"])
        self.assertFalse(commands["/raw-output"]["enabled"])

    def test_execute_gemini_readonly_list_command_returns_output(self):
        options = {
            "selected": {"company_id": "google", "model": "gemini-2.5-pro"},
            "companies": [
                {
                    "id": "google",
                    "name": "Gemini",
                    "command": "gemini",
                    "connected": True,
                    "selectable": True,
                }
            ],
        }
        with patch("llm.llm_commands.get_llm_options_snapshot", return_value=options), patch(
            "llm.llm_commands._discover_provider_commands",
            return_value=([], llm_commands._base_capabilities(False, "No slash commands were found in CLI help output.")),
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/gemini"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=Mock(stdout="server-a enabled\n", stderr="", returncode=0),
        ) as run:
            result = llm_commands.execute_llm_command(
                name="/mcp",
                provider_id="google",
                model="gemini-2.5-pro",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["server_action"], "gemini_mcp_list")
        self.assertIn("server-a enabled", result["message"])
        run.assert_called_once()

    def test_execute_gemini_gemma_status_preserves_nonzero_status_output(self):
        options = {"selected": {"company_id": "google", "model": "gemini-2.5-pro"}}
        with patch("llm.llm_commands.get_llm_options_snapshot", return_value=options), patch(
            "llm.llm_commands._discover_provider_commands",
            return_value=([], llm_commands._base_capabilities(False, "No slash commands were found in CLI help output.")),
        ), patch("llm.llm_commands.shutil.which", return_value="/usr/local/bin/gemini"), patch(
            "llm.llm_commands.subprocess.run",
            return_value=Mock(stdout="Server: not running\n", stderr="", returncode=1),
        ):
            result = llm_commands.execute_llm_command(
                name="/gemma",
                provider_id="google",
                model="gemini-2.5-pro",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["server_action"], "gemini_gemma_status")
        self.assertEqual(result["payload"]["returncode"], 1)
        self.assertIn("Server: not running", result["message"])

    def test_execute_gemini_approval_reports_modes_without_subprocess(self):
        options = {"selected": {"company_id": "google", "model": "gemini-2.5-pro"}}
        with patch("llm.llm_commands.get_llm_options_snapshot", return_value=options), patch(
            "llm.llm_commands._discover_provider_commands",
            return_value=([], llm_commands._base_capabilities(False, "No slash commands were found in CLI help output.")),
        ):
            result = llm_commands.execute_llm_command(
                name="/approval",
                provider_id="google",
                model="gemini-2.5-pro",
                scope="project",
                project_name="CodeBridge",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["payload"]["modes"], ["default", "auto_edit", "yolo", "plan"])
        self.assertFalse(result["payload"]["live_mutation_supported"])


if __name__ == "__main__":
    unittest.main()
