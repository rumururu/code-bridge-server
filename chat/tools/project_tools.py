"""AI tools for project management.

These tool definitions can be used:
1. As context hints for AI to understand project configuration
2. As API documentation for AI to recommend updates
3. Future: As inline tools if LLM session supports custom tools

Note: Currently, Claude Code and Codex manage their own tools. These definitions
serve as documentation and context for AI to recommend actions.
"""

from typing import Any

from core.database import get_project_db


# Tool definition (Claude API tools format)
# Currently used for documentation/context, not direct API tool registration
UPDATE_PROJECT_SETTINGS_TOOL = {
    "name": "update_project_settings",
    "description": (
        "Update project dev server settings. Use when you detect the project "
        "uses a different package manager (pnpm, yarn, bun) or needs a custom "
        "dev command. For example, if pnpm-lock.yaml exists, set command to 'pnpm dev'. "
        "This can be done via API: PUT /api/projects/{project_name} with JSON body "
        "containing dev_server.command field."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Name of the project to update",
            },
            "dev_server_command": {
                "type": "string",
                "description": (
                    "Dev server command (e.g., 'pnpm dev', 'cd frontend && npm run dev')"
                ),
            },
            "dev_server_port": {
                "type": "integer",
                "description": "Optional: Expected port number",
            },
        },
        "required": ["project_name", "dev_server_command"],
    },
}


async def execute_update_project_settings(input_data: dict[str, Any]) -> dict[str, Any]:
    """Execute update_project_settings tool.

    This function is available for direct execution if needed, but primarily
    the AI should recommend users update via the app UI or API.
    """
    project_name = input_data.get("project_name")
    command = input_data.get("dev_server_command")
    port = input_data.get("dev_server_port")

    if not project_name or not command:
        return {"success": False, "error": "project_name and dev_server_command required"}

    db = get_project_db()
    if not db.exists(project_name):
        return {"success": False, "error": f"Project '{project_name}' not found"}

    update: dict[str, Any] = {"dev_server": {"command": command}}
    if port:
        update["dev_server"]["port"] = port

    updated = db.update(project_name, update)
    return {
        "success": True,
        "message": f"Dev server command set to '{command}' for {project_name}",
        "project": updated,
    }


def get_project_tools_context() -> str:
    """Get context string describing available project tools for AI.

    This can be injected into system prompts to help AI understand
    what project management capabilities are available.
    """
    return """## Project Management Capabilities

You can help users configure their project's dev server settings:

### Update Dev Server Command
If you detect a mismatch between the project's package manager (from lockfiles)
and the configured dev command, suggest updating it:

**Via App UI:** Project Settings → Dev Server Command
**Via API:** PUT /api/projects/{project_name}
```json
{
  "dev_server": {
    "command": "pnpm dev"
  }
}
```

### Package Manager Detection
Lockfile precedence for detecting package manager:
1. pnpm-lock.yaml → pnpm
2. yarn.lock → yarn
3. bun.lockb → bun
4. package-lock.json → npm

### Common Dev Commands
- pnpm: `pnpm dev` or `pnpm run dev`
- yarn: `yarn dev` or `yarn run dev`
- npm: `npm run dev`
- bun: `bun run dev`
- Monorepo with subdirectory: `cd frontend && pnpm dev`
"""
