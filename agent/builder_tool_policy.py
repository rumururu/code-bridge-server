"""Which tools the Agent Builder conversation may call, and who decides.

The Configurator runs on the `claude_code` preset, so it can try to call
tools. For a long time nothing answered those calls at all: the session parked
on its permission future, the builder's reader ignored the event, and the turn
sat silent until the job timeout. That is fixed elsewhere; this module is about
what the *answer* should be.

Two rules, and the first is not negotiable by a setting.

**Only MCP tools are ever eligible.** A builder turn is a background job. The
preset's own tools — `Bash`, `Write`, `Edit` — are arbitrary code and arbitrary
writes on the user's machine, and no toggle should be able to hand those to a
design conversation. `mcp__*` is a different proposition: those servers exist
because the user configured them, each one for a stated purpose, and the same
registry already decides what a running agent may reach
(`capability_registry`). So the boundary is drawn by prefix, here, once.

**Eligible is not the same as allowed.** An eligible call is *asked about*: the
job parks, the client's poll surfaces the request, and a person answers. The
setting below decides whether the asking happens at all — with it off, an
eligible call is refused exactly like an ineligible one, so the default state
of a fresh install is that a design conversation calls nothing.

Nothing here approves anything on its own. `may_ask` is the whole of this
module's authority: it says whether a question may be put to the user.
"""

from __future__ import annotations

from typing import Any

#: Prefix the Claude Agent SDK gives every MCP-provided tool.
MCP_TOOL_PREFIX = "mcp__"

#: Setting key for "the design conversation may ask to use an MCP tool".
#: Off unless the user turns it on: a fresh install should not let a
#: background job reach a configured server without anyone saying so.
BUILDER_MCP_SETTING_KEY = "builder.allow_mcp_tools"


def is_mcp_tool(tool_name: Any) -> bool:
    """True for a tool provided by a configured MCP server."""
    return isinstance(tool_name, str) and tool_name.startswith(MCP_TOOL_PREFIX)


def builder_mcp_enabled() -> bool:
    """Whether the user has allowed the builder to ask about MCP tools."""
    # Imported here rather than at module scope: this module is read by the
    # prompt builder and the tests, and neither should open a database to
    # answer a question about a prefix.
    from core.database import get_settings_db

    raw = get_settings_db().get(BUILDER_MCP_SETTING_KEY, "false")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def set_builder_mcp_enabled(enabled: bool) -> None:
    from core.database import get_settings_db

    get_settings_db().set(BUILDER_MCP_SETTING_KEY, "true" if enabled else "false")


def may_ask(tool_name: Any) -> bool:
    """Whether this tool call may be put to the user as a question.

    False means refuse without asking — either because the tool is outside
    the boundary entirely, or because the user has not switched the asking on.
    """
    return is_mcp_tool(tool_name) and builder_mcp_enabled()


def refusal_reason(tool_name: Any) -> str:
    """What the model is told, and it has to be able to act on it.

    A bare "denied" invites a retry, and the retry is refused identically. So
    each branch says which wall was hit and what does work instead: ask the
    person for the fact, or design the check as a step in the agent, where it
    runs at execution time with a human able to answer.
    """
    if not is_mcp_tool(tool_name):
        return (
            "Only MCP tools can be used from the Agent Builder conversation, "
            f"and '{tool_name}' is not one. Shell commands and file writes are "
            "never available here — this turn is a background job. Do not try "
            "another one. Ask the person for anything you cannot know, or "
            "design it as a step in the agent so the check runs when the agent "
            "runs."
        )
    return (
        f"'{tool_name}' is an MCP tool, but using MCP tools from the Agent "
        "Builder conversation is switched off in this server's settings. Do "
        "not try another one. Ask the person for what you need, or design it "
        "as a step in the agent so the check runs when the agent runs."
    )
