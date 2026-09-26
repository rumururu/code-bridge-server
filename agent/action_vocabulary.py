"""The shape an executor uses to publish the actions it can run.

An executor is the only place that knows which keys it reads off an action —
`_wait` reads `timeout_ms`, `_extract` reads `name`/`pattern`/`source`. Every
other surface had to guess. The phone's action editor guessed `duration_ms`
and `state` for a wait and `label`/`output` for an extract: none of those four
keys is read by anything, so an action authored on the phone ran with the
default timeout and an extract saved no name, which is the same as not
extracting at all (no `{{binding}}` is ever created).

So each executor publishes one table of :class:`ActionType`, next to the
dispatch that reads these keys, and every other surface is generated from it:

* the Configurator's prompt block (an LLM cannot author what it was not told
  exists — see `browser_action_vocabulary_block`),
* the published step schema (`code_bridge_core.workflow_step_schema`), which is what the
  phone draws its editor from,
* a drift-guard test per executor that fails when a documented key is not one
  the dispatch actually reads.

Two kinds only: a line of text, or a number. Anything richer would have to be
rendered by every client, and there is no action parameter today that needs
more than those two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KIND_TEXT = "text"
KIND_NUMBER = "number"
ACTION_PARAM_KINDS: tuple[str, ...] = (KIND_TEXT, KIND_NUMBER)


def _localized(en: str, ko: str | None) -> dict[str, str]:
    """Publish `en`, and `ko` only when a translation actually exists.

    This used to fill a missing `ko` with the English text, so a translated
    entry and an untranslated one produced byte-identical payloads — a client
    rendered both as "Korean" and a developer inspecting the JSON could not
    tell a gap from a real (if coincidentally identical) translation. Omitting
    the key instead makes a gap visible — to `test_action_vocabulary_translations.py`
    below and to anyone reading the payload — while every current reader of
    this payload already falls back to `en` when `ko` is absent
    (`lib/models/workflow_step_schema.dart`'s `_localized`/`_localizedOrNull`,
    the flow-canvas registry's `pickLocalized`), so an untranslated entry still
    renders as English instead of blank or a raw key.
    """
    localized = {"en": en}
    if ko is not None:
        localized["ko"] = ko
    return localized


@dataclass(frozen=True)
class ActionParam:
    """One key an executor reads off an action.

    ``key`` is the literal JSON key — not a display name, not an alias. The
    drift guard checks this exact string against the executor's own source, so
    a parameter that no dispatch reads cannot be published to a client.
    """

    key: str
    label_en: str
    label_ko: str | None = None
    kind: str = KIND_TEXT
    required: bool = False
    help_en: str = ""
    help_ko: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ACTION_PARAM_KINDS:
            raise ValueError(f"unknown action param kind: {self.kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "required": self.required,
            "label": _localized(self.label_en, self.label_ko),
            "help": _localized(self.help_en, self.help_ko) if self.help_en else None,
        }


@dataclass(frozen=True)
class ActionType:
    """One action an executor dispatches, with the keys it reads for it.

    ``note`` is the prompt sentence — written for whoever authors the action
    (a model, or a person reading the block), so it says *why* you would use
    this action rather than restating its name. ``note_ko`` is the same
    sentence in Korean; it is the client-facing `help` text for this action
    (`note` itself only ever renders as prompt text for the Configurator,
    which is English-only), so a `None` here is a translation gap, not a
    deliberate absence.
    """

    type: str
    label_en: str
    note: str
    params: tuple[ActionParam, ...] = ()
    label_ko: str | None = None
    note_ko: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "label": _localized(self.label_en, self.label_ko),
            "help": _localized(self.note, self.note_ko),
            "params": [param.to_dict() for param in self.params],
        }

    @property
    def param_keys(self) -> tuple[str, ...]:
        return tuple(param.key for param in self.params)


def vocabulary_block(
    header: str,
    actions: tuple[ActionType, ...],
    *,
    not_executed: tuple[str, ...] = (),
    footer: tuple[str, ...] = (),
) -> str:
    """Render a vocabulary as the prompt text an action author reads."""
    width = max((len(action.type) for action in actions), default=0) + 2
    lines = [header]
    lines += [f"  {action.type:<{width}}{action.note}" for action in actions]
    if not_executed:
        lines.append(
            "  " + "/".join(not_executed)
            + "   NOT executed — a step using these stops and asks the user"
        )
    lines += [f"  {line}" for line in footer]
    return "\n".join(lines)


def vocabulary_payload(actions: tuple[ActionType, ...]) -> list[dict[str, Any]]:
    """The same vocabulary as JSON, for a client that draws an editor."""
    return [action.to_dict() for action in actions]


__all__ = [
    "ACTION_PARAM_KINDS",
    "KIND_NUMBER",
    "KIND_TEXT",
    "ActionParam",
    "ActionType",
    "vocabulary_block",
    "vocabulary_payload",
]
