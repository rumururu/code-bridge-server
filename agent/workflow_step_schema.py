"""Publishes ``WORKFLOW_STEP_SCHEMA`` as a form a client can render.

Every authoring surface used to hardcode its own list of workflow-step field
names (see AGENT_COMPOSITION_SPEC.md Phase 2, WORKFLOW_STEP_FIELDS_SPEC.md).
The phone's copy never learned about ``shell`` or ``notify`` and offered
``memory_read`` / ``memory_write`` / ``success_criteria`` on step types that
ignore them; every new step type had to be added by hand in the phone, the
dashboard and the Configurator prompt. This module is the single place that
turns :data:`agent.workflow_v2.WORKFLOW_STEP_SCHEMA` — the definition
``normalize_workflow_step`` already enforces — into presentation metadata
(kind, label, help text, where its options come from) a client can draw a
form from without knowing any field names. ``routes.agents`` publishes it
over HTTP; :mod:`agent.configurator` generates its prompt's schema block from
it.

The field list itself is never retyped here. ``_FIELD_PRESENTATION`` and
``_TYPE_PRESENTATION`` attach *metadata* to field/type names that
:data:`agent.workflow_v2.WORKFLOW_STEP_SCHEMA` and
:data:`agent.workflow_v2.ALLOWED_STEP_TYPES` already define; which fields
belong to which type is decided there and nowhere else. :func:`build_step_schema`
walks that definition and raises if it finds a field or type with no metadata
attached, so a field added to ``WORKFLOW_STEP_SCHEMA` without updating this
module fails loudly at schema-build time instead of silently rendering as a
raw key in some client, or not rendering at all.
"""

from __future__ import annotations

from typing import Any

from agent.workflow_v2 import (
    ALLOWED_STEP_TYPES,
    NOTIFY_LEVELS,
    WORKFLOW_STEP_SCHEMA,
)


class WorkflowSchemaDriftError(RuntimeError):
    """Raised when a field or type has no presentation metadata attached.

    This is the failure Phase 2 exists to make impossible to miss: a new
    workflow field or step type landing in ``workflow_v2`` without a matching
    entry here would otherwise ship silently — no label, no help text, no
    input kind — and every client reading the published schema would either
    drop the field or render it with nothing to say about what it does.
    """


# The small, closed set of input kinds a client needs to draw a form field
# without knowing its name. `action_list` is not one of the four scalar kinds
# (single-line text / multi-line text / a select / a list of strings) because
# `actions` is a list of *structured* action objects (`{"type": "navigate",
# "url": ...}`, `{"type": "click", ...}`, each with type-dependent params) —
# calling it "a list of strings" would tell a client to draw a bare text-entry
# list, which cannot represent it. It already has a dedicated editor on every
# surface that supports it today; this kind exists so the schema does not
# have to lie about what shape the field is, not so a client is expected to
# build a generic renderer for it yet.
KIND_TEXT = "text"
KIND_LONG_TEXT = "long_text"
KIND_SELECT = "select"
KIND_STRING_LIST = "string_list"
KIND_ACTION_LIST = "action_list"

ALL_KINDS: tuple[str, ...] = (
    KIND_TEXT,
    KIND_LONG_TEXT,
    KIND_SELECT,
    KIND_STRING_LIST,
    KIND_ACTION_LIST,
)

# Where a `select` field's options come from when they are not simply given
# inline (see `notify.level`, whose fixed set is small enough to embed). Each
# entry below is a real, already-existing endpoint — this module does not
# invent an option source that has nothing behind it.
OPTION_SOURCES: dict[str, dict[str, str]] = {
    "scripts": {
        # routes/scripts.py — the shell step's script_id catalog. Phone/API
        # callers use the api-key-gated path; the dashboard has no key (see
        # routes/dashboard_agents.py) and uses its own mirror instead.
        "endpoint": "/api/agent/scripts",
        "dashboard_endpoint": "/api/dashboard/agent/scripts",
    },
    "devices": {
        # routes/devices.py — connected Android devices. There is no
        # dashboard-only mirror of this one today; it is registered on both
        # apps (see routes/__init__.py _SHARED_ROUTERS) and, unlike the
        # scripts list, callers on localhost with IP-login enabled already
        # reach it without a key.
        "endpoint": "/api/devices",
    },
}


def _localized(en: str, ko: str) -> dict[str, str]:
    return {"en": en, "ko": ko}


class _FieldSpec:
    """Presentation metadata for one field key, independent of step type.

    A field means the same thing on every type it appears on (``tool_hint``
    does not change meaning between an ``llm`` step and an ``mcp_tool``
    step), so this is keyed once per field, not once per (type, field) pair —
    the type-to-field mapping stays solely in ``WORKFLOW_STEP_SCHEMA``.
    """

    __slots__ = ("kind", "required", "options_source", "options", "label", "help")

    def __init__(
        self,
        *,
        kind: str,
        required: bool,
        label: dict[str, str],
        help: dict[str, str],
        options_source: str | None = None,
        options: tuple[str, ...] | None = None,
    ) -> None:
        if kind not in ALL_KINDS:
            raise WorkflowSchemaDriftError(f"unknown field kind: {kind}")
        if options_source is not None and options_source not in OPTION_SOURCES:
            raise WorkflowSchemaDriftError(f"unknown option source: {options_source}")
        self.kind = kind
        self.required = required
        self.options_source = options_source
        self.options = options
        self.label = label
        self.help = help

    def to_dict(self, key: str) -> dict[str, Any]:
        return {
            "key": key,
            "kind": self.kind,
            "required": self.required,
            "options_source": self.options_source,
            "options": list(self.options) if self.options is not None else None,
            "label": dict(self.label),
            "help": dict(self.help),
        }


# One entry per field name that can appear in any type's
# `WORKFLOW_STEP_SCHEMA` set. `notify` is a nested object
# (`{title, body, level}`); it is expanded into three dotted keys
# (`notify.title` / `notify.body` / `notify.level`) so a client can render
# three plain inputs without parsing a nested object shape out of a field
# name — see `_expand_field` below. The dotted keys are therefore what needs
# an entry here, not the bare `notify` key.
_FIELD_PRESENTATION: dict[str, _FieldSpec] = {
    "script_id": _FieldSpec(
        kind=KIND_SELECT,
        required=True,
        options_source="scripts",
        label=_localized("Script to run", "실행할 스크립트"),
        help=_localized(
            "Only scripts already registered on this server can be picked here "
            "— the step can never carry a command line of its own.",
            "이 서버에 이미 등록된 스크립트만 선택할 수 있습니다. 스텝에 직접 "
            "명령어를 넣을 수는 없습니다.",
        ),
    ),
    "script_args": _FieldSpec(
        kind=KIND_STRING_LIST,
        required=False,
        label=_localized("Extra arguments for the script", "스크립트에 전달할 추가 인자"),
        help=_localized(
            "Each entry becomes one argument, passed to the script in order.",
            "각 항목이 순서대로 스크립트에 전달되는 인자가 됩니다.",
        ),
    ),
    "instruction": _FieldSpec(
        kind=KIND_LONG_TEXT,
        required=False,
        label=_localized("What the AI should do in this step", "이 스텝에서 AI가 할 일"),
        help=_localized(
            "The concrete instruction sent to the model when this step runs.",
            "이 스텝이 실행될 때 모델에 전달되는 구체적인 지시입니다.",
        ),
    ),
    "observation": _FieldSpec(
        kind=KIND_LONG_TEXT,
        required=False,
        label=_localized("What to look at before acting", "행동하기 전에 살펴볼 것"),
        help=_localized(
            "Evidence — earlier step output, logs, screenshots — the AI should "
            "read before deciding what to do.",
            "행동을 결정하기 전에 AI가 먼저 읽어야 할 근거(이전 스텝 결과, 로그, "
            "스크린샷 등)입니다.",
        ),
    ),
    "memory_read": _FieldSpec(
        kind=KIND_LONG_TEXT,
        required=False,
        label=_localized(
            "What to recall from the agent's memory", "에이전트 메모리에서 불러올 내용"
        ),
        help=_localized(
            "A query. Matching memories are fetched before the step runs and "
            "given to the AI as context.",
            "질의문입니다. 스텝 실행 전에 일치하는 메모리를 찾아 AI에게 참고 "
            "자료로 전달합니다.",
        ),
    ),
    "memory_write": _FieldSpec(
        kind=KIND_LONG_TEXT,
        required=False,
        label=_localized("What to remember after this step", "이 스텝이 끝난 뒤 기억할 내용"),
        help=_localized(
            "After the step completes, this is written to the agent's memory "
            "so a future run can read it back.",
            "스텝이 끝나면 이 내용을 에이전트 메모리에 저장하여 다음 실행에서 "
            "참고할 수 있게 합니다.",
        ),
    ),
    "success_criteria": _FieldSpec(
        kind=KIND_LONG_TEXT,
        required=False,
        label=_localized("How the AI knows it succeeded", "AI가 성공을 판단하는 기준"),
        help=_localized(
            "Reaches only this step's own AI prompt. It does not decide pass "
            "or fail — the step's own outcome does that.",
            "이 스텝의 AI 프롬프트에만 전달됩니다. 스텝의 성공/실패 자체를 "
            "판정하지는 않습니다.",
        ),
    ),
    "tool_hint": _FieldSpec(
        kind=KIND_TEXT,
        required=False,
        label=_localized(
            "Which tool this step is likely to use", "이 스텝이 사용할 가능성이 높은 도구"
        ),
        help=_localized(
            "Names one of the agent's attached tools by id, as a hint. There "
            "is no catalog to pick from here — type the id directly.",
            "에이전트에 연결된 도구 중 하나를 아이디로 알려주는 힌트입니다. "
            "목록에서 고르지 않고 직접 입력합니다.",
        ),
    ),
    "notify.title": _FieldSpec(
        kind=KIND_TEXT,
        required=False,
        label=_localized("Notification title", "알림 제목"),
        help=_localized(
            "Shown first in the inbox. Falls back to this step's name when "
            "left blank.",
            "인박스에 가장 먼저 표시되는 문구입니다. 비워두면 스텝 이름을 "
            "대신 사용합니다.",
        ),
    ),
    "notify.body": _FieldSpec(
        kind=KIND_LONG_TEXT,
        required=False,
        label=_localized("Notification message", "알림 내용"),
        help=_localized(
            "The detail shown once the notification is opened.",
            "알림을 열었을 때 표시되는 자세한 내용입니다.",
        ),
    ),
    "notify.level": _FieldSpec(
        kind=KIND_SELECT,
        required=False,
        options=NOTIFY_LEVELS,
        label=_localized("How urgent the notification is", "알림의 긴급도"),
        help=_localized(
            "Controls how the notification is highlighted in the inbox.",
            "인박스에서 이 알림이 강조되는 방식을 결정합니다.",
        ),
    ),
    "actions": _FieldSpec(
        kind=KIND_ACTION_LIST,
        required=False,
        label=_localized(
            "The ordered list of actions this step performs",
            "이 스텝이 순서대로 수행할 동작 목록",
        ),
        help=_localized(
            "A structured list (navigate, click, type, ...) — edited with a "
            "dedicated action editor, not a plain text or select field.",
            "구조화된 목록입니다(navigate, click, type 등). 일반 텍스트나 선택 "
            "목록이 아닌 전용 동작 편집기로 편집합니다.",
        ),
    ),
    "device_id": _FieldSpec(
        kind=KIND_SELECT,
        required=False,
        options_source="devices",
        label=_localized("Device to run on", "실행할 기기"),
        help=_localized(
            "One of the devices currently connected to this server. Ignored "
            "if android_device_id is also set.",
            "현재 이 서버에 연결된 기기 중 하나입니다. android_device_id가 함께 "
            "설정되면 그 값이 우선합니다.",
        ),
    ),
    "android_device_id": _FieldSpec(
        kind=KIND_SELECT,
        required=False,
        options_source="devices",
        label=_localized("Android device to run on", "실행할 Android 기기"),
        help=_localized(
            "One of the devices currently connected to this server. Takes "
            "priority over device_id when both are set.",
            "현재 이 서버에 연결된 기기 중 하나입니다. device_id와 함께 설정되면 "
            "이 값이 우선합니다.",
        ),
    ),
}

# The one field `WORKFLOW_STEP_SCHEMA` stores as a single nested key but that
# a form renders as several plain inputs. Expanding it here — instead of
# giving the client a nested-object kind to interpret — keeps the four
# scalar kinds sufficient for every leaf field a client actually draws.
_FIELD_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "notify": ("notify.title", "notify.body", "notify.level"),
}


def _expand_field(field: str) -> tuple[str, ...]:
    return _FIELD_EXPANSIONS.get(field, (field,))


class _TypeSpec:
    __slots__ = ("label", "help")

    def __init__(self, *, label: dict[str, str], help: dict[str, str]) -> None:
        self.label = label
        self.help = help


_TYPE_PRESENTATION: dict[str, _TypeSpec] = {
    "shell": _TypeSpec(
        label=_localized("Run a script", "스크립트 실행"),
        help=_localized(
            "Runs one registered script. Deterministic — no model call, no "
            "tokens, and the same input always gives the same output.",
            "등록된 스크립트 하나를 실행합니다. 모델 호출이나 토큰 소모 없이 "
            "항상 같은 입력에 같은 결과를 냅니다.",
        ),
    ),
    "llm": _TypeSpec(
        label=_localized("Ask the AI", "AI에게 맡기기"),
        help=_localized(
            "Opens a chat turn with the model to do judgement work — "
            "deciding, summarizing, diagnosing.",
            "모델과 대화를 열어 판단이 필요한 작업(결정, 요약, 진단 등)을 "
            "맡깁니다.",
        ),
    ),
    "notify": _TypeSpec(
        label=_localized("Send a notification", "알림 보내기"),
        help=_localized(
            "Leaves a message in the app inbox for whoever comes back to the "
            "run later.",
            "나중에 실행 결과를 확인할 사람을 위해 앱 인박스에 메시지를 "
            "남깁니다.",
        ),
    ),
    "browser_action": _TypeSpec(
        label=_localized("Automate a browser", "브라우저 자동화"),
        help=_localized(
            "Drives a browser on the server machine through a fixed list of "
            "actions.",
            "서버 기기의 브라우저를 정해진 동작 목록에 따라 조작합니다.",
        ),
    ),
    "app_action": _TypeSpec(
        label=_localized("Automate an app", "앱 자동화"),
        help=_localized(
            "Drives an app on a connected device through a fixed list of "
            "actions.",
            "연결된 기기의 앱을 정해진 동작 목록에 따라 조작합니다.",
        ),
    ),
    "android_action": _TypeSpec(
        label=_localized("Automate an Android device", "Android 기기 자동화"),
        help=_localized(
            "Drives a connected Android device directly (adb) through a "
            "fixed list of actions.",
            "연결된 Android 기기를 adb로 직접 조작하며, 정해진 동작 목록을 "
            "수행합니다.",
        ),
    ),
    "mobile_action": _TypeSpec(
        label=_localized("Automate a mobile device", "모바일 기기 자동화"),
        help=_localized(
            "Drives a connected mobile device through a fixed list of "
            "actions.",
            "연결된 모바일 기기를 정해진 동작 목록에 따라 조작합니다.",
        ),
    ),
    "device_action": _TypeSpec(
        label=_localized("Automate a device", "기기 자동화"),
        help=_localized(
            "Drives a connected device through a fixed list of actions.",
            "연결된 기기를 정해진 동작 목록에 따라 조작합니다.",
        ),
    ),
    "mcp_tool": _TypeSpec(
        label=_localized("Call an MCP tool", "MCP 도구 호출"),
        help=_localized(
            "Invokes one tool from an MCP the agent has attached.",
            "에이전트에 연결된 MCP 중 하나의 도구를 호출합니다.",
        ),
    ),
    "manual_handoff": _TypeSpec(
        label=_localized("Hand off to a person", "사람에게 넘기기"),
        help=_localized(
            "Pauses the run until a person completes something outside the "
            "agent's reach, then resumes it.",
            "에이전트가 할 수 없는 작업을 사람이 처리할 때까지 실행을 "
            "멈췄다가 다시 이어갑니다.",
        ),
    ),
    "approval_gate": _TypeSpec(
        label=_localized("Wait for approval", "승인 대기"),
        help=_localized(
            "Pauses the run until a person explicitly approves continuing.",
            "사람이 명시적으로 승인할 때까지 실행을 멈춥니다.",
        ),
    ),
    "condition": _TypeSpec(
        label=_localized("Branch", "조건 분기"),
        help=_localized(
            "Routes to a different step depending on on_success / "
            "on_failure, without doing any work itself.",
            "직접 작업을 수행하지 않고 on_success/on_failure에 따라 다른 "
            "스텝으로 분기합니다.",
        ),
    ),
}

# Presentation order only — never which fields a type has (that is decided
# solely by `WORKFLOW_STEP_SCHEMA`). A field missing from this tuple is not
# dropped: `build_step_schema` appends anything absent here, alphabetically,
# after everything listed. So forgetting to add a new field to this tuple
# degrades its position in the form, not its presence.
_FIELD_DISPLAY_ORDER: tuple[str, ...] = (
    "script_id",
    "script_args",
    "notify.title",
    "notify.body",
    "notify.level",
    "instruction",
    "observation",
    "memory_read",
    "memory_write",
    "success_criteria",
    "tool_hint",
    "actions",
    "device_id",
    "android_device_id",
)


def _sort_key(field: str) -> tuple[int, str]:
    try:
        return (_FIELD_DISPLAY_ORDER.index(field), "")
    except ValueError:
        return (len(_FIELD_DISPLAY_ORDER), field)


def build_step_schema(
    step_schema: dict[str, frozenset[str]] | None = None,
    step_types: Any = None,
) -> dict[str, Any]:
    """Build the published schema from ``WORKFLOW_STEP_SCHEMA``.

    ``step_schema`` / ``step_types`` are overridable only so tests can prove
    the drift guard actually fires (pass a fake schema with an unmetadata'd
    field and expect :class:`WorkflowSchemaDriftError`); production code
    should call this with no arguments and get the real definition.
    """

    schema = WORKFLOW_STEP_SCHEMA if step_schema is None else step_schema
    types = ALLOWED_STEP_TYPES if step_types is None else step_types

    types_out: list[dict[str, Any]] = []
    for step_type in sorted(types):
        type_spec = _TYPE_PRESENTATION.get(step_type)
        if type_spec is None:
            raise WorkflowSchemaDriftError(
                f"step type '{step_type}' has no presentation metadata "
                "(_TYPE_PRESENTATION) — add a label/help entry before "
                "publishing the schema"
            )

        raw_fields = schema.get(step_type, frozenset())
        expanded_fields: list[str] = []
        for field in raw_fields:
            expanded_fields.extend(_expand_field(field))
        expanded_fields.sort(key=_sort_key)

        fields_out: list[dict[str, Any]] = []
        for field in expanded_fields:
            field_spec = _FIELD_PRESENTATION.get(field)
            if field_spec is None:
                raise WorkflowSchemaDriftError(
                    f"field '{field}' on step type '{step_type}' has no "
                    "presentation metadata (_FIELD_PRESENTATION) — add a "
                    "kind/label/help entry before publishing the schema"
                )
            fields_out.append(field_spec.to_dict(field))

        types_out.append(
            {
                "type": step_type,
                "label": dict(type_spec.label),
                "help": dict(type_spec.help),
                "fields": fields_out,
            }
        )

    return {
        "types": types_out,
        "option_sources": {name: dict(value) for name, value in OPTION_SOURCES.items()},
        "kinds": list(ALL_KINDS),
    }


def get_step_schema() -> dict[str, Any]:
    """The schema the running server publishes — real definitions, no overrides."""

    return build_step_schema()


def base_field_types(
    step_schema: dict[str, frozenset[str]] | None = None,
    step_types: Any = None,
) -> dict[str, list[str]]:
    """Un-expanded ``WORKFLOW_STEP_SCHEMA`` field name -> sorted step types.

    Unlike :func:`build_step_schema`, ``notify`` stays a single key here
    instead of splitting into ``notify.title`` / ``notify.body`` /
    ``notify.level`` — for a caller that needs the real wire field name
    (the Configurator's drafted JSON has one ``notify`` object, not three
    top-level keys) rather than a per-input-widget breakdown.
    """

    schema = WORKFLOW_STEP_SCHEMA if step_schema is None else step_schema
    types = ALLOWED_STEP_TYPES if step_types is None else step_types
    result: dict[str, list[str]] = {}
    for step_type in sorted(types):
        for field_name in schema.get(step_type, frozenset()):
            result.setdefault(field_name, []).append(step_type)
    return result


def field_help_en(field_key: str) -> str:
    """English one-line help for a scalar field (not ``notify`` — see :data:`agent.workflow_v2.NOTIFY_LEVELS`
    for its nested shape instead)."""

    spec = _FIELD_PRESENTATION.get(field_key)
    if spec is None:
        raise WorkflowSchemaDriftError(
            f"field '{field_key}' has no presentation metadata (_FIELD_PRESENTATION)"
        )
    return spec.help["en"]


__all__ = [
    "ALL_KINDS",
    "KIND_ACTION_LIST",
    "KIND_LONG_TEXT",
    "KIND_SELECT",
    "KIND_STRING_LIST",
    "KIND_TEXT",
    "OPTION_SOURCES",
    "WorkflowSchemaDriftError",
    "base_field_types",
    "build_step_schema",
    "field_help_en",
    "get_step_schema",
]
