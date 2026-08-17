"""Linear flow_json <-> kernel graph converters (agent-flow-core T-B-02/T-B-03).

Contract (spec: agent-flow-core ``docs/LINEAR_FLOW_MAPPING.md``, T-B-01):

* **The linear flow_json is, and stays, the canon.** ``agents.flow_json``
  (as normalized by :func:`agent.workflow_v2.normalize_workflow`) is the
  single source of truth for an agent's workflow. The kernel
  :class:`agent_flow_core.model.Flow` produced by :func:`to_graph` is a
  *derived view*: edges are a pure function of the step policies
  (spec section 3), and :func:`from_graph` restores the linear list from the
  node policies alone — input edges are only *verified* against the edges
  re-derived from those policies (spec section 4). A graph whose edges and
  policies disagree is rejected, never silently corrected.

* ``to_graph`` input is always the output of ``normalize_workflow`` — raw
  stored shapes (legacy string policies etc.) are not accepted here;
  normalization is ``normalize_workflow``'s job alone (spec section 1).

* ``from_graph`` rejects every graph outside the linear+goto subset with
  :class:`UnsupportedTopologyError`, carrying the *full* issue list in the
  kernel ``FlowIssue`` shape with the ``linear.*`` codes of spec section 6.3
  (all violations at once, never fail-fast on the first).

.. warning::
   **Do not import this module from any existing runner/route path yet**
   (task orchestrator, workflow runtime, routes, dashboard). It imports
   ``agent_flow_core`` at module top, and the *deployed* server venv does
   not have the kernel installed — an import from a live code path would
   crash the server at startup. Wiring this module into read views is
   T-B-04 and into the write path T-B-05; until then only tests may import
   it.

Deliberate deviation from the spec (reported, not silent):

* Spec sections 2/6.1 exclude all seven ``COMMON_STEP_FIELDS`` — including
  the legacy alias key ``step_type`` — from ``config``. But
  ``normalize_workflow`` provably *keeps* a leftover ``step_type`` alias key
  in its output when the author used it (pinned by the
  ``step_type_alias_notify_mcp`` golden in
  ``tests/test_flow_json_snapshot_regression.py``), and the kernel
  ``FlowStep`` has no other slot for it — so excluding it from ``config``
  would break the spec's own round-trip proposition 1
  (``from_graph(to_graph(L)) == L``) for that stored shape. This module
  therefore lets a leftover ``step_type`` key ride in ``config`` verbatim
  and exempts it from the ``linear.config_field_collision`` check. The
  collision check's stated rationale (the fold would let config overwrite
  the folded key) does not apply to ``step_type``: the fold never writes a
  ``step_type`` key, so nothing is overwritten.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent_flow_core.model import (
    Flow,
    FlowEdge,
    FlowStep,
    UnknownStepTypeError,
    validate_step_types,
)
from agent_flow_core.policy import (
    WorkflowNormalizationError as KernelPolicyError,
    normalize_failure_policy,
    normalize_success_policy,
)
from agent_flow_core.validate import FlowIssue

from .workflow_v2 import (
    ALLOWED_STEP_TYPES,
    WorkflowNormalizationError,
    normalize_workflow,
)

__all__ = ["UnsupportedTopologyError", "to_graph", "from_graph"]


# The step keys that map onto dedicated FlowStep fields; everything else
# rides in ``config`` verbatim. Note ``step_type`` (the legacy alias key a
# normalized step can still carry) is deliberately NOT in this set — see the
# module docstring's deviation note.
_FOLDED_STEP_KEYS = frozenset(
    {"id", "type", "name", "description", "on_failure", "on_success"}
)

# Keys that would be overwritten if they appeared in ``config`` when the
# fold rebuilds ``{**common, **config}`` — spec 6.3
# ``linear.config_field_collision``. ``step_type`` exempted (deviation note).
_CONFIG_COLLISION_KEYS = _FOLDED_STEP_KEYS

_ALLOWED_EDGE_KINDS = frozenset({"seq", "goto"})
_ANNOTATION_KEYS = frozenset({"on", "via"})  # edge-level codeBridgeLinear
_EDGE_META_KEYS = frozenset({"on", "via", "kind"})  # legacy flow-level meta
_ALLOWED_ON = frozenset({"success", "failure"})
_ALLOWED_VIA = frozenset({"retry_then"})


class UnsupportedTopologyError(ValueError):
    """A graph cannot be folded back into the linear Code Bridge subset.

    ``issues`` carries every violation found (spec 6.2: "show everything",
    never stop at the first) as kernel-shaped :class:`FlowIssue` objects;
    ``detail`` serializes them camelCase for HTTP pass-through, mirroring
    ``agent_flow_core.validate.InvalidFlowError``.
    """

    def __init__(self, message: str, *, issues: list[FlowIssue]) -> None:
        super().__init__(message)
        self.issues: list[FlowIssue] = list(issues)
        self.detail: dict[str, Any] = {
            "errorCode": "UNSUPPORTED_TOPOLOGY",
            "message": message,
            "issues": [issue.model_dump(by_alias=True) for issue in issues],
        }


def _issue(
    code: str,
    message: str,
    *,
    step_id: str | None = None,
    edge_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> FlowIssue:
    return FlowIssue(
        code=code,
        severity="error",
        step_id=step_id,
        edge_id=edge_id,
        message=message,
        detail=detail or {},
    )


# ---------------------------------------------------------------------------
# Edge derivation (spec section 3) — shared by to_graph and from_graph so the
# two directions can never disagree about which edges a policy produces.
# ---------------------------------------------------------------------------


def _terminal_retry_goto(policy: dict[str, Any]) -> str | None:
    """Follow a ``retry`` policy's ``then`` chain to a terminal ``goto_step``.

    ``then`` is a single (recursively normalized) policy, so a
    retry -> retry -> goto escalation still ends in at most one jump
    (spec E4). Returns the target step id, or ``None`` when the chain ends
    in a non-goto policy (abort/ask_user/...).
    """

    current: Any = policy
    while isinstance(current, dict) and current.get("type") == "retry":
        current = current.get("then")
    if isinstance(current, dict) and current.get("type") == "goto_step":
        target = current.get("target_step_id")
        return target if isinstance(target, str) else None
    return None


def _derive_edges(
    entries: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]],
) -> list[dict[str, Any]]:
    """Derive the control-transfer edges from normalized step policies.

    ``entries`` is ``[(step_id, on_success, on_failure), ...]`` in linear
    order; a ``None`` policy pair member means "unverifiable, derive
    nothing for this step" (from_graph uses that for steps whose policies
    failed normalization — they already carry a ``policy.invalid`` error).

    Emission order is canonical (spec 7): step-list order, success edge
    before failure edge per step.
    """

    derived: list[dict[str, Any]] = []
    for index, (step_id, on_success, on_failure) in enumerate(entries):
        if on_success is not None:
            success_type = on_success.get("type")
            if success_type == "continue" and index + 1 < len(entries):
                # E1 — sequential progression to the next list index.
                derived.append(
                    {
                        "from": step_id,
                        "to": entries[index + 1][0],
                        "on": "success",
                        "kind": "seq",
                        "via": None,
                        "matched": False,
                    }
                )
            elif success_type == "goto_step":
                # E2 — success jump.
                derived.append(
                    {
                        "from": step_id,
                        "to": on_success.get("target_step_id"),
                        "on": "success",
                        "kind": "goto",
                        "via": None,
                        "matched": False,
                    }
                )
            # "end" (and a trailing continue) derive no edge — natural
            # termination lives in the node policy only (spec section 5).
        if on_failure is not None:
            failure_type = on_failure.get("type")
            if failure_type == "goto_step":
                # E3 — failure jump.
                derived.append(
                    {
                        "from": step_id,
                        "to": on_failure.get("target_step_id"),
                        "on": "failure",
                        "kind": "goto",
                        "via": None,
                        "matched": False,
                    }
                )
            elif failure_type == "retry":
                # E4 — retry escalation chain ending in a jump.
                target = _terminal_retry_goto(on_failure)
                if target is not None:
                    derived.append(
                        {
                            "from": step_id,
                            "to": target,
                            "on": "failure",
                            "kind": "goto",
                            "via": "retry_then",
                            "matched": False,
                        }
                    )
            # retry-without-goto / ask_user / manual_handoff / abort /
            # failure-continue: no edge (spec section 3, "엣지가 나오지 않는
            # 경우") — failure-continue moves to the same next step E1
            # already draws, and the "step stays failed" meaning lives in
            # the policy alone.
    return derived


# ---------------------------------------------------------------------------
# to_graph (T-B-02)
# ---------------------------------------------------------------------------


def to_graph(steps: list[dict[str, Any]]) -> Flow:
    """Represent a normalized linear workflow as a kernel :class:`Flow`.

    ``steps`` must be the output of
    :func:`agent.workflow_v2.normalize_workflow` (spec section 1). The
    result is a derived view: node policies stay the canon, every edge is
    re-derivable from them, and edge ids are deterministic
    (``{from_step_id}:success`` / ``{from_step_id}:failure``, spec 3.1).

    Flow ``name``/``description``/``triggers`` are left empty (schedules
    live in ``task_schedules``, outside flow_json) and ``Flow.extensions``
    is empty — edge annotations ride on the edges themselves
    (``FlowEdge.kind`` + ``FlowEdge.extensions["codeBridgeLinear"]``,
    spec 3.3 as revised by section 9).
    """

    if not isinstance(steps, list):
        raise TypeError("to_graph expects the normalized step list")

    flow_steps: list[FlowStep] = []
    for step in steps:
        config = {
            key: deepcopy(value)
            for key, value in step.items()
            if key not in _FOLDED_STEP_KEYS
        }
        flow_steps.append(
            FlowStep(
                id=step["id"],
                step_type=step["type"],
                name=step.get("name", ""),
                description=step.get("description", ""),
                config=config,
                on_failure=deepcopy(step.get("on_failure")),
                on_success=deepcopy(step.get("on_success")),
            )
        )

    entries = [
        (step["id"], step["on_success"], step["on_failure"]) for step in steps
    ]
    edges: list[FlowEdge] = []
    for derived in _derive_edges(entries):
        annotation: dict[str, Any] = {"on": derived["on"]}
        if derived["via"] is not None:
            annotation["via"] = derived["via"]
        edges.append(
            FlowEdge(
                id=f"{derived['from']}:{derived['on']}",
                from_step_id=derived["from"],
                to_step_id=derived["to"],
                kind=derived["kind"],
                extensions={"codeBridgeLinear": annotation},
            )
        )

    return Flow(steps=flow_steps, edges=edges)


# ---------------------------------------------------------------------------
# from_graph (T-B-03)
# ---------------------------------------------------------------------------


def from_graph(flow: Flow) -> list[dict[str, Any]]:
    """Fold a kernel :class:`Flow` back into the normalized linear list.

    Restores the linear workflow **from the node policies only** (spec
    section 4) — the linear order is ``flow.steps`` list order, never
    inferred from edges. The input edges are used for verification alone:
    the edges re-derived from the policies must match the input edges as a
    ``(from, to)`` multiset, and any present annotations
    (``kind`` / ``extensions.codeBridgeLinear`` / legacy flow-level
    ``edgeMeta``) must not contradict the derivation.

    Every graph outside the linear subset (spec 6.1) raises
    :class:`UnsupportedTopologyError` with the full ``linear.*`` /
    kernel-code issue list of spec 6.3. No silent approximation, no edge
    dropping, no policy auto-correction.
    """

    issues: list[FlowIssue] = []

    # --- flow envelope (spec 6.1 rule 4; name/description are ignored) ---
    if flow.triggers:
        issues.append(
            _issue(
                "linear.trigger_unsupported",
                "flow declares triggers; Code Bridge schedules live in"
                " task_schedules, outside flow_json",
                detail={"triggerCount": len(flow.triggers)},
            )
        )

    legacy_edge_meta: dict[str, Any] = {}
    for key in flow.extensions:
        if key != "codeBridgeLinear":
            issues.append(
                _issue(
                    "linear.flow_extensions_unsupported",
                    f"flow extensions key {key!r} has no slot in the linear"
                    " canon",
                    detail={"key": key},
                )
            )
    flow_annotation = flow.extensions.get("codeBridgeLinear")
    if flow_annotation is not None:
        if not isinstance(flow_annotation, dict):
            issues.append(
                _issue(
                    "linear.flow_extensions_unsupported",
                    "flow extensions codeBridgeLinear must be an object",
                    detail={"key": "codeBridgeLinear"},
                )
            )
        else:
            for key in flow_annotation:
                if key != "edgeMeta":
                    issues.append(
                        _issue(
                            "linear.flow_extensions_unsupported",
                            "flow extensions codeBridgeLinear key"
                            f" {key!r} is not a known annotation",
                            detail={"key": f"codeBridgeLinear.{key}"},
                        )
                    )
            raw_meta = flow_annotation.get("edgeMeta")
            if raw_meta is not None:
                if isinstance(raw_meta, dict):
                    legacy_edge_meta = raw_meta
                else:
                    issues.append(
                        _issue(
                            "linear.flow_extensions_unsupported",
                            "flow extensions codeBridgeLinear.edgeMeta must"
                            " be an object keyed by edge id",
                            detail={"key": "codeBridgeLinear.edgeMeta"},
                        )
                    )

    # --- step ids (kernel codes, spec 6.1 rule 1) -------------------------
    seen_ids: set[str] = set()
    for index, step in enumerate(flow.steps):
        sid = step.id.strip()
        if not sid:
            issues.append(
                _issue(
                    "step.id_required",
                    f"step at position {index} has an empty id",
                    step_id=f"#{index}",
                )
            )
        elif sid in seen_ids:
            issues.append(
                _issue(
                    "step.id_duplicate",
                    f"duplicate step id {sid!r}",
                    step_id=sid,
                )
            )
        else:
            seen_ids.add(sid)

    # --- kernel-only step fields (spec 6.1 rule 4) ------------------------
    for index, step in enumerate(flow.steps):
        location = step.id.strip() or f"#{index}"
        for field_name, is_default in (
            ("connectorRef", step.connector_ref is None),
            ("inputFields", not step.input_fields),
            ("extensions", not step.extensions),
        ):
            if not is_default:
                issues.append(
                    _issue(
                        "linear.step_field_unsupported",
                        f"step {location!r} sets kernel field {field_name!r};"
                        " the linear canon has no slot to preserve it",
                        step_id=location,
                        detail={"field": field_name},
                    )
                )
        for key in sorted(set(step.config) & _CONFIG_COLLISION_KEYS):
            issues.append(
                _issue(
                    "linear.config_field_collision",
                    f"step {location!r} config key {key!r} collides with a"
                    " common step field; the fold would overwrite it",
                    step_id=location,
                    detail={"key": key},
                )
            )

    # --- step-type allowlist (kernel check, CB vocabulary; rule 5) --------
    try:
        validate_step_types(flow.steps, ALLOWED_STEP_TYPES)
    except UnknownStepTypeError as exc:
        for step_id, step_type in exc.violations.items():
            issues.append(
                _issue(
                    "step.type_unknown",
                    f"step type {step_type!r} is not a Code Bridge step type",
                    step_id=step_id,
                    detail={"stepType": step_type},
                )
            )

    # --- policies (kernel normalizers = byte-identical to CB; rule 3/5) ---
    normalized_policies: list[
        tuple[dict[str, Any] | None, dict[str, Any] | None]
    ] = []
    unverifiable_steps: set[str] = set()
    for index, step in enumerate(flow.steps):
        location = step.id.strip() or f"#{index}"
        step_ok = True
        success_policy: dict[str, Any] | None = None
        failure_policy: dict[str, Any] | None = None
        try:
            success_policy = normalize_success_policy(step.on_success)
        except KernelPolicyError as exc:
            step_ok = False
            issues.append(
                _issue(
                    "policy.invalid",
                    f"on_success policy cannot be normalized: {exc}",
                    step_id=location,
                )
            )
        try:
            failure_policy = normalize_failure_policy(step.on_failure)
        except KernelPolicyError as exc:
            step_ok = False
            issues.append(
                _issue(
                    "policy.invalid",
                    f"on_failure policy cannot be normalized: {exc}",
                    step_id=location,
                )
            )
        if not step_ok:
            # Edge backing for this step is unverifiable; do not stack
            # speculative unbacked/missing issues on a known error.
            unverifiable_steps.add(step.id)
            normalized_policies.append((None, None))
            continue
        normalized_policies.append((success_policy, failure_policy))
        for attr, policy in (
            ("on_success", success_policy),
            ("on_failure", failure_policy),
        ):
            for target in _missing_goto_targets(policy, seen_ids):
                issues.append(
                    _issue(
                        "policy.goto_target_missing",
                        f"{attr} goto_step target {target!r} is not a known"
                        " step id",
                        step_id=location,
                        detail={"targetStepId": target},
                    )
                )

    # --- edge-vs-policy verification (spec section 4 / 6.1 rules 3+4) ----
    entries = [
        (step.id, policies[0], policies[1])
        for step, policies in zip(flow.steps, normalized_policies)
    ]
    expected = _derive_edges(entries)
    known_edge_ids = {edge.id for edge in flow.edges}
    for meta_edge_id in legacy_edge_meta:
        if meta_edge_id not in known_edge_ids:
            issues.append(
                _issue(
                    "linear.edge_meta_mismatch",
                    f"flow-level edgeMeta annotates edge {meta_edge_id!r},"
                    " which does not exist in the graph",
                    edge_id=meta_edge_id,
                )
            )

    for edge in flow.edges:
        if edge.from_field is not None or edge.to_field is not None:
            issues.append(
                _issue(
                    "linear.data_edge",
                    f"edge {edge.id!r} sets fromField/toField; Code Bridge"
                    " passes step data through run scope, not edges",
                    edge_id=edge.id,
                    detail={
                        "fromField": edge.from_field,
                        "toField": edge.to_field,
                    },
                )
            )

        # Edge-level annotation namespace (spec 3.3 / 6.1 rule 4).
        annotation: dict[str, Any] = {}
        for key in edge.extensions:
            if key != "codeBridgeLinear":
                issues.append(
                    _issue(
                        "linear.flow_extensions_unsupported",
                        f"edge {edge.id!r} extensions key {key!r} has no"
                        " slot in the linear canon",
                        edge_id=edge.id,
                        detail={"key": key},
                    )
                )
        raw_annotation = edge.extensions.get("codeBridgeLinear")
        if raw_annotation is not None:
            if not isinstance(raw_annotation, dict):
                issues.append(
                    _issue(
                        "linear.edge_meta_mismatch",
                        f"edge {edge.id!r} codeBridgeLinear annotation must"
                        " be an object",
                        edge_id=edge.id,
                    )
                )
            else:
                annotation = raw_annotation
                for key in annotation:
                    if key not in _ANNOTATION_KEYS:
                        issues.append(
                            _issue(
                                "linear.edge_meta_mismatch",
                                f"edge {edge.id!r} annotation key {key!r} is"
                                " not derivable from any policy",
                                edge_id=edge.id,
                                detail={"key": key},
                            )
                        )

        # Legacy flow-level meta for this edge (backward compat, spec 3.3).
        meta = legacy_edge_meta.get(edge.id)
        if meta is not None and not isinstance(meta, dict):
            issues.append(
                _issue(
                    "linear.edge_meta_mismatch",
                    f"flow-level edgeMeta entry for edge {edge.id!r} must"
                    " be an object",
                    edge_id=edge.id,
                )
            )
            meta = None
        meta = meta or {}
        for key in meta:
            if key not in _EDGE_META_KEYS:
                issues.append(
                    _issue(
                        "linear.edge_meta_mismatch",
                        f"flow-level edgeMeta key {key!r} for edge"
                        f" {edge.id!r} is not derivable from any policy",
                        edge_id=edge.id,
                        detail={"key": key},
                    )
                )

        # Collect explicit annotation claims from both levels. An edge-level
        # ``kind`` of "seq" is indistinguishable from an omitted kind (the
        # model default *is* the migration, spec 3.3/9), so only "goto" (or
        # an unknown value) counts as an explicit edge-level kind claim;
        # legacy edgeMeta values are always explicit.
        on_claims = {
            value
            for value in (annotation.get("on"), meta.get("on"))
            if value is not None
        }
        via_claims = {
            value
            for value in (annotation.get("via"), meta.get("via"))
            if value is not None
        }
        kind_claims: set[Any] = set()
        if edge.kind != "seq":
            kind_claims.add(edge.kind)
        if "kind" in meta:
            kind_claims.add(meta["kind"])

        claims_valid = True
        for claims, allowed, label in (
            (on_claims, _ALLOWED_ON, "on"),
            (via_claims, _ALLOWED_VIA, "via"),
            (kind_claims, _ALLOWED_EDGE_KINDS, "kind"),
        ):
            invalid = claims - allowed
            if invalid:
                claims_valid = False
                issues.append(
                    _issue(
                        "linear.edge_meta_mismatch",
                        f"edge {edge.id!r} annotation {label}="
                        f"{sorted(map(str, invalid))} is not a value any"
                        " derivation produces",
                        edge_id=edge.id,
                        detail={"key": label},
                    )
                )
            elif len(claims) > 1:
                claims_valid = False
                issues.append(
                    _issue(
                        "linear.edge_meta_mismatch",
                        f"edge {edge.id!r} carries contradictory {label}"
                        " annotations between the edge level and the"
                        " flow-level edgeMeta",
                        edge_id=edge.id,
                        detail={"key": label},
                    )
                )

        if edge.from_step_id in unverifiable_steps:
            # policy.invalid already reported; backing is unverifiable.
            continue

        candidates = [
            derived
            for derived in expected
            if not derived["matched"]
            and derived["from"] == edge.from_step_id
            and derived["to"] == edge.to_step_id
        ]
        if not candidates:
            issues.append(
                _issue(
                    "linear.edge_unbacked",
                    f"edge {edge.id!r} ({edge.from_step_id} ->"
                    f" {edge.to_step_id}) is not derivable from any step"
                    " policy; the linear subset only draws edges its"
                    " policies produce",
                    step_id=edge.from_step_id or None,
                    edge_id=edge.id,
                    detail={
                        "fromStepId": edge.from_step_id,
                        "toStepId": edge.to_step_id,
                    },
                )
            )
            continue

        claimed_on = next(iter(on_claims)) if len(on_claims) == 1 else None
        claimed_via = next(iter(via_claims)) if len(via_claims) == 1 else None
        claimed_kind = (
            next(iter(kind_claims)) if len(kind_claims) == 1 else None
        )
        chosen = None
        if claims_valid:
            for derived in candidates:
                if claimed_on is not None and derived["on"] != claimed_on:
                    continue
                if claimed_via is not None and derived["via"] != claimed_via:
                    continue
                if claimed_kind is not None and derived["kind"] != claimed_kind:
                    continue
                chosen = derived
                break
            if chosen is None:
                issues.append(
                    _issue(
                        "linear.edge_meta_mismatch",
                        f"edge {edge.id!r} annotations (on={claimed_on!r},"
                        f" via={claimed_via!r}, kind={claimed_kind!r})"
                        " contradict the edge derived from the step"
                        " policies",
                        edge_id=edge.id,
                        detail={
                            "derived": {
                                "on": candidates[0]["on"],
                                "via": candidates[0]["via"],
                                "kind": candidates[0]["kind"],
                            }
                        },
                    )
                )
        # Consume a candidate either way so the structural multiset check
        # does not double-report the same edge as missing.
        (chosen or candidates[0])["matched"] = True

    for derived in expected:
        if derived["matched"] or derived["from"] in unverifiable_steps:
            continue
        issues.append(
            _issue(
                "linear.edge_missing",
                f"step {derived['from']!r} policy requires the"
                f" {derived['on']} edge {derived['from']} ->"
                f" {derived['to']}; the graph does not draw it",
                step_id=derived["from"],
                detail={
                    "fromStepId": derived["from"],
                    "toStepId": derived["to"],
                    "on": derived["on"],
                },
            )
        )

    if issues:
        raise UnsupportedTopologyError(_summarize(issues), issues=issues)

    # --- fold (spec 6.1 rule 6): rebuild the linear dicts and let
    # normalize_workflow be the single validator of the step vocabulary. ---
    folded: list[dict[str, Any]] = []
    for step in flow.steps:
        entry: dict[str, Any] = {
            "id": step.id,
            "type": step.step_type,
            "name": step.name,
            "description": step.description,
            "on_failure": deepcopy(step.on_failure),
            "on_success": deepcopy(step.on_success),
        }
        for key, value in step.config.items():
            entry[key] = deepcopy(value)
        folded.append(entry)
    try:
        return normalize_workflow(folded)
    except WorkflowNormalizationError as exc:
        issue = _issue(
            "linear.normalize_failed",
            f"folded workflow failed normalize_workflow: {exc}",
            detail={"error": str(exc)},
        )
        raise UnsupportedTopologyError(
            _summarize([issue]), issues=[issue]
        ) from exc


def _missing_goto_targets(
    policy: dict[str, Any], step_ids: set[str]
) -> list[str]:
    """Every ``goto_step`` target (nested ``retry.then`` included) not in
    ``step_ids`` — same rule as kernel ``validate._missing_goto_targets``."""

    missing: list[str] = []
    policy_type = policy.get("type")
    if policy_type == "goto_step":
        target = policy.get("target_step_id")
        if target not in step_ids:
            missing.append(str(target))
    elif policy_type == "retry":
        then_policy = policy.get("then")
        if isinstance(then_policy, dict):
            missing.extend(_missing_goto_targets(then_policy, step_ids))
    return missing


def _summarize(issues: list[FlowIssue]) -> str:
    """Human-readable one-liner naming the offending nodes/edges (spec 6.2)."""

    locations = "; ".join(
        f"{issue.code}[{issue.edge_id or issue.step_id or 'flow'}]"
        for issue in issues
    )
    return (
        f"graph is not foldable to a linear Code Bridge workflow"
        f" ({len(issues)} issue(s)): {locations}"
    )
