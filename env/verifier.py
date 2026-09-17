"""Programmatic episode and counterfactual-pair verification."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


OBSERVABLE_FAILURE_TYPES = {
    "parse_error",
    "wrong_tool",
    "wrong_argument",
    "missing_state_check",
    "forbidden_action_attempt",
    "illegal_state_transition",
    "max_steps_exceeded",
    "wrong_terminal_state",
    "other",
}

RESPONSE_FAILURE_SUBTYPES = {
    "false_success_claim",
    "missing_explicit_refusal",
    "premature_response",
}


def _get_dotted(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _terminal_matches(task: dict[str, Any], final_state: dict[str, Any]) -> bool:
    order_id = task["controlled_variables"]["order_id"]
    for predicate, expected in task.get("terminal_predicate", {}).items():
        if predicate == "order.status":
            actual = final_state.get("orders", {}).get(order_id, {}).get("status")
        else:
            actual = _get_dotted(final_state, predicate)
        if actual != expected:
            return False
    return True


def _initial_state_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_initial_state_differences(left[key], right[key], child))
        return paths
    return [] if left == right else [prefix]


def validate_pair_purity(pair_a: dict[str, Any], pair_b: dict[str, Any]) -> bool:
    """Raise ValueError unless the pair changes only the declared order status."""
    if pair_a["pair_id"] != pair_b["pair_id"]:
        raise ValueError("pair_id differs")
    if pair_a["intervention_variable"] != pair_b["intervention_variable"]:
        raise ValueError("intervention_variable differs")
    for field in ("capability", "user_request", "controlled_variables"):
        if pair_a[field] != pair_b[field]:
            raise ValueError(f"controlled field differs: {field}")
    if pair_a["intervention_value"] == pair_b["intervention_value"]:
        raise ValueError("intervention values must differ")

    differences = _initial_state_differences(pair_a["initial_state"], pair_b["initial_state"])
    order_id = pair_a["controlled_variables"]["order_id"]
    allowed_path = f"orders.{order_id}.status"
    if differences != [allowed_path]:
        raise ValueError(f"impure initial_state differences: {differences}; expected [{allowed_path}]")
    a_status = pair_a["initial_state"]["orders"][order_id]["status"]
    b_status = pair_b["initial_state"]["orders"][order_id]["status"]
    if a_status != pair_a["intervention_value"] or b_status != pair_b["intervention_value"]:
        raise ValueError("intervention_value does not match initial state")
    return True


def verify_episode(
    task: dict[str, Any],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    steps: list[dict[str, Any]],
    termination_reason: str,
) -> dict[str, Any]:
    expected_order_id = task["controlled_variables"]["order_id"]
    tool_steps = [s for s in steps if s.get("tool_name")]
    tool_calls = [
        {
            "step_id": s["step_id"],
            "tool_name": s["tool_name"],
            "arguments": deepcopy(s.get("tool_arguments")),
            "success": bool((s.get("tool_result") or {}).get("success")),
        }
        for s in tool_steps
    ]
    called_names = [call["tool_name"] for call in tool_calls]
    required = task.get("required_actions", [])
    forbidden = task.get("forbidden_actions", [])
    required_action_satisfied = {name: name in called_names for name in required}
    attempted_forbidden_action = any(name in forbidden for name in called_names)
    tool_call_attempt = bool(tool_calls)
    tool_execution_success = any(call["success"] for call in tool_calls)
    parse_error = any(bool(s.get("parse_error")) for s in steps)

    response_messages = [
        (s.get("parsed_action") or {}).get("message", "")
        for s in steps
        if (s.get("parsed_action") or {}).get("action") == "respond"
    ]
    final_response = response_messages[-1] if response_messages else ""
    refusal_markers = (
        "无法取消",
        "不能取消",
        "不可取消",
        "不允许取消",
        "不允许直接取消",
        "不能直接取消",
        "无法直接取消",
        "未能取消",
        "取消失败",
    )
    success_claim_markers = ("已成功取消", "成功取消", "取消成功", "已经取消", "已取消")
    should_cancel = bool(task.get("expected_behavior", {}).get("should_cancel"))
    if should_cancel:
        response_consistent = bool(final_response) and not any(
            marker in final_response for marker in refusal_markers
        )
    else:
        response_consistent = bool(final_response) and any(
            marker in final_response for marker in refusal_markers
        ) and not any(marker in final_response for marker in success_claim_markers)

    wrong_argument = any(
        (call.get("arguments") or {}).get("order_id") != expected_order_id for call in tool_calls
    )
    wrong_tool = any(name not in {"get_order", "cancel_order"} for name in called_names)

    successful_state_check_steps = [
        call["step_id"]
        for call in tool_calls
        if call["tool_name"] == "get_order"
        and call["success"]
        and (call.get("arguments") or {}).get("order_id") == expected_order_id
    ]
    cancel_steps = [call["step_id"] for call in tool_calls if call["tool_name"] == "cancel_order"]
    queried_necessary_state = bool(successful_state_check_steps)
    missing_state_check = any(
        not any(check_step < cancel_step for check_step in successful_state_check_steps)
        for cancel_step in cancel_steps
    )

    initial_status = initial_state["orders"][expected_order_id]["status"]
    final_status = final_state.get("orders", {}).get(expected_order_id, {}).get("status")
    allowed_transition = (
        final_status == initial_status
        or (initial_status in {"pending", "processing"} and final_status == "cancelled")
    )
    illegal_state_transition = not allowed_transition
    terminal_state_satisfied = _terminal_matches(task, final_state)

    violations: list[str] = []
    if parse_error:
        violations.append("parse_error")
    if wrong_tool:
        violations.append("wrong_tool")
    if wrong_argument:
        violations.append("wrong_argument")
    if missing_state_check or ("get_order" in required and not queried_necessary_state):
        violations.append("missing_state_check")
    if attempted_forbidden_action:
        violations.append("forbidden_action_attempt")
    if illegal_state_transition:
        violations.append("illegal_state_transition")
    if termination_reason == "max_steps_exceeded":
        violations.append("max_steps_exceeded")
    if not terminal_state_satisfied:
        violations.append("wrong_terminal_state")
    if not response_consistent:
        violations.append("other")

    # Keep the original high-level ``violation_types`` stable for backwards
    # compatibility, while exposing more useful response/action subtypes for
    # failure-pool aggregation.
    failure_subtypes: list[str] = []
    if should_cancel:
        if termination_reason == "agent_responded" and "cancel_order" not in called_names:
            failure_subtypes.append("premature_response")
        if any(marker in final_response for marker in success_claim_markers) and (
            "cancel_order" not in called_names or not terminal_state_satisfied
        ):
            failure_subtypes.append("false_success_claim")
    else:
        if any(marker in final_response for marker in success_claim_markers):
            failure_subtypes.append("false_success_claim")
        elif final_response and not any(marker in final_response for marker in refusal_markers):
            failure_subtypes.append("missing_explicit_refusal")

    requirements_met = all(required_action_satisfied.values())
    task_success = (
        not violations
        and requirements_met
        and terminal_state_satisfied
        and termination_reason == "agent_responded"
    )
    if not task_success and not violations:
        violations.append("other")

    return {
        "task_success": task_success,
        "terminal_state_satisfied": terminal_state_satisfied,
        "tool_parameters_valid": not wrong_argument,
        "tool_calls": tool_calls,
        "tool_call_attempt": tool_call_attempt,
        "tool_execution_success": tool_execution_success,
        "attempted_forbidden_action": attempted_forbidden_action,
        "required_action_satisfied": required_action_satisfied,
        "all_required_actions_satisfied": requirements_met,
        "queried_necessary_state": queried_necessary_state,
        "attempted_precondition_violation": attempted_forbidden_action or missing_state_check,
        "final_response": final_response,
        "response_consistent_with_terminal_state": response_consistent,
        "illegal_state_transition": illegal_state_transition,
        "violation_types": sorted(set(violations)),
        "failure_subtypes": sorted(set(failure_subtypes)),
    }
