"""Replay teacher candidates through the local environment and verifier.

The DeepSeek script checks syntax, but this second gate checks behavior.  Each
candidate is replayed from the *source episode's initial state*; no live Qwen
or network call is involved.  Only candidates that satisfy the same task
predicate as the source scenario are copied to ``--accepted-output``.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR
from env.environment import MiniRetailEnvironment
from env.tools import RetailTools
from env.verifier import verify_episode
from utils.action_parser import parse_action
from utils.trajectory_logger import write_jsonl


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _task_from_source(source: dict[str, Any]) -> dict[str, Any]:
    status = source["intervention_value"]
    order_id = source["controlled_variables"]["order_id"]
    should_cancel = status in {"pending", "processing"}
    return {
        "task_id": source.get("task_id", "teacher_source"),
        "pair_id": source.get("pair_id", "teacher_pair"),
        "capability": source.get("capability", "precondition_checking"),
        "intervention_variable": "order.status",
        "intervention_value": status,
        "controlled_variables": deepcopy(source["controlled_variables"]),
        "initial_state": deepcopy(source["initial_state"]),
        "user_request": source["user_request"],
        "expected_behavior": {"should_cancel": should_cancel},
        "required_actions": ["get_order", "cancel_order"]
        if should_cancel
        else ["get_order"],
        "forbidden_actions": [] if should_cancel else ["cancel_order"],
        "terminal_predicate": {"order.status": "cancelled" if should_cancel else status},
    }


def replay_teacher_record(
    record: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    """Replay assistant action messages and return a verifier result."""
    task = _task_from_source(source)
    environment = MiniRetailEnvironment(task["initial_state"])
    tools = RetailTools(environment)
    steps: list[dict[str, Any]] = []
    termination_reason = "max_steps_exceeded"
    for message in record.get("messages", []):
        if message.get("role") != "assistant":
            continue
        raw_output = message.get("content", "")
        parsed = parse_action(raw_output)
        before = environment.snapshot()
        step: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "raw_model_output": raw_output,
            "parsed_action": parsed.action,
            "parse_error": parsed.parse_error,
            "parse_error_message": parsed.error,
            "tool_name": None,
            "tool_arguments": None,
            "tool_result": None,
            "environment_state_before": before,
            "environment_state_after": before,
        }
        if parsed.parse_error or parsed.action is None:
            steps.append(step)
            break
        action = parsed.action["action"]
        if action == "respond":
            termination_reason = "agent_responded"
            steps.append(step)
            break
        arguments = parsed.action.get("arguments", {})
        step["tool_name"] = action
        step["tool_arguments"] = arguments
        step["tool_result"] = tools.execute(action, arguments)
        step["environment_state_after"] = environment.snapshot()
        steps.append(step)

    verifier = verify_episode(
        task,
        task["initial_state"],
        environment.get_state(),
        steps,
        termination_reason,
    )
    return {
        "example_id": record.get("example_id"),
        "hypothesis_id": record.get("hypothesis_id"),
        "source_task_id": record.get("source_task_id"),
        "action_sequence": [
            (step.get("parsed_action") or {}).get("action", "parse_error") for step in steps
        ],
        "task_success": verifier["task_success"],
        "violation_types": verifier["violation_types"],
        "failure_subtypes": verifier["failure_subtypes"],
        "verifier_result": verifier,
    }


def validate_candidates(
    records: list[dict[str, Any]],
    source_episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {episode.get("task_id"): episode for episode in source_episodes}
    accepted: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for record in records:
        source_id = record.get("source_task_id")
        source = by_id.get(source_id)
        if source is None:
            results.append(
                {
                    "example_id": record.get("example_id"),
                    "source_task_id": source_id,
                    "task_success": False,
                    "violation_types": ["missing_source_trajectory"],
                    "failure_subtypes": [],
                }
            )
            continue
        result = replay_teacher_record(record, source)
        results.append(result)
        if result["task_success"]:
            accepted_record = deepcopy(record)
            accepted_record["status"] = "validated_teacher_candidate"
            accepted_record["needs_student_verification"] = False
            accepted_record["verified_by"] = "mini_retail_verifier"
            accepted.append(accepted_record)
    summary = {
        "total_candidates": len(records),
        "accepted": len(accepted),
        "rejected": len(records) - len(accepted),
        "acceptance_rate": len(accepted) / len(records) if records else 0.0,
        "results": results,
        "interpretation_boundary": (
            "This is a programmatic replay gate for the mini environment; it does not prove "
            "generalization to unseen tasks or causal capability attribution."
        ),
    }
    return accepted, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument(
        "--accepted-output",
        type=Path,
        default=OUTPUT_DIR / "teacher" / "accepted_candidates.jsonl",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=OUTPUT_DIR / "teacher" / "validation_report.json",
    )
    args = parser.parse_args()
    candidates = load_jsonl(args.candidates)
    sources = load_jsonl(args.trajectories)
    accepted, summary = validate_candidates(candidates, sources)
    write_jsonl(args.accepted_output, accepted)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: summary[key] for key in ("total_candidates", "accepted", "rejected", "acceptance_rate")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
