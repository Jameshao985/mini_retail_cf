"""Shared episode execution and pair-level analysis."""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from config import DATA_DIR, MAX_STEPS, OUTPUT_DIR, SEED
from env.environment import MiniRetailEnvironment
from env.verifier import validate_pair_purity, verify_episode
from utils.trajectory_logger import write_json, write_jsonl


def load_tasks(path: Path | None = None) -> list[dict[str, Any]]:
    task_path = path or DATA_DIR / "tasks.json"
    with task_path.open("r", encoding="utf-8") as handle:
        tasks = json.load(handle)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[task["pair_id"]].append(task)
    for pair_id, pair in grouped.items():
        if len(pair) != 2:
            raise ValueError(f"{pair_id} must contain exactly two tasks")
        validate_pair_purity(*sorted(pair, key=lambda item: item["task_id"]))
    return tasks


def execute_tasks(
    agent: BaseAgent,
    tasks: list[dict[str, Any]],
    max_steps: int = MAX_STEPS,
    progress: bool = False,
) -> list[dict[str, Any]]:
    environment = MiniRetailEnvironment()
    experiment_id = f"{agent.agent_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    episodes: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        rollout = agent.run_episode(task, environment, max_steps=max_steps)
        verifier = verify_episode(
            task,
            rollout["initial_state"],
            rollout["final_state"],
            rollout["steps"],
            rollout["termination_reason"],
        )
        metadata = agent.metadata()
        episode = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task["task_id"],
            "pair_id": task["pair_id"],
            "capability": task["capability"],
            "model_name": metadata.get("model_name", metadata.get("agent_name")),
            "model_path": metadata.get("model_path"),
            "seed": metadata.get("seed", SEED),
            "metadata": metadata,
            "intervention_variable": task["intervention_variable"],
            "intervention_value": task["intervention_value"],
            "controlled_variables": task["controlled_variables"],
            "initial_state": rollout["initial_state"],
            "user_request": task["user_request"],
            "rule_visibility": metadata.get("rule_visibility", "programmatic"),
            "steps": rollout["steps"],
            "final_state": rollout["final_state"],
            "tool_calls": verifier["tool_calls"],
            "tool_call_attempt": verifier["tool_call_attempt"],
            "tool_execution_success": verifier["tool_execution_success"],
            "attempted_forbidden_action": verifier["attempted_forbidden_action"],
            "required_action_satisfied": verifier["required_action_satisfied"],
            "violation_types": verifier["violation_types"],
            "task_success": verifier["task_success"],
            "termination_reason": rollout["termination_reason"],
            "verifier_result": verifier,
            "failure_subtypes": verifier["failure_subtypes"],
        }
        episodes.append(episode)
        if progress:
            print(
                f"[{index}/{len(tasks)}] {task['task_id']}: "
                f"{'PASS' if episode['task_success'] else 'FAIL'}",
                flush=True,
            )
    return episodes


def analyze_pairs(episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["pair_id"]].append(episode)
    patterns: Counter[str] = Counter()
    pair_results: list[dict[str, Any]] = []
    for pair_id, pair in sorted(grouped.items()):
        ordered = sorted(pair, key=lambda item: item["task_id"])
        statuses = ["PASS" if item["task_success"] else "FAIL" for item in ordered]
        pattern = "/".join(statuses)
        patterns[pattern] += 1
        pair_results.append(
            {
                "pair_id": pair_id,
                "tasks": [
                    {
                        "task_id": item["task_id"],
                        "intervention_value": item["intervention_value"],
                        "success": item["task_success"],
                        "violation_types": item["violation_types"],
                        "failure_subtypes": item.get("failure_subtypes", []),
                    }
                    for item in ordered
                ],
                "pair_pattern": pattern,
                "pair_behavior": "success_flip"
                if len(statuses) == 2 and statuses[0] != statuses[1]
                else ("both_pass" if statuses == ["PASS", "PASS"] else "both_fail"),
            }
        )

    all_steps = [step for ep in episodes for step in ep["steps"]]
    parse_errors = sum(bool(step["parse_error"]) for step in all_steps)
    valid_actions = sum(step.get("parsed_action") is not None for step in all_steps)
    failures = [ep for ep in episodes if not ep["task_success"] or ep["violation_types"]]
    failure_counts = Counter(v for ep in failures for v in ep["violation_types"])
    failure_subtype_counts = Counter(
        subtype for ep in failures for subtype in ep.get("failure_subtypes", [])
    )
    status_metrics: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        status = episode["intervention_value"]
        bucket = status_metrics.setdefault(
            status,
            {
                "total_tasks": 0,
                "success": 0,
                "task_success_rate": 0.0,
                "forbidden_action_attempts": 0,
            },
        )
        bucket["total_tasks"] += 1
        bucket["success"] += int(bool(episode["task_success"]))
        bucket["forbidden_action_attempts"] += int(bool(episode["attempted_forbidden_action"]))
    for bucket in status_metrics.values():
        bucket["task_success_rate"] = bucket["success"] / bucket["total_tasks"]
    total = len(episodes)
    total_pairs = len(pair_results)
    complete_pairs = sum(item["pair_pattern"] == "PASS/PASS" for item in pair_results)
    summary = {
        "total_tasks": total,
        "success": sum(ep["task_success"] for ep in episodes),
        "failure": sum(not ep["task_success"] for ep in episodes),
        "task_success_rate": sum(ep["task_success"] for ep in episodes) / total if total else 0.0,
        "parse_errors": parse_errors,
        "parse_error_rate": parse_errors / len(all_steps) if all_steps else 0.0,
        "valid_json_rate": (len(all_steps) - parse_errors) / len(all_steps) if all_steps else 0.0,
        "valid_action_rate": valid_actions / len(all_steps) if all_steps else 0.0,
        "forbidden_action_attempts": sum(ep["attempted_forbidden_action"] for ep in episodes),
        "forbidden_action_attempt_rate": (
            sum(ep["attempted_forbidden_action"] for ep in episodes) / total if total else 0.0
        ),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "failure_subtype_counts": dict(sorted(failure_subtype_counts.items())),
        "status_metrics": status_metrics,
        "total_pairs": total_pairs,
        "pair_complete_success_rate": complete_pairs / total_pairs if total_pairs else 0.0,
        "pair_patterns": {
            "PASS/PASS": patterns["PASS/PASS"],
            "PASS/FAIL": patterns["PASS/FAIL"],
            "FAIL/PASS": patterns["FAIL/PASS"],
            "FAIL/FAIL": patterns["FAIL/FAIL"],
        },
        "interpretation_boundary": "Observed counterfactual behavior only; not causal capability attribution.",
    }
    return pair_results, summary


def persist_results(label: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    pair_results, summary = analyze_pairs(episodes)
    failures = [ep for ep in episodes if not ep["task_success"] or ep["violation_types"]]

    # Every run gets an immutable-by-convention artifact directory keyed by the
    # generated experiment ID. Legacy label-based files below are retained for
    # compatibility with existing scripts and reports.
    experiment_id = episodes[0]["experiment_id"] if episodes else f"{label}-empty"
    run_dir = OUTPUT_DIR / "runs" / experiment_id
    summary.update(
        {
            "label": label,
            "experiment_id": experiment_id,
            "run_artifact_dir": str(run_dir),
        }
    )
    metadata = episodes[0].get("metadata", {}) if episodes else {}
    write_json(run_dir / "manifest.json", {
        "experiment_id": experiment_id,
        "label": label,
        "task_count": len(episodes),
        "task_ids": [episode["task_id"] for episode in episodes],
        "model_metadata": metadata,
    })
    write_jsonl(run_dir / "trajectories.jsonl", episodes)
    write_jsonl(run_dir / "failures.jsonl", failures)
    write_json(run_dir / "pair_results.json", pair_results)
    write_json(run_dir / "summary.json", summary)

    write_jsonl(OUTPUT_DIR / "trajectories" / f"{label}_trajectories.jsonl", episodes)
    write_jsonl(OUTPUT_DIR / "failures" / f"{label}_failures.jsonl", failures)
    write_json(OUTPUT_DIR / "summaries" / f"{label}_pair_results.json", pair_results)
    write_json(OUTPUT_DIR / "summaries" / f"{label}_summary.json", summary)
    if label == "qwen_provided":
        write_jsonl(OUTPUT_DIR / "trajectories.jsonl", episodes)
        write_jsonl(OUTPUT_DIR / "failures.jsonl", failures)
        write_json(OUTPUT_DIR / "pair_results.json", pair_results)
        write_json(OUTPUT_DIR / "summary.json", summary)
    return summary
