"""Offline Qwen sanity check and counterfactual batch runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.qwen_agent import QwenAgent
from config import OUTPUT_DIR
from env.environment import MiniRetailEnvironment
from utils.model_path import ModelPathError, discover_model_path
from utils.trajectory_logger import write_json

from .common import execute_tasks, load_tasks, persist_results


def _sanity_task(index: int) -> dict[str, Any]:
    order_id = str(9000 + index)
    return {
        "task_id": f"sanity_{index}",
        "pair_id": f"sanity_{index}",
        "capability": "tool_schema_following",
        "user_request": f"请查询订单{order_id}的当前信息，然后告诉我查询结果。",
        "initial_state": {
            "orders": {
                order_id: {
                    "user_id": "sanity_user",
                    "item": "测试商品",
                    "price": 100,
                    "status": "pending",
                }
            }
        },
        "controlled_variables": {"order_id": order_id},
    }


def run_sanity_check(agent: QwenAgent, trials: int = 3) -> dict[str, Any]:
    environment = MiniRetailEnvironment()
    rollouts: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index in range(1, trials + 1):
        task = _sanity_task(index)
        rollouts.append((task, agent.run_episode(task, environment, max_steps=3)))
    steps = [step for _, rollout in rollouts for step in rollout["steps"]]
    tool_actions = [
        (task, step)
        for task, rollout in rollouts
        for step in rollout["steps"]
        if step.get("tool_name")
    ]
    valid = [step for step in steps if not step["parse_error"]]
    correct_order_ids = sum(
        step["tool_arguments"].get("order_id") == task["controlled_variables"]["order_id"]
        for task, step in tool_actions
    )
    basic_successes = 0
    for task, rollout in rollouts:
        order_id = task["controlled_variables"]["order_id"]
        got_order = any(
            step.get("tool_name") == "get_order"
            and (step.get("tool_arguments") or {}).get("order_id") == order_id
            and (step.get("tool_result") or {}).get("success")
            for step in rollout["steps"]
        )
        responded = rollout["termination_reason"] == "agent_responded"
        basic_successes += bool(got_order and responded)
    summary = {
        "model_loaded": "YES",
        "model_metadata": agent.metadata(),
        "trials": trials,
        "total_model_steps": len(steps),
        "valid_json_rate": len(valid) / len(steps) if steps else 0.0,
        "valid_action_rate": len(valid) / len(steps) if steps else 0.0,
        "correct_order_id_rate": correct_order_ids / len(tool_actions) if tool_actions else 0.0,
        "basic_get_order_success_rate": basic_successes / trials,
        "parse_error_count": sum(step["parse_error"] for step in steps),
        "competing_explanation_warning": (
            "If schema/action/order-id rates are low, later task failures cannot be attributed only to "
            "precondition_checking."
        ),
        "rollouts": [rollout for _, rollout in rollouts],
    }
    write_json(OUTPUT_DIR / "summaries" / "qwen_sanity_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--rule-visibility", choices=["provided", "hidden"], default="provided")
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tasks-path", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()
    try:
        model_path = discover_model_path(args.model_path)
    except ModelPathError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Offline model snapshot: {model_path}")
    agent = QwenAgent(model_path, rule_visibility=args.rule_visibility)
    if args.sanity_check:
        print(json.dumps(run_sanity_check(agent), ensure_ascii=False, indent=2))
        return
    tasks = load_tasks(args.tasks_path)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    episodes = execute_tasks(agent, tasks, progress=True)
    label = args.label or f"qwen_{args.rule_visibility}"
    if args.limit is not None:
        label += f"_limit{args.limit}"
    print(json.dumps(persist_results(label, episodes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
