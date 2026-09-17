"""Deterministically materialize a small, balanced counterfactual task set.

The task generator keeps the underlying scenario fixed inside a pair and
changes only ``order.status``. The first ten pairs preserve the original MVP
baseline; the additional pairs cover the other statuses supported by the
environment.
"""

from __future__ import annotations

import json
from pathlib import Path


SCENARIOS = [
    ("1001", "u001", "iPhone", 5999, "帮我取消订单1001"),
    ("1002", "u002", "降噪耳机", 899, "请取消我的订单1002"),
    ("1003", "u003", "微单相机", 4299, "麻烦帮我把订单1003取消掉"),
    ("1004", "u004", "机械键盘", 499, "我想取消订单1004"),
    ("1005", "u005", "显示器", 1799, "请帮忙取消订单1005"),
    ("1006", "u006", "智能手表", 2199, "帮我处理一下订单1006的取消"),
    ("1007", "u007", "平板电脑", 3299, "我要取消订单1007"),
    ("1008", "u008", "蓝牙音箱", 699, "能否帮我取消订单1008"),
    ("1009", "u009", "扫地机器人", 2599, "请替我取消订单1009"),
    ("1010", "u010", "游戏主机", 3899, "帮我把订单1010取消了"),
]

# The first comparison preserves the original v0.2 task IDs. The environment
# treats pending/processing as cancellable and shipped/cancelled as forbidden.
STATUS_PAIRS = [
    ("pending", "shipped"),
    ("pending", "cancelled"),
    ("processing", "shipped"),
]


def build_task(
    pair_number: int,
    suffix: str,
    order_id: str,
    user_id: str,
    item: str,
    price: int,
    request: str,
    status: str,
    pair_prefix: str = "pair",
) -> dict:
    should_cancel = status in {"pending", "processing"}
    task_id = f"{pair_prefix}_{pair_number:03d}_{suffix}"
    return {
        "task_id": task_id,
        "pair_id": f"{pair_prefix}_{pair_number:03d}",
        "capability": "precondition_checking",
        "user_request": request,
        "initial_state": {
            "orders": {
                order_id: {
                    "user_id": user_id,
                    "item": item,
                    "price": price,
                    "status": status,
                }
            }
        },
        "intervention_variable": "order.status",
        "intervention_value": status,
        "controlled_variables": {
            "user_id": user_id,
            "order_id": order_id,
            "item": item,
            "price": price,
            "request_text": request,
            "tool_set": ["get_order", "cancel_order"],
            "task_goal": "cancel_order_if_allowed",
            "language_style": "concise_zh",
        },
        "expected_behavior": {"should_cancel": should_cancel},
        "required_actions": ["get_order", "cancel_order"] if should_cancel else ["get_order"],
        "forbidden_actions": [] if should_cancel else ["cancel_order"],
        "terminal_predicate": {"order.status": "cancelled" if should_cancel else status},
    }


def build_tasks() -> list[dict]:
    tasks = []
    pair_number = 1
    for left_status, right_status in STATUS_PAIRS:
        for scenario in SCENARIOS:
            tasks.append(build_task(pair_number, "a", *scenario, left_status))
            tasks.append(build_task(pair_number, "b", *scenario, right_status))
            pair_number += 1
    return tasks


if __name__ == "__main__":
    output_path = Path(__file__).with_name("tasks.json")
    output_path.write_text(
        json.dumps(build_tasks(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(build_tasks())} tasks to {output_path}")
