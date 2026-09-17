"""Generate a clean replication set with imperative cancellation requests.

Unlike the broader validation set, every request here unambiguously asks the
agent to perform cancellation. This isolates status sensitivity from the
question-vs-command wording difference observed in the first held-out run.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from data.generate_tasks import STATUS_PAIRS, build_task
except ModuleNotFoundError:  # Support direct script execution.
    from generate_tasks import STATUS_PAIRS, build_task


REPLICATION_SCENARIOS = [
    ("3001", "r001", "无线键盘", 299, "请直接取消订单3001。"),
    ("3002", "r002", "便携显示器", 1199, "请帮我取消订单3002。"),
    ("3003", "r003", "扫地机器人", 2399, "麻烦取消我的订单3003。"),
    ("3004", "r004", "手机支架", 89, "请替我把订单3004取消。"),
    ("3005", "r005", "平板保护套", 129, "帮我取消掉订单3005。"),
    ("3006", "r006", "蓝牙耳机", 599, "我需要取消订单3006。"),
    ("3007", "r007", "智能台灯", 239, "请协助取消订单3007。"),
    ("3008", "r008", "空气炸锅", 699, "帮忙取消一下订单3008。"),
    ("3009", "r009", "机械鼠标", 399, "请把我的订单3009取消。"),
    ("3010", "r010", "移动硬盘", 799, "麻烦帮我处理订单3010取消。"),
]


def build_replication_tasks() -> list[dict]:
    tasks: list[dict] = []
    pair_number = 1
    for left_status, right_status in STATUS_PAIRS:
        for scenario in REPLICATION_SCENARIOS:
            tasks.append(build_task(pair_number, "a", *scenario, left_status, "rep_pair"))
            tasks.append(build_task(pair_number, "b", *scenario, right_status, "rep_pair"))
            pair_number += 1
    return tasks


if __name__ == "__main__":
    output_path = Path(__file__).with_name("replication_tasks.json")
    tasks = build_replication_tasks()
    output_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} replication tasks to {output_path}")
