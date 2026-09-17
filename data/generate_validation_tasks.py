"""Generate an independent held-out counterfactual validation set.

The scenarios and wording are intentionally different from ``tasks.json``.
This set is for testing candidate hypotheses discovered on the development
run; it must not be used to invent or filter teacher training examples.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from data.generate_tasks import STATUS_PAIRS, build_task
except ModuleNotFoundError:  # Support ``python data/generate_validation_tasks.py``.
    from generate_tasks import STATUS_PAIRS, build_task


VALIDATION_SCENARIOS = [
    ("2001", "v001", "无线耳塞", 299, "请问订单2001还能撤销吗？"),
    ("2002", "v002", "智能门锁", 1299, "我不想要订单2002了，帮我取消。"),
    ("2003", "v003", "空气净化器", 1599, "请协助处理订单2003的取消申请。"),
    ("2004", "v004", "投影仪", 2699, "订单2004可以帮我终止吗？"),
    ("2005", "v005", "咖啡机", 899, "麻烦确认一下订单2005是否能取消。"),
    ("2006", "v006", "运动手环", 199, "请替我撤销一下2006号订单。"),
    ("2007", "v007", "路由器", 399, "我想取消购买的订单2007。"),
    ("2008", "v008", "电饭煲", 499, "能帮我处理订单2008的取消吗？"),
    ("2009", "v009", "游戏耳机", 799, "请问2009号订单还能取消不？"),
    ("2010", "v010", "扫地机配件", 599, "帮我把订单2010撤掉。"),
]


def build_validation_tasks() -> list[dict]:
    tasks: list[dict] = []
    pair_number = 1
    for left_status, right_status in STATUS_PAIRS:
        for scenario in VALIDATION_SCENARIOS:
            tasks.append(build_task(pair_number, "a", *scenario, left_status, "val_pair"))
            tasks.append(build_task(pair_number, "b", *scenario, right_status, "val_pair"))
            pair_number += 1
    return tasks


if __name__ == "__main__":
    output_path = Path(__file__).with_name("validation_tasks.json")
    tasks = build_validation_tasks()
    output_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} validation tasks to {output_path}")
