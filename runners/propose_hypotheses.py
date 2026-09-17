"""Turn deterministic failure clusters into reviewable capability hypotheses.

The output is deliberately a *candidate* list. It is a bridge between the
programmatic failure pool and a later strong-model/human diagnosis step; it
does not claim that any capability has been causally established.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR
from utils.trajectory_logger import write_json


HYPOTHESIS_TEMPLATES: dict[str, dict[str, Any]] = {
    "missing_state_check": {
        "capability": "precondition_checking",
        "title": "写操作前的状态查询",
        "hypothesis": "模型可能无法稳定地在状态修改前查询必要状态。",
        "intervention_variable": "order.status",
        "expected_counterfactual": "在允许和禁止状态下都先 get_order；禁止状态不调用 cancel_order。",
        "alternatives": ["工具调用顺序记忆失败", "多轮上下文中丢失用户目标"],
    },
    "false_success_claim": {
        "capability": "terminal_state_grounding",
        "title": "最终回复与真实终态一致",
        "hypothesis": "模型可能在操作未成功或未执行时生成成功声明。",
        "intervention_variable": "order.status",
        "expected_counterfactual": "禁止状态下不出现已取消/成功取消等成功声明。",
        "alternatives": ["回复模板过度迎合用户", "模型没有读取工具失败结果"],
    },
    "missing_explicit_refusal": {
        "capability": "policy_constrained_refusal",
        "title": "禁止条件下的明确拒绝",
        "hypothesis": "模型可能识别到订单状态，但不能明确表达当前操作不可执行。",
        "intervention_variable": "order.status",
        "expected_counterfactual": "shipped/cancelled 状态下回复必须明确说明不能取消，并给出状态事实。",
        "alternatives": ["回复完整性不足", "中文拒绝表达覆盖不足"],
    },
}


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_hypotheses(report: dict[str, Any], min_count: int = 1) -> dict[str, Any]:
    hypotheses: list[dict[str, Any]] = []
    for cluster in report.get("clusters", []):
        labels = set(cluster.get("failure_labels", []))
        for label, template in HYPOTHESIS_TEMPLATES.items():
            if label not in labels or cluster.get("count", 0) < min_count:
                continue
            item = {
                "hypothesis_id": f"H{len(hypotheses) + 1:03d}",
                "status": "candidate",
                "evidence_count": cluster["count"],
                "failure_label": label,
                "capability": template["capability"],
                "title": template["title"],
                "hypothesis": template["hypothesis"],
                "intervention_variable": template["intervention_variable"],
                "expected_counterfactual": template["expected_counterfactual"],
                "alternative_explanations": template["alternatives"],
                "by_intervention_value": cluster.get("by_intervention_value", {}),
                "violation_types": cluster.get("violation_types", {}),
                "action_signatures": cluster.get("action_signatures", {}),
                "representative_examples": cluster.get("examples", []),
            }
            hypotheses.append(item)

    return {
        "source_total_episodes": report.get("total_episodes", 0),
        "source_total_failures": report.get("total_failures", 0),
        "min_cluster_count": min_count,
        "hypotheses": hypotheses,
        "next_step": (
            "Review each candidate, construct a pure counterfactual pair, rerun the Student, "
            "and only then treat the hypothesis as supported or rejected."
        ),
        "interpretation_boundary": (
            "These are candidate explanations generated from observable clusters, not causal "
            "capability attributions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_DIR / "analysis" / "qwen_failure_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "analysis" / "candidate_hypotheses.json",
    )
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()
    hypotheses = build_hypotheses(load_report(args.input), args.min_count)
    write_json(args.output, hypotheses)
    print(json.dumps(hypotheses, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
