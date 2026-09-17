"""Aggregate trajectory failures into compact, reviewable failure clusters.

This is the deterministic first stage of the later teacher-model diagnosis
pipeline. It never invents a capability label; it groups episodes using the
verifier's observable violation and subtype fields, then keeps a few example
task IDs and response snippets for human or strong-model review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from config import OUTPUT_DIR
from utils.trajectory_logger import write_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _action_signature(episode: dict[str, Any]) -> str:
    actions = [
        (step.get("parsed_action") or {}).get("action", "parse_error")
        for step in episode.get("steps", [])
    ]
    return "->".join(actions) if actions else "empty"


def _legacy_response_subtypes(episode: dict[str, Any]) -> list[str]:
    """Recover the new response tags when reading an older trajectory file."""
    verifier = episode.get("verifier_result") or {}
    response = verifier.get("final_response", "")
    called_names = [call.get("tool_name") for call in episode.get("tool_calls", [])]
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
    success_markers = ("已成功取消", "成功取消", "取消成功", "已经取消", "已取消")
    should_cancel = episode.get("intervention_value") in {"pending", "processing"}
    if should_cancel and episode.get("termination_reason") == "agent_responded":
        if "cancel_order" not in called_names:
            return ["premature_response"]
    if not should_cancel:
        if any(marker in response for marker in success_markers):
            return ["false_success_claim"]
        if response and not any(marker in response for marker in refusal_markers):
            return ["missing_explicit_refusal"]
    return []


def _failure_labels(episode: dict[str, Any]) -> tuple[str, ...]:
    labels = list(episode.get("failure_subtypes", []))
    if not labels:
        labels.extend(_legacy_response_subtypes(episode))
    if not labels:
        labels.extend(episode.get("violation_types", []))
    return tuple(sorted(set(labels))) or ("pass",)


def build_failure_report(
    episodes: Iterable[dict[str, Any]], max_examples_per_cluster: int = 5
) -> dict[str, Any]:
    episodes = list(episodes)
    failures = [episode for episode in episodes if not episode.get("task_success", False)]
    clusters: dict[tuple[str, ...], dict[str, Any]] = {}
    for episode in failures:
        labels = _failure_labels(episode)
        cluster = clusters.setdefault(
            labels,
            {
                "failure_labels": list(labels),
                "count": 0,
                "by_intervention_value": Counter(),
                "violation_types": Counter(),
                "action_signatures": Counter(),
                "examples": [],
            },
        )
        cluster["count"] += 1
        cluster["by_intervention_value"][episode.get("intervention_value", "unknown")] += 1
        cluster["violation_types"].update(episode.get("violation_types", []))
        cluster["action_signatures"][_action_signature(episode)] += 1
        if len(cluster["examples"]) < max_examples_per_cluster:
            cluster["examples"].append(
                {
                    "task_id": episode.get("task_id"),
                    "pair_id": episode.get("pair_id"),
                    "intervention_value": episode.get("intervention_value"),
                    "user_request": episode.get("user_request"),
                    "final_response": (episode.get("verifier_result") or {}).get(
                        "final_response", ""
                    ),
                    "action_signature": _action_signature(episode),
                }
            )

    serialized_clusters = []
    for cluster in sorted(clusters.values(), key=lambda item: (-item["count"], item["failure_labels"])):
        cluster["by_intervention_value"] = dict(sorted(cluster["by_intervention_value"].items()))
        cluster["violation_types"] = dict(sorted(cluster["violation_types"].items()))
        cluster["action_signatures"] = dict(
            sorted(cluster["action_signatures"].items(), key=lambda item: (-item[1], item[0]))
        )
        serialized_clusters.append(cluster)

    intervention_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    for episode in episodes:
        status = episode.get("intervention_value", "unknown")
        intervention_counts[status]["total"] += 1
        intervention_counts[status]["success"] += int(bool(episode.get("task_success")))

    return {
        "total_episodes": len(episodes),
        "total_failures": len(failures),
        "failure_rate": len(failures) / len(episodes) if episodes else 0.0,
        "intervention_metrics": dict(sorted(intervention_counts.items())),
        "clusters": serialized_clusters,
        "interpretation_boundary": (
            "Clusters summarize observable trajectory patterns; they are candidate evidence, "
            "not confirmed capability diagnoses."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_DIR / "trajectories.jsonl",
        help="JSONL trajectory file to aggregate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "analysis" / "failure_report.json",
        help="JSON report path",
    )
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()
    report = build_failure_report(load_jsonl(args.input), args.max_examples)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
