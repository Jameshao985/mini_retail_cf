from runners.analyze_failures import build_failure_report


def episode(task_id, status, success, labels, actions):
    return {
        "task_id": task_id,
        "pair_id": task_id.rsplit("_", 1)[0],
        "intervention_value": status,
        "task_success": success,
        "failure_subtypes": labels,
        "violation_types": labels,
        "steps": [{"parsed_action": {"action": action}} for action in actions],
        "verifier_result": {"final_response": ""},
    }


def test_failure_report_clusters_and_keeps_examples():
    report = build_failure_report(
        [
            episode("p1_a", "pending", False, ["premature_response"], ["get_order", "respond"]),
            episode("p2_a", "pending", False, ["premature_response"], ["get_order", "respond"]),
            episode("p3_b", "shipped", True, [], ["get_order", "respond"]),
        ],
        max_examples_per_cluster=1,
    )
    assert report["total_episodes"] == 3
    assert report["total_failures"] == 2
    assert report["clusters"][0]["failure_labels"] == ["premature_response"]
    assert report["clusters"][0]["count"] == 2
    assert len(report["clusters"][0]["examples"]) == 1
