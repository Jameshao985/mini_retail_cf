from runners.propose_hypotheses import build_hypotheses


def test_build_hypotheses_preserves_cluster_evidence():
    result = build_hypotheses(
        {
            "total_episodes": 10,
            "total_failures": 3,
            "clusters": [
                {
                    "failure_labels": ["false_success_claim"],
                    "count": 3,
                    "by_intervention_value": {"shipped": 3},
                    "violation_types": {"other": 3},
                    "action_signatures": {"get_order->respond": 3},
                    "examples": [{"task_id": "pair_001_b"}],
                }
            ],
        }
    )
    assert len(result["hypotheses"]) == 1
    hypothesis = result["hypotheses"][0]
    assert hypothesis["status"] == "candidate"
    assert hypothesis["capability"] == "terminal_state_grounding"
    assert hypothesis["evidence_count"] == 3
    assert hypothesis["representative_examples"][0]["task_id"] == "pair_001_b"
