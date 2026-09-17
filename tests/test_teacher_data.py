from runners.generate_teacher_data import (
    build_teacher_request,
    normalize_teacher_examples,
    validate_teacher_payload,
)


def hypothesis():
    return {
        "hypothesis_id": "H001",
        "failure_label": "missing_explicit_refusal",
        "capability": "policy_constrained_refusal",
        "title": "禁止条件下的明确拒绝",
        "hypothesis": "模型可能不能在禁止状态下明确拒绝。",
        "expected_counterfactual": "shipped 状态不调用 cancel_order，并明确告知不能取消。",
        "alternative_explanations": [],
    }


def valid_payload():
    return {
        "examples": [
            {
                "source_task_id": "rep_pair_001_b",
                "target_behavior": "先查询状态，再明确拒绝。",
                "messages": [
                    {"role": "system", "content": "规则"},
                    {"role": "user", "content": "请取消订单3001"},
                    {
                        "role": "assistant",
                        "content": '{"action":"get_order","arguments":{"order_id":"3001"}}',
                    },
                    {"role": "user", "content": "工具返回：{\"status\":\"shipped\"}"},
                    {
                        "role": "assistant",
                        "content": '{"action":"respond","message":"订单已发货，不能取消。"}',
                    },
                ],
            }
        ]
    }


def test_teacher_payload_requires_valid_action_messages():
    assert validate_teacher_payload(valid_payload()) == []
    invalid = {
        "examples": [
            {
                "messages": [
                    {"role": "user", "content": "请取消"},
                    {"role": "assistant", "content": "我不能帮你"},
                ]
            }
        ]
    }
    errors = validate_teacher_payload(invalid)
    assert any("valid action JSON" in error for error in errors)


def test_teacher_payload_rejects_cancel_before_get():
    payload = valid_payload()
    payload["examples"][0]["messages"][2]["content"] = (
        '{"action":"cancel_order","arguments":{"order_id":"3001"}}'
    )
    errors = validate_teacher_payload(payload)
    assert any("before get_order" in error for error in errors)


def test_request_is_local_and_normalization_marks_review_gate():
    request = build_teacher_request(hypothesis(), [], examples_per_request=2)
    assert request[0]["role"] == "system"
    assert "examples_per_request" in request[1]["content"]
    records = normalize_teacher_examples(valid_payload(), hypothesis(), "test-model", ["rep_pair_001_b"])
    assert records[0]["status"] == "teacher_candidate"
    assert records[0]["needs_student_verification"] is True
    assert records[0]["source_task_id"] == "rep_pair_001_b"
