from runners.validate_teacher_data import replay_teacher_record, validate_candidates


def source(status="shipped"):
    return {
        "task_id": "rep_pair_001_b",
        "pair_id": "rep_pair_001",
        "capability": "precondition_checking",
        "intervention_value": status,
        "controlled_variables": {
            "user_id": "r001",
            "order_id": "3001",
            "item": "键盘",
            "price": 10,
            "request_text": "请取消订单3001",
            "tool_set": ["get_order", "cancel_order"],
            "task_goal": "cancel_order_if_allowed",
            "language_style": "concise_zh",
        },
        "initial_state": {
            "orders": {
                "3001": {"user_id": "r001", "item": "键盘", "price": 10, "status": status}
            }
        },
        "user_request": "请取消订单3001",
    }


def candidate(status="shipped"):
    refusal = "订单已发货，不能取消。" if status == "shipped" else "订单已取消，不能重复取消。"
    return {
        "example_id": "H001-T001",
        "hypothesis_id": "H001",
        "source_task_id": "rep_pair_001_b",
        "messages": [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": "请取消订单3001"},
            {
                "role": "assistant",
                "content": '{"action":"get_order","arguments":{"order_id":"3001"}}',
            },
            {"role": "user", "content": "工具返回：状态"},
            {"role": "assistant", "content": '{"action":"respond","message":"' + refusal + '"}'},
        ],
    }


def test_replay_accepts_safe_refusal():
    result = replay_teacher_record(candidate(), source())
    assert result["task_success"] is True
    assert result["action_sequence"] == ["get_order", "respond"]


def test_replay_rejects_forbidden_cancel():
    record = candidate()
    record["messages"][4]["content"] = (
        '{"action":"cancel_order","arguments":{"order_id":"3001"}}'
    )
    result = replay_teacher_record(record, source())
    assert result["task_success"] is False
    assert "forbidden_action_attempt" in result["violation_types"]


def test_validate_candidates_copies_only_passes():
    accepted, summary = validate_candidates([candidate()], [source()])
    assert len(accepted) == 1
    assert summary["accepted"] == 1
    assert accepted[0]["status"] == "validated_teacher_candidate"
    assert accepted[0]["needs_student_verification"] is False
