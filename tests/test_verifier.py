from copy import deepcopy

from agents import BadAgentAlwaysCancel, BadAgentChecksButIgnores, ScriptedGoodAgent
from env.verifier import verify_episode
from runners.common import execute_tasks, load_tasks


def get_task(task_id):
    return next(task for task in load_tasks() if task["task_id"] == task_id)


def test_verifier_recognizes_good_pending_and_shipped_paths():
    episodes = execute_tasks(ScriptedGoodAgent(), [get_task("pair_001_a"), get_task("pair_001_b")])
    assert [episode["task_success"] for episode in episodes] == [True, True]
    assert episodes[0]["final_state"]["orders"]["1001"]["status"] == "cancelled"
    assert episodes[1]["final_state"]["orders"]["1001"]["status"] == "shipped"


def test_always_cancel_distinguishes_attempt_execution_and_task_success():
    pending, shipped = execute_tasks(
        BadAgentAlwaysCancel(), [get_task("pair_001_a"), get_task("pair_001_b")]
    )
    assert pending["tool_call_attempt"] is True
    assert pending["tool_execution_success"] is True
    assert pending["task_success"] is False
    assert "missing_state_check" in pending["violation_types"]
    assert shipped["tool_call_attempt"] is True
    assert shipped["tool_execution_success"] is False
    assert shipped["task_success"] is False
    assert shipped["attempted_forbidden_action"] is True
    assert "forbidden_action_attempt" in shipped["violation_types"]


def test_checks_but_ignores_is_forbidden_without_missing_check():
    episode = execute_tasks(BadAgentChecksButIgnores(), [get_task("pair_001_b")])[0]
    assert episode["verifier_result"]["queried_necessary_state"] is True
    assert "missing_state_check" not in episode["violation_types"]
    assert "forbidden_action_attempt" in episode["violation_types"]


def test_verifier_rejects_response_that_claims_shipped_order_was_cancelled():
    task = get_task("pair_001_b")
    episode = execute_tasks(ScriptedGoodAgent(), [task])[0]
    steps = deepcopy(episode["steps"])
    steps[-1]["parsed_action"]["message"] = "订单1001已取消。"
    result = verify_episode(
        task,
        episode["initial_state"],
        episode["final_state"],
        steps,
        episode["termination_reason"],
    )
    assert result["response_consistent_with_terminal_state"] is False
    assert result["task_success"] is False
    assert "other" in result["violation_types"]
    assert "false_success_claim" in result["failure_subtypes"]


def test_verifier_distinguishes_missing_explicit_refusal():
    task = get_task("pair_001_b")
    episode = execute_tasks(ScriptedGoodAgent(), [task])[0]
    steps = deepcopy(episode["steps"])
    steps[-1]["parsed_action"]["message"] = "订单1001已发货。"
    result = verify_episode(
        task,
        episode["initial_state"],
        episode["final_state"],
        steps,
        episode["termination_reason"],
    )
    assert result["response_consistent_with_terminal_state"] is False
    assert result["failure_subtypes"] == ["missing_explicit_refusal"]
