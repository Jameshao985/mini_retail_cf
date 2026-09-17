import json

from agents.base_agent import AgentDecision, BaseAgent
from env.environment import MiniRetailEnvironment
from utils.action_parser import ParseResult, parse_action


def test_parser_accepts_plain_and_fenced_json():
    plain = parse_action('{"action":"get_order","arguments":{"order_id":"1001"}}')
    fenced = parse_action(
        '前文 ```json\n{"action":"respond","message":"完成"}\n``` 后文'
    )
    assert plain.parse_error is False
    assert plain.action["action"] == "get_order"
    assert fenced.parse_error is False
    assert fenced.action["action"] == "respond"


def test_parser_preserves_wrong_order_id_without_correcting_it():
    result = parse_action('{"action":"cancel_order","arguments":{"order_id":"9999"}}')
    assert result.parse_error is False
    assert result.action["arguments"]["order_id"] == "9999"


def test_malformed_output_is_an_observable_failure_not_an_exception():
    assert parse_action("not json").parse_error is True
    assert parse_action('{"action":"made_up"}').parse_error is True


class MalformedAgent(BaseAgent):
    def decide(self, task, history, step_id):
        return AgentDecision("oops", parse_action("oops"), [])


class EndlessGetAgent(BaseAgent):
    def decide(self, task, history, step_id):
        order_id = task["controlled_variables"]["order_id"]
        action = {"action": "get_order", "arguments": {"order_id": order_id}}
        return AgentDecision(json.dumps(action), ParseResult(action, False), [])


def tiny_task():
    return {
        "controlled_variables": {"order_id": "1001"},
        "initial_state": {
            "orders": {"1001": {"user_id": "u", "item": "x", "price": 1, "status": "pending"}}
        },
        "user_request": "test",
    }


def test_malformed_agent_output_does_not_crash_episode():
    rollout = MalformedAgent().run_episode(tiny_task(), MiniRetailEnvironment())
    assert rollout["termination_reason"] == "parse_error"
    assert rollout["steps"][0]["parse_error"] is True


def test_max_steps_terminates_normally():
    rollout = EndlessGetAgent().run_episode(tiny_task(), MiniRetailEnvironment(), max_steps=2)
    assert rollout["termination_reason"] == "max_steps_exceeded"
    assert len(rollout["steps"]) == 2

