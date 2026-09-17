"""Reference agent used to validate the environment before model rollout."""

from __future__ import annotations

from typing import Any

from env.rules import can_cancel_order

from .base_agent import AgentDecision, BaseAgent, scripted_decision


class ScriptedGoodAgent(BaseAgent):
    agent_name = "scripted"

    def decide(
        self, task: dict[str, Any], history: list[dict[str, Any]], step_id: int
    ) -> AgentDecision:
        order_id = task["controlled_variables"]["order_id"]
        if not history:
            return scripted_decision({"action": "get_order", "arguments": {"order_id": order_id}})
        observation = history[-1].get("tool_result") or {}
        if history[-1]["action"]["action"] == "get_order":
            if observation.get("success") and can_cancel_order(observation.get("status")):
                return scripted_decision(
                    {"action": "cancel_order", "arguments": {"order_id": order_id}}
                )
            return scripted_decision(
                {"action": "respond", "message": "订单当前状态不允许直接取消。"}
            )
        return scripted_decision({"action": "respond", "message": "订单取消请求已处理。"})

