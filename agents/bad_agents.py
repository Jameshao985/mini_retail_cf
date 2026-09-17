"""Known-bad controls that exercise distinct verifier failure paths."""

from __future__ import annotations

from typing import Any

from .base_agent import AgentDecision, BaseAgent, scripted_decision


class BadAgentAlwaysCancel(BaseAgent):
    agent_name = "always_cancel"

    def decide(
        self, task: dict[str, Any], history: list[dict[str, Any]], step_id: int
    ) -> AgentDecision:
        order_id = task["controlled_variables"]["order_id"]
        if not history:
            return scripted_decision(
                {"action": "cancel_order", "arguments": {"order_id": order_id}}
            )
        return scripted_decision({"action": "respond", "message": "已尝试取消订单。"})


class BadAgentChecksButIgnores(BaseAgent):
    agent_name = "checks_but_ignores"

    def decide(
        self, task: dict[str, Any], history: list[dict[str, Any]], step_id: int
    ) -> AgentDecision:
        order_id = task["controlled_variables"]["order_id"]
        if not history:
            return scripted_decision({"action": "get_order", "arguments": {"order_id": order_id}})
        if history[-1]["action"]["action"] == "get_order":
            return scripted_decision(
                {"action": "cancel_order", "arguments": {"order_id": order_id}}
            )
        return scripted_decision({"action": "respond", "message": "已尝试取消订单。"})

