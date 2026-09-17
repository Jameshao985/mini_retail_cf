"""Shared deterministic multi-step interaction loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from env.environment import MiniRetailEnvironment
from env.tools import RetailTools
from utils.action_parser import ParseResult


@dataclass
class AgentDecision:
    raw_output: str
    parse_result: ParseResult
    prompt_messages: list[dict[str, str]]


class BaseAgent:
    agent_name = "base"

    def metadata(self) -> dict[str, Any]:
        return {"agent_name": self.agent_name}

    def decide(
        self,
        task: dict[str, Any],
        history: list[dict[str, Any]],
        step_id: int,
    ) -> AgentDecision:
        raise NotImplementedError

    def run_episode(
        self,
        task: dict[str, Any],
        environment: MiniRetailEnvironment,
        max_steps: int = 6,
    ) -> dict[str, Any]:
        initial_state = environment.reset(task["initial_state"])
        tools = RetailTools(environment)
        history: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        termination_reason = "max_steps_exceeded"

        for step_id in range(1, max_steps + 1):
            before = environment.snapshot()
            decision = self.decide(task, history, step_id)
            parsed = decision.parse_result.action
            step: dict[str, Any] = {
                "step_id": step_id,
                "prompt_messages": decision.prompt_messages,
                "raw_model_output": decision.raw_output,
                "parsed_action": parsed,
                "parse_error": decision.parse_result.parse_error,
                "parse_error_message": decision.parse_result.error,
                "tool_name": None,
                "tool_arguments": None,
                "tool_result": None,
                "environment_state_before": before,
                "environment_state_after": before,
            }
            if decision.parse_result.parse_error or parsed is None:
                steps.append(step)
                termination_reason = "parse_error"
                break

            action = parsed["action"]
            if action == "respond":
                step["environment_state_after"] = environment.snapshot()
                steps.append(step)
                history.append({"action": parsed, "tool_result": None})
                termination_reason = "agent_responded"
                break

            arguments = parsed.get("arguments", {})
            result = tools.execute(action, arguments)
            after = environment.snapshot()
            step.update(
                {
                    "tool_name": action,
                    "tool_arguments": arguments,
                    "tool_result": result,
                    "environment_state_after": after,
                }
            )
            steps.append(step)
            history.append({"action": parsed, "tool_result": result, "raw_output": decision.raw_output})

        return {
            "initial_state": initial_state,
            "steps": steps,
            "final_state": environment.get_state(),
            "termination_reason": termination_reason,
        }


def scripted_decision(action: dict[str, Any]) -> AgentDecision:
    raw = json.dumps(action, ensure_ascii=False)
    return AgentDecision(raw, ParseResult(action, False, None), [])

