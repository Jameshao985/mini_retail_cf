"""Offline Qwen3 Student with a minimal JSON tool-action protocol."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MAX_NEW_TOKENS, MODEL_NAME, RULE_VISIBILITY, SEED
from utils.action_parser import parse_action

from .base_agent import AgentDecision, BaseAgent


PROVIDED_POLICY = """业务 policy：取消订单属于会修改状态的操作。执行前必须先查询订单状态；pending 或 processing 可取消，shipped 或 cancelled 不可取消。工具不会告诉你未知订单的当前状态，必须调用 get_order 获取。"""


def build_system_prompt(order_id: str) -> str:
    """Build the protocol prompt with the current task's order ID.

    A task-specific example prevents the old hard-coded ``1001`` from becoming
    an accidental model prior while still showing the exact JSON schema.
    """
    examples = [
        {"action": "get_order", "arguments": {"order_id": order_id}},
        {"action": "cancel_order", "arguments": {"order_id": order_id}},
        {"action": "respond", "message": "给用户的最终回复"},
    ]
    return "\n".join(
        [
            "你是一个电商订单助手。每一步只能输出一个 JSON 对象，不能输出两个动作。",
            "可用动作只有：",
            f"1. {json.dumps(examples[0], ensure_ascii=False)}",
            f"2. {json.dumps(examples[1], ensure_ascii=False)}",
            f"3. {json.dumps(examples[2], ensure_ascii=False)}",
            "你必须根据用户请求和工具返回结果选择下一步。不要输出 Markdown。",
        ]
    )


class QwenAgent(BaseAgent):
    agent_name = "qwen"

    def __init__(
        self,
        model_path: str | Path,
        rule_visibility: str = RULE_VISIBILITY,
        seed: int = SEED,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        if rule_visibility not in {"provided", "hidden"}:
            raise ValueError("rule_visibility must be 'provided' or 'hidden'")
        self.model_path = str(Path(model_path).resolve())
        self.rule_visibility = rule_visibility
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype="auto",
        )
        self.model.to(self.device)
        self.model.eval()

    def metadata(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "model_name": MODEL_NAME,
            "model_path": self.model_path,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": self.device,
            "cuda_device_name": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "generation_config": {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
            },
            "seed": self.seed,
            "rule_visibility": self.rule_visibility,
        }

    def _messages(
        self, task: dict[str, Any], history: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        system = build_system_prompt(task["controlled_variables"]["order_id"])
        if self.rule_visibility == "provided":
            system += "\n" + PROVIDED_POLICY
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task["user_request"]},
        ]
        for event in history:
            messages.append({"role": "assistant", "content": event["raw_output"]})
            messages.append(
                {
                    "role": "user",
                    "content": "工具返回：" + json.dumps(event["tool_result"], ensure_ascii=False),
                }
            )
        return messages

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> Any:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError):
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def decide(
        self, task: dict[str, Any], history: list[dict[str, Any]], step_id: int
    ) -> AgentDecision:
        messages = self._messages(task, history)
        encoded = self._apply_chat_template(messages)
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
            attention_mask = getattr(encoded, "attention_mask", None)
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = encoded
            attention_mask = None
        input_ids = input_ids.to(self.device)
        attention_mask = (
            attention_mask.to(self.device) if attention_mask is not None else torch.ones_like(input_ids)
        )
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, input_ids.shape[-1] :]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return AgentDecision(raw_output, parse_action(raw_output), messages)
