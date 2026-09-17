"""Conservative JSON action extraction without semantic correction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


ALLOWED_ACTIONS = frozenset({"get_order", "cancel_order", "respond"})


@dataclass(frozen=True)
class ParseResult:
    action: dict[str, Any] | None
    parse_error: bool
    error: str | None = None


def _first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found")


def _validate_schema(value: dict[str, Any]) -> None:
    action = value.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported action: {action!r}")
    if action == "respond":
        if not isinstance(value.get("message"), str):
            raise ValueError("respond requires a string message")
        return
    arguments = value.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"{action} requires an arguments object")
    if "order_id" not in arguments or not isinstance(arguments["order_id"], (str, int)):
        raise ValueError(f"{action} requires a scalar order_id")


def parse_action(raw_output: str) -> ParseResult:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return ParseResult(None, True, "empty model output")
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = _first_json_object(cleaned)
        _validate_schema(value)
        return ParseResult(value, False, None)
    except (ValueError, TypeError) as exc:
        return ParseResult(None, True, str(exc))

