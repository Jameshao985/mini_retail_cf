"""Structured tools exposed to agents."""

from __future__ import annotations

from typing import Any

from .environment import MiniRetailEnvironment
from .rules import evaluate_cancel_precondition


class RetailTools:
    TOOL_NAMES = frozenset({"get_order", "cancel_order"})

    def __init__(self, environment: MiniRetailEnvironment) -> None:
        self.environment = environment

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self.TOOL_NAMES:
            return {"success": False, "error": "UNKNOWN_TOOL", "tool_name": tool_name}
        order_id = arguments.get("order_id") if isinstance(arguments, dict) else None
        if not isinstance(order_id, str) or not order_id:
            return {"success": False, "error": "INVALID_ORDER_ID", "order_id": order_id}
        if tool_name == "get_order":
            return self.get_order(order_id)
        return self.cancel_order(order_id)

    def get_order(self, order_id: str) -> dict[str, Any]:
        order = self.environment.query_order(order_id)
        if order is None:
            return {"success": False, "order_id": order_id, "error": "ORDER_NOT_FOUND"}
        return {
            "success": True,
            "order_id": order_id,
            "status": order["status"],
            "item": order["item"],
            "price": order["price"],
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        order = self.environment.query_order(order_id)
        if order is None:
            return {"success": False, "order_id": order_id, "error": "ORDER_NOT_FOUND"}
        decision = evaluate_cancel_precondition(order)
        if not decision["allowed"]:
            return {"success": False, "order_id": order_id, "error": decision["error"]}
        old_status, new_status = self.environment.update_order_status(order_id, "cancelled")
        return {
            "success": True,
            "order_id": order_id,
            "old_status": old_status,
            "new_status": new_status,
        }

