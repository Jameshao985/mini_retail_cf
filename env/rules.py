"""Business policies kept independent from tool implementation."""

from __future__ import annotations

from typing import Any


CANCELLABLE_STATUSES = frozenset({"pending", "processing"})
CANCEL_ERROR_BY_STATUS = {
    "shipped": "ORDER_ALREADY_SHIPPED",
    "cancelled": "ORDER_ALREADY_CANCELLED",
}


def can_cancel_order(status: str) -> bool:
    return status in CANCELLABLE_STATUSES


def evaluate_cancel_precondition(order: dict[str, Any]) -> dict[str, Any]:
    status = order.get("status")
    if can_cancel_order(status):
        return {"allowed": True, "status": status, "error": None}
    return {
        "allowed": False,
        "status": status,
        "error": CANCEL_ERROR_BY_STATUS.get(status, "ORDER_STATUS_NOT_CANCELLABLE"),
    }

