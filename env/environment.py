"""In-memory JSON-like state with explicit reset/snapshot/restore semantics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class MiniRetailEnvironment:
    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = {}
        if initial_state is not None:
            self.reset(initial_state)

    def reset(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        self._state = deepcopy(initial_state)
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def snapshot(self) -> dict[str, Any]:
        return self.get_state()

    def restore(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self._state = deepcopy(snapshot)
        return self.get_state()

    def query_order(self, order_id: str) -> dict[str, Any] | None:
        order = self._state.get("orders", {}).get(order_id)
        return deepcopy(order) if order is not None else None

    def update_order_status(self, order_id: str, new_status: str) -> tuple[str, str]:
        orders = self._state.get("orders", {})
        if order_id not in orders:
            raise KeyError(order_id)
        old_status = orders[order_id]["status"]
        orders[order_id]["status"] = new_status
        return old_status, new_status

