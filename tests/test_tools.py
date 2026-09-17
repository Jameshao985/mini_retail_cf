import pytest

from env.environment import MiniRetailEnvironment
from env.tools import RetailTools


def make_tools(status):
    environment = MiniRetailEnvironment(
        {"orders": {"1001": {"user_id": "u", "item": "x", "price": 1, "status": status}}}
    )
    return environment, RetailTools(environment)


@pytest.mark.parametrize("status", ["pending", "processing"])
def test_cancellable_statuses(status):
    environment, tools = make_tools(status)
    result = tools.cancel_order("1001")
    assert result == {
        "success": True,
        "order_id": "1001",
        "old_status": status,
        "new_status": "cancelled",
    }
    assert environment.query_order("1001")["status"] == "cancelled"


@pytest.mark.parametrize(
    "status,error", [("shipped", "ORDER_ALREADY_SHIPPED"), ("cancelled", "ORDER_ALREADY_CANCELLED")]
)
def test_forbidden_statuses_are_not_modified(status, error):
    environment, tools = make_tools(status)
    assert tools.cancel_order("1001") == {"success": False, "order_id": "1001", "error": error}
    assert environment.query_order("1001")["status"] == status


def test_get_order_is_structured():
    _, tools = make_tools("pending")
    result = tools.get_order("1001")
    assert result["success"] is True
    assert result["status"] == "pending"
    assert set(result) == {"success", "order_id", "status", "item", "price"}

