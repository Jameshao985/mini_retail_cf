from env.environment import MiniRetailEnvironment


def state(status="pending"):
    return {"orders": {"1001": {"user_id": "u", "item": "x", "price": 1, "status": status}}}


def test_environment_reset_and_restore():
    environment = MiniRetailEnvironment(state())
    snapshot = environment.snapshot()
    environment.update_order_status("1001", "cancelled")
    assert environment.get_state()["orders"]["1001"]["status"] == "cancelled"
    environment.restore(snapshot)
    assert environment.get_state() == state()


def test_reset_does_not_alias_input():
    initial = state()
    environment = MiniRetailEnvironment(initial)
    environment.update_order_status("1001", "cancelled")
    assert initial["orders"]["1001"]["status"] == "pending"
    environment.reset(state("shipped"))
    assert environment.get_state()["orders"]["1001"]["status"] == "shipped"

