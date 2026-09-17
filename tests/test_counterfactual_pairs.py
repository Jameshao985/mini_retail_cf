from copy import deepcopy

import pytest

from env.verifier import validate_pair_purity
from runners.common import execute_tasks, load_tasks
from agents import ScriptedGoodAgent
from data.generate_validation_tasks import build_validation_tasks
from data.generate_replication_tasks import build_replication_tasks


def test_all_ten_pairs_are_pure():
    tasks = load_tasks()
    assert len(tasks) == 60
    assert len({task["pair_id"] for task in tasks}) == 30
    assert {task["intervention_value"] for task in tasks} == {
        "pending",
        "processing",
        "shipped",
        "cancelled",
    }
    for index in range(0, len(tasks), 2):
        assert validate_pair_purity(tasks[index], tasks[index + 1])


def test_pair_purity_rejects_controlled_variable_drift():
    pair_a, pair_b = load_tasks()[:2]
    impure = deepcopy(pair_b)
    impure["controlled_variables"]["price"] += 1
    with pytest.raises(ValueError, match="controlled field"):
        validate_pair_purity(pair_a, impure)


def test_batch_tasks_are_state_isolated():
    tasks = load_tasks()[:4]
    episodes = execute_tasks(ScriptedGoodAgent(), tasks)
    for task, episode in zip(tasks, episodes):
        assert episode["initial_state"] == task["initial_state"]
    assert episodes[0]["final_state"] != episodes[1]["final_state"]
    assert episodes[2]["initial_state"] == tasks[2]["initial_state"]


def test_validation_tasks_are_independent_and_pure():
    tasks = build_validation_tasks()
    assert len(tasks) == 60
    assert all(task["pair_id"].startswith("val_pair_") for task in tasks)
    assert not {task["controlled_variables"]["order_id"] for task in tasks} & {
        task["controlled_variables"]["order_id"] for task in load_tasks()
    }
    pairs = {}
    for task in tasks:
        pairs.setdefault(task["pair_id"], []).append(task)
    assert len(pairs) == 30
    for pair in pairs.values():
        assert validate_pair_purity(*sorted(pair, key=lambda item: item["task_id"]))


def test_replication_tasks_are_command_style_and_independent():
    tasks = build_replication_tasks()
    assert len(tasks) == 60
    assert all("取消" in task["user_request"] for task in tasks)
    assert all(task["pair_id"].startswith("rep_pair_") for task in tasks)
    old_ids = {
        task["controlled_variables"]["order_id"]
        for task in load_tasks()
    }
    validation_ids = {
        task["controlled_variables"]["order_id"]
        for task in build_validation_tasks()
    }
    replication_ids = {task["controlled_variables"]["order_id"] for task in tasks}
    assert not old_ids & replication_ids
    assert not validation_ids & replication_ids
