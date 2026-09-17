import copy

from scripts.audit_study import sampled_indices
from scripts.probe_refund_decisions import make_probe_tasks, query_context, summarize


def test_sampler_epoch_boundary_and_partial_epoch():
    first = sampled_indices(400, 80, 4, 44)
    assert len(first) == len(set(first)) == 320
    across = sampled_indices(400, 104, 4, 44)
    assert across[:320] == first
    assert set(across[:400]) == set(range(400))
    assert len(across) == 416
    assert len(set(across[400:])) == 16


def test_probe_is_controlled_and_oracle_decisions_all_pass():
    tasks = make_probe_tasks()
    rows = []
    for task in tasks:
        world = query_context(task)
        expected = world.expected()
        p = task['probe']
        assert expected['action'] == ('refund_order' if not p['used'] and p['days'] <= 7 else 'respond')
        if expected['action'] == 'respond':
            assert expected['outcome'] == 'refused'
        rows.append(dict(**p, decision_correct=True))
        other = next(t for t in tasks if t['probe'] == dict(p, price_change=17-p['price_change']))
        a, b = copy.deepcopy(task), copy.deepcopy(other)
        for t in (a, b):
            t.pop('task_id')
            t.pop('probe')
            t['initial_state']['orders'][t['target']]['price'] = 0
        assert a == b
    summary = summarize(rows)
    assert summary['correct'] == 24
    assert summary['days7vs8_both_correct'] == [4, 4]
    assert summary['used_masks_time_both_refuse'] == [4, 4]
    assert summary['price_both_correct'] == [12, 12]
