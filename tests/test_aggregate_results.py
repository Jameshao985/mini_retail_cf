import csv

import pytest

from experiment.io import dump
from scripts.aggregate_results import aggregate


def write_run(folder, seed, score):
    folder.mkdir()
    dump(folder / 'COMPLETE.json', {'status': 'complete'})
    dump(folder / 'manifest.json', {'config': {'seed': seed, 'profile': 'fixture'}, 'source_hash': 'fixed'})
    row = dict(arm='targeted_cf', split='test', success_rate=score, pair_both_correct=score,
               allowed_task_success=score, forbidden_action_rate=0, protocol_error_rate=0)
    with (folder / 'results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)


def test_aggregation_uses_independent_completed_seeds(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    write_run(a, 42, .5)
    write_run(b, 43, 1.)
    result = next(r for r in aggregate([a, b]) if r['metric'] == 'success_rate')
    assert result['mean'] == .75
    assert result['sample_std'] == pytest.approx(.35355339)
    with pytest.raises(ValueError, match='Duplicate seed'):
        aggregate([a, a])


def test_aggregation_rejects_incompatible_runs(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    write_run(a, 42, .5)
    write_run(b, 43, 1.)
    dump(b / 'manifest.json', {'config': {'seed': 43, 'profile': 'different'}, 'source_hash': 'fixed'})
    with pytest.raises(ValueError, match='Different configurations'):
        aggregate([a, b])
