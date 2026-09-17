"""Orchestration tests use oracle controls, never report them as Qwen measurements."""
import copy
from pathlib import Path

from experiment.io import dump, jsonl
from experiment.run import PROFILES, pipeline
from experiment.world import rollout


def test_rounds_continue_only_their_own_adapter_and_reuse_completed_training(tmp_path, monkeypatch):
    import experiment.model as models
    import experiment.run as runner
    observed_training = []

    class ControlStudent:
        def __init__(self, *args, **kwargs):
            pass
        def close(self):
            pass

    def controlled_evaluation(tasks, student, output):
        episodes = [rollout(t, None) for t in tasks]
        jsonl(output, episodes)
        return episodes

    def controlled_training(model_path, rows, output, cfg, previous_adapter=None):
        observed_training.append((Path(output), previous_adapter))
        (Path(output) / 'adapter').mkdir(parents=True)
        dump(Path(output) / 'training_summary.json', {'unit_test_control': True})

    monkeypatch.setattr(models, 'Student', ControlStudent)
    monkeypatch.setattr(models, 'train_lora', controlled_training)
    monkeypatch.setattr(runner, 'evaluate', controlled_evaluation)
    cfg = dict(PROFILES['smoke'], profile='smoke', model_path='unit-test-control', device='cpu', seed=42,
               task_seed=20260914,
               max_new_tokens=96, embedding='tfidf', cluster_threshold=.25, rounds=2,
               arms=['random', 'targeted_cf'])
    pipeline(tmp_path, cfg, 'baseline')
    pipeline(tmp_path, cfg, 'train')
    assert len(observed_training) == 5
    for path, previous in observed_training:
        if 'common_warmup' in path.parts:
            assert previous is None
            continue
        arm = path.parent.parent.name
        if path.parent.name == 'round_1':
            assert previous == tmp_path / 'common_warmup' / 'train' / 'adapter'
        else:
            assert previous == tmp_path / arm / 'round_1' / 'train' / 'adapter'
    # Resume completed rounds: no accidental extra updates or cross-arm contamination.
    pipeline(tmp_path, cfg, 'train')
    assert len(observed_training) == 5


def test_task_checkpoints_reject_changed_tasks(tmp_path):
    from experiment.run import evaluate
    from experiment.tasks import make_split
    tasks = make_split('test', 1, 42)
    path = tmp_path / 'episodes.jsonl'
    original = [rollout(t, None) for t in tasks]
    jsonl(path, original)
    def must_not_run(messages):
        raise AssertionError('Completed episode should be reused')
    assert evaluate(tasks, must_not_run, path) == original
    changed = copy.deepcopy(tasks)
    changed[0]['request'] += ' Changed'
    import pytest
    with pytest.raises(ValueError, match='Task changed'):
        evaluate(changed, must_not_run, path)
