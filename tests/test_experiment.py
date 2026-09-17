import copy
import json

import pytest

from experiment.tasks import (CAPABILITIES, assert_disjoint, make_composition, make_pair,
                              make_protocol_tasks, make_split, validate_pair)
from experiment.world import SYSTEM, World, parse, rollout, teacher_examples
from experiment.discovery import allocate, cluster_failures, extract_failures, synthesize
from experiment.report import metrics, paired_interval


@pytest.mark.parametrize('capability', CAPABILITIES)
def test_oracle_pairs_pass_and_are_minimal(capability):
    for seed in (1, 42, 99):
        pair = make_pair('test', capability, 0, seed)
        assert validate_pair(*pair)
        assert all(rollout(t, None)['task_success'] for t in pair)
        changed = copy.deepcopy(pair)
        changed[1]['initial_state']['orders'][changed[1]['target']]['price'] += 1
        with pytest.raises(ValueError):
            validate_pair(*changed)


def test_no_hidden_target_in_missing_information_prompt():
    task = make_pair('train', 'clarification', 0)[1]
    world = World(task)
    prompt = json.dumps(world.messages, ensure_ascii=False)
    assert task['target'] not in prompt
    assert task['capability'] not in prompt
    assert world.expected()['action'] == 'ask_user'
    world.apply(json.dumps({'action': 'get_order', 'order_id': task['target']}))
    assert 'missing_clarification' in world.errors  # Lucky guessing is still a failure.


def test_forbidden_attempt_fails_even_when_state_is_protected():
    task = make_pair('test', 'precondition', 0)[1]
    world = World(task)
    oid = task['target']
    for action in ({'action': 'get_order', 'order_id': oid}, {'action': 'cancel_order', 'order_id': oid},
                   {'action': 'respond', 'order_id': oid, 'outcome': 'refused'}):
        world.apply(json.dumps(action))
    result = world.result()
    assert result['terminal_correct']
    assert not result['task_success']
    assert result['violations'] == ['forbidden_action']


def test_correction_requires_new_target_and_preserves_old_order():
    task = make_pair('test', 'correction', 0)[1]
    good = rollout(task, None)
    assert good['task_success']
    assert [s['action']['action'] for s in good['steps']] == ['get_order', 'get_order', 'cancel_order', 'respond']
    world = World(task)
    old = task['initial_target']
    world.apply(json.dumps({'action': 'get_order', 'order_id': old}))
    world.apply(json.dumps({'action': 'cancel_order', 'order_id': old}))
    assert 'stale_target' in world.errors
    assert not world.result()['terminal_correct']


def test_refusal_cannot_pass_allowed_task_and_double_json_rejected():
    task = make_pair('test', 'precondition', 0)[0]
    world = World(task)
    world.apply(json.dumps({'action': 'get_order', 'order_id': task['target']}))
    world.apply(json.dumps({'action': 'respond', 'order_id': task['target'], 'outcome': 'refused'}))
    assert 'over_refusal' in world.result()['violations']
    assert parse('{"action":"ask_user","field":"order_id"} {}')[1]


@pytest.mark.parametrize('capability,field', [('refund_window', 'delivered_days'),
                                               ('refund_condition', 'used')])
def test_refund_pairs_cross_exactly_one_policy_boundary(capability, field):
    left, right = make_pair('test', capability, 2)
    assert left['operation'] == right['operation'] == 'refund'
    assert left['initial_state']['orders'][left['target']][field] != right['initial_state']['orders'][right['target']][field]
    assert rollout(left, None)['gold_outcome'] == 'refunded'
    assert rollout(right, None)['gold_outcome'] == 'refused'


def test_composition_pairs_keep_one_edit_but_mask_both_outcomes():
    tasks = make_composition(4, 42)
    assert len(tasks) == 8
    assert all(rollout(t, None)['task_success'] for t in tasks)
    assert all(rollout(t, None)['gold_outcome'] == 'refused' for t in tasks)
    for left, right in zip(tasks[::2], tasks[1::2]):
        assert left['pair_kind'] == right['pair_kind'] == 'masked_boundary'
        assert left['intervention_variable'] == right['intervention_variable']


def test_splits_no_groups_or_entity_leakage():
    splits = [make_split(s, 5, 42) for s in ('discovery', 'train', 'dev', 'test', 'ood')]
    assert_disjoint(make_protocol_tasks(6, 42), *splits)
    with pytest.raises(ValueError):
        assert_disjoint(splits[0], splits[0])


def test_first_observed_error_and_embedding_excludes_identifiers():
    t = make_pair('discovery', 'clarification', 0)[1]
    ep = rollout(t, lambda _: json.dumps({'action': 'respond', 'order_id': t['target'], 'outcome': 'cancelled'}))
    report = cluster_failures([ep])
    failure = report['failures'][0]
    assert failure['step'] == 1
    assert failure['code'] == 'missing_clarification'
    assert t['target'] not in failure['description']
    assert report['weights']['clarification'] > report['weights']['correction']
    assert sum(allocate(7, report['weights']).values()) == 7


def test_failure_description_uses_only_state_visible_before_bad_action():
    task = make_pair('discovery', 'precondition', 0)[0]
    episode = rollout(task, lambda _: json.dumps({'action': 'cancel_order', 'order_id': task['target']}))
    failure = extract_failures([episode])[0]
    assert failure['evidence']['visible_before_action'] == {}
    assert failure['evidence']['action_observation']['status'] == 'cancelled'
    assert "可见状态={}" in failure['description']


def test_cluster_backend_is_actually_used_and_empty_pool_has_uniform_fallback():
    tasks = make_split('discovery', 1, 42)
    good = [rollout(t, None) for t in tasks]
    assert cluster_failures(good)['embedding'] == 'none_no_failures'
    bad = [rollout(t, lambda _: 'not json') for t in tasks]
    report = cluster_failures(bad)
    assert len(report['failures']) == len(tasks)
    assert sum(c['count'] for c in report['clusters']) == len(tasks)
    assert all('cluster_id' in f for f in report['failures'])


def test_synthesis_fair_task_and_side_counts_and_replay():
    analysis = dict(weights={c: (.6 if c == 'precondition' else .1) for c in CAPABILITIES}, failures=[])
    ordinary, plain_rows, _ = synthesize('targeted', 15, analysis, 42, 'test_plain')
    paired, pair_rows, _ = synthesize('targeted_cf', 15, analysis, 42, 'test_pairs')
    assert len(ordinary) == len(paired) == 30
    assert sum(t['variant'] == 0 for t in ordinary) == sum(t['variant'] == 0 for t in paired) == 15
    assert {r['teacher'] for r in plain_rows + pair_rows} == {'rule_oracle'}
    assert all(r['verified'] for r in plain_rows + pair_rows)
    assert all(t['pair_id'] is None for t in ordinary)
    for capability in ('clarification', 'correction'):
        plain_ops = [t['operation'] for t in ordinary if t['capability'] == capability]
        paired_ops = [t['operation'] for t in paired if t['capability'] == capability]
        assert plain_ops.count('cancel') == paired_ops.count('cancel')
        assert plain_ops.count('refund') == paired_ops.count('refund')
    assert_disjoint(ordinary, paired)


def test_pair_score_requires_both_correct_and_ci_is_paired():
    tasks = make_split('test', 1, 42)
    good = [rollout(t, None) for t in tasks]
    bad = [rollout(t, lambda _: 'invalid') for t in tasks]
    assert metrics(good)['pair_both_correct'] == 1
    assert metrics(bad)['pair_both_correct'] == 0
    assert paired_interval(bad, good, repeats=100)['ci95_low'] == 1
    with pytest.raises(ValueError):
        metrics(good[:-1])


def test_teacher_contexts_are_exactly_student_contexts():
    task = make_pair('train', 'correction', 0)[1]
    examples = teacher_examples(task)
    world = World(task)
    for e in examples:
        assert e['messages'] == world.messages
        assert e['messages'][0]['content'] == SYSTEM
        world.apply(e['response'])
    assert world.result()['task_success']


def test_loss_mask_excludes_all_prompt_tokens():
    # A tokenizer stub with a nonempty generation prefix tests masking independently.
    from experiment.model import encode_example
    class TinyTokenizer:
        eos_token_id = 99
        def apply_chat_template(self, messages, **kw):
            assert kw['enable_thinking'] is False
            assert kw['add_generation_prompt'] is True
            return 'prompt'
        def encode(self, text, **kw):
            return [1, 2, 3, 4] if text == 'prompt' else [10, 11]
    row = dict(task_id='x', messages=[], response='answer')
    encoded = encode_example(TinyTokenizer(), row, 8)
    assert encoded['labels'] == [-100, -100, -100, -100, 10, 11, 99]
    with pytest.raises(ValueError):
        encode_example(TinyTokenizer(), row, 6)


def test_clarification_changes_only_order_id_sentence():
    left, right = make_pair('train', 'clarification', 4)
    assert left['request'].replace(f'订单号是{left["target"]}。', '') == right['request']
