"""Small controlled, post-hoc decision probe using existing local adapters.

Provides the same valid query history and changes one visible field at a time.
Scores only the next action, NOT whole-task success or a new held-out benchmark.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from experiment.io import dump, read
from experiment.tasks import REFUND_REQUESTS, decision, make_pair
from experiment.world import World, parse, rollout


def make_probe_tasks():
    tasks = []
    for context in range(2):
        base = make_pair('discovery', 'refund_window', context, 20260917, 'audit_probe_v1')[0]
        base['request'] = REFUND_REQUESTS['discovery'][context].format(a=base['target'])
        original_price = base['initial_state']['orders'][base['target']]['price']
        for used in (False, True):
            for days in (6, 7, 8):
                for price_change in (0, 17):
                    task = copy.deepcopy(base)
                    task['task_id'] = f"audit_probe_v1/c{context}/u{int(used)}/d{days}/p{price_change}"
                    task['pair_id'] = None
                    task['group_id'] = f'audit_probe_v1/c{context}'
                    task['probe'] = dict(context=context, used=used, days=days, price_change=price_change)
                    order = task['initial_state']['orders'][task['target']]
                    order.update(status='delivered', used=used, delivered_days=days,
                                 price=original_price+price_change)
                    assert (decision(task) == 'refunded') == (not used and days <= 7)
                    assert rollout(task, None)['task_success']
                    tasks.append(task)
    return tasks


def query_context(task):
    world = World(task)
    world.apply(json.dumps({'action': 'get_order', 'order_id': task['target']}))
    assert not world.errors
    return world


def summarize(rows):
    indexed = {(r['context'], r['used'], r['days'], r['price_change']): r for r in rows}
    assert len(indexed) == 24
    def both(keys):
        return all(indexed[k]['decision_correct'] for k in keys)
    window, condition, mask, price = [], [], [], []
    for c in range(2):
        for p in (0, 17):
            window.append(both([(c, False, 7, p), (c, False, 8, p)]))
            condition.append(both([(c, False, 6, p), (c, True, 6, p)]))
            mask.append(both([(c, True, 7, p), (c, True, 8, p)]))
        for u in (False, True):
            for d in (6, 7, 8):
                price.append(both([(c, u, d, 0), (c, u, d, 17)]))
    return dict(decisions=len(rows), correct=sum(r['decision_correct'] for r in rows),
                days7vs8_both_correct=[sum(window), len(window)],
                unusedvsused_both_correct=[sum(condition), len(condition)],
                used_masks_time_both_refuse=[sum(mask), len(mask)],
                price_both_correct=[sum(price), len(price)],
                note='Pairs overlap and contexts are only two; no independent-sample inference.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    from experiment.model import Student
    cfg = read(args.run / 'manifest.json')['config']
    tasks = make_probe_tasks()
    old_ids = set()
    for old in args.run.glob('tasks/*.json'):
        for task in read(old):
            old_ids.update(task['initial_state']['orders'])
    assert not old_ids & {oid for t in tasks for oid in t['initial_state']['orders']}
    dump(args.output / 'tasks.json', tasks)
    all_rows, summaries = [], {}
    started = time.monotonic()
    for arm in ('base', 'random', 'targeted', 'random_cf', 'targeted_cf'):
        adapter = (args.run / 'common_warmup/train/adapter' if arm == 'base'
                   else args.run / arm / 'round_1/train/adapter')
        model = Student(cfg['model_path'], cfg['device'], cfg['seed'], cfg['max_new_tokens'], adapter)
        rows = []
        try:
            for i, task in enumerate(tasks):
                world = query_context(task)
                expected = world.expected()
                raw = model(world.messages)
                action, error = parse(raw)
                row = dict(arm=arm, task_id=task['task_id'], **task['probe'],
                           expected=expected, actual=action, raw=raw, parse_error=error,
                           decision_correct=action == expected, messages=copy.deepcopy(world.messages))
                rows.append(row)
                all_rows.append(row)
                dump(args.output / 'partial_decisions.json', all_rows)
                if (i + 1) % 6 == 0:
                    print(f'{arm}: {i+1}/{len(tasks)} conditional decisions', flush=True)
        finally:
            model.close()
        summaries[arm] = summarize(rows)
        print(f'{arm}: {summaries[arm]}', flush=True)
    dump(args.output / 'results.json', dict(scope='post_hoc_oracle_query_conditioned_probe',
        model_seed=cfg['seed'], context_count=2, source_run=str(args.run.resolve()),
        summaries=summaries, elapsed_seconds=time.monotonic()-started, decisions=all_rows))
    lines = ['# 退款条件干预小实验', '',
        '使用seed44已有五个模型，不重新训练、不调用API。每个模型24次下一动作预测。',
        '统一由程序先执行正确查询，学生只决定下一步；它测的是给定正确上下文的条件判断，不是完整任务成功率。',
        '两个发现集措辞/订单背景，交叉已使用状态×6/7/8天×原价/加17元。除目标字段外，同一背景完全固定。',
        '只有两个背景，各种对照重复使用预测，不能作为120个独立样本做显著性检验或正式方法结论。', '',
        '| 模型 | 正确动作/24 | 未使用7→8天，两侧对/4 | 6天未使用→已使用，两侧对/4 | 已使用7→8天，均拒绝/4 | 价格变化两侧对/12 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, s in summaries.items():
        def fmt(key):
            return '/'.join(str(n) for n in s[key])
        lines.append(f"| {arm} | {s['correct']}/24 | {fmt('days7vs8_both_correct')} | "
                     f"{fmt('unusedvsused_both_correct')} | {fmt('used_masks_time_both_refuse')} | {fmt('price_both_correct')} |")
    lines += ['', '## 逐条件动作（两种价格依次列出）', '',
              '| 模型 | 背景 | 已使用 | 天数 | 正确动作 | 原价预测 | 改价预测 |',
              '|---|---:|---|---:|---|---|---|']
    def action_name(action):
        if action is None:
            return 'invalid'
        return action['action'] + ('/'+action['outcome'] if 'outcome' in action else '')
    for arm in summaries:
        rows = {(r['context'], r['used'], r['days'], r['price_change']): r for r in all_rows if r['arm'] == arm}
        for c in range(2):
            for u in (False, True):
                for d in (6, 7, 8):
                    a, b = rows[c,u,d,0], rows[c,u,d,17]
                    lines.append(f"| {arm} | {c} | {u} | {d} | {action_name(a['expected'])} | {action_name(a['actual'])} | {action_name(b['actual'])} |")
    (args.output / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    dump(args.output / 'COMPLETE.json', dict(status='complete', elapsed_seconds=time.monotonic()-started))
    print(args.output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
