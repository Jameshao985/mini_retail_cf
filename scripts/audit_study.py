"""Reconstruct saved study exposure and audit attribution without changing runs.

Uses the archived sampler, verified oracle contexts and saved token totals.
No model inference, training, API requests or edits to existing experiment files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from experiment.discovery import allocate, cluster_failures, extract_failures
from experiment.io import dump, read
from experiment.tasks import CAPABILITIES, decision
from experiment.world import rollout


def load_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]


def sampled_indices(count, steps, accum, seed):
    rng = random.Random(seed)
    order, cursor, result = [], 0, []
    for _ in range(steps * accum):
        if cursor == len(order):
            order = list(range(count))
            rng.shuffle(order)
            cursor = 0
        result.append(order[cursor])
        cursor += 1
    return result


def example_key(task_id, messages, response):
    return json.dumps([task_id, messages, json.loads(response)], ensure_ascii=False, sort_keys=True)


def token_lengths(tokenizer, rows):
    lengths = []
    for r in rows:
        prefix = tokenizer.apply_chat_template(r['messages'], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        p = tokenizer.encode(prefix, add_special_tokens=False)
        a = tokenizer.encode(r['response'], add_special_tokens=False) + [tokenizer.eos_token_id]
        lengths.append((len(p) + len(a), len(a)))
    return lengths


def audit_training(run, arm, tokenizer):
    cfg = read(run / 'manifest.json')['config']
    folder = run / arm / 'round_1'
    tasks = read(folder / 'tasks.json')
    rows = load_rows(folder / 'sft.jsonl')
    summary = read(folder / 'train/training_summary.json')
    selected = sampled_indices(len(rows), cfg['train_steps'], cfg['grad_accum'], cfg['seed'])
    seen = set(selected)
    by_key = {example_key(r['task_id'], r['messages'], r['response']): i for i, r in enumerate(rows)}
    assert len(by_key) == len(rows), 'Ambiguous duplicate action contexts'
    lengths = token_lengths(tokenizer, rows)
    assert sum(x[0] for x in lengths) == summary['dataset_input_tokens']
    assert sum(x[1] for x in lengths) == summary['dataset_supervised_tokens']
    assert sum(lengths[i][0] for i in selected) == summary['seen_input_tokens']
    assert sum(lengths[i][1] for i in selected) == summary['seen_supervised_tokens']
    assert len(rows) == summary['examples']
    assert cfg['train_steps'] == summary['optimizer_updates']
    full_tasks, decision_seen = 0, 0
    pair_episodes = defaultdict(list)
    teacher_episodes = {}
    states, selected_decisions = Counter(), Counter()
    exposure_by_capability = {}
    for t in tasks:
        ep = rollout(t, None)
        assert ep['task_success']
        indices = [by_key[example_key(t['task_id'], s['messages'], s['raw'])] for s in ep['steps']]
        assert len(indices) == len(set(indices))
        teacher_episodes[t['task_id']] = (ep, indices)
        full_tasks += set(indices).issubset(seen)
        decision_step = next(i for i, s in enumerate(ep['steps'])
                             if s['action']['action'] in ('cancel_order', 'refund_order')
                             or s['action'].get('outcome') == 'refused')
        is_seen = indices[decision_step] in seen
        decision_seen += is_seen
        label = f"{t['capability']}/{decision(t)}"
        selected_decisions[label] += int(is_seen)
        if t['operation'] == 'refund':
            o = t['initial_state']['orders'][t['target']]
            states[f"used={o['used']},over7={o['delivered_days'] > 7}"] += 1
        if t['pair_id']:
            pair_episodes[t['pair_id']].append((ep, indices))
    full_pairs, critical_pairs = 0, 0
    pair_detail = []
    for pair_id, members in sorted(pair_episodes.items()):
        assert len(members) == 2
        members.sort(key=lambda x: x[0]['task']['variant'])
        (a, ai), (b, bi) = members
        # The first teacher-action difference is the intended intervention's
        # immediate decision point, including ask-vs-query and target update.
        k = next(i for i, (sa, sb) in enumerate(zip(a['steps'], b['steps']))
                 if sa['action'] != sb['action'])
        both = ai[k] in seen and bi[k] in seen
        complete = set(ai + bi).issubset(seen)
        critical_pairs += both
        full_pairs += complete
        pair_detail.append(dict(pair_id=pair_id, capability=a['capability'],
                                critical_step=k + 1, left_seen=ai[k] in seen,
                                right_seen=bi[k] in seen, both_critical_seen=both,
                                all_actions_seen=complete))
    for c in CAPABILITIES:
        available = [i for i, r in enumerate(rows) if r['capability'] == c]
        exposure_by_capability[c] = dict(available=len(available),
                                         unique_seen=len(set(available) & seen))
    return dict(seed=cfg['seed'], arm=arm, tasks=len(tasks), action_rows=len(rows),
                optimizer_updates=cfg['train_steps'], examples_processed=len(selected),
                effective_action_epochs=len(selected) / len(rows),
                unique_actions_seen=len(seen), unseen_actions=len(rows)-len(seen),
                complete_task_trajectories_seen=full_tasks, business_decisions_seen=decision_seen,
                cf_pairs=len(pair_episodes), cf_pairs_both_critical_seen=critical_pairs,
                cf_pairs_all_actions_seen=full_pairs,
                minimum_updates_for_one_pass=math.ceil(len(rows)/cfg['grad_accum']),
                token_totals_match_saved_summary=True,
                exposure_by_capability=exposure_by_capability,
                selected_business_decisions=dict(selected_decisions),
                refund_training_conditions=dict(states), pair_detail=pair_detail)


def audit_episodes(episodes):
    failures = extract_failures(episodes)
    first_errors = Counter((f['capability'], f['code']) for f in failures)
    groups = defaultdict(list)
    for ep in episodes:
        if ep['pair_id']:
            groups[ep['pair_id']].append(ep)
    pair_patterns = defaultdict(Counter)
    for pair in groups.values():
        assert len(pair) == 2
        pair.sort(key=lambda e: e['task']['variant'])
        pattern = ''.join('correct' if e['task_success'] else 'wrong' for e in pair)
        pair_patterns[pair[0]['capability']][pattern] += 1
    refund = defaultdict(Counter)
    examples = []
    for ep in episodes:
        if ep['operation'] != 'refund':
            continue
        o = ep['task']['initial_state']['orders'][ep['task']['target']]
        label = f"used={o['used']},days={o['delivered_days']}"
        refund[label]['tasks'] += 1
        refund[label]['successes'] += ep['task_success']
        refund[label]['forbidden_attempts'] += 'forbidden_action' in ep['violations']
        # Restrict decision evidence to first query of the true target with no
        # previous violations. Do not label inaccessible-rule failures as reasoning.
        clean = True
        for i, step in enumerate(ep['steps']):
            obs = step['observation']
            if (clean and not step['violations'] and step['action']
                    and step['action']['action'] == 'get_order'
                    and isinstance(obs, dict) and obs.get('success')
                    and obs.get('order_id') == ep['task']['target']
                    and not step['user_continuation'] and i + 1 < len(ep['steps'])):
                nxt = ep['steps'][i+1]
                actual = nxt['action'] or {}
                category = actual.get('action', 'invalid')
                if category == 'respond':
                    category += '/' + actual.get('outcome', '')
                refund[label]['clean_post_query_decisions'] += 1
                refund[label]['next/' + category] += 1
                if nxt['action'] != nxt['expected'] and len(examples) < 6:
                    examples.append(dict(task_id=ep['task_id'], capability=ep['capability'],
                                         visible=obs, actual=actual, expected=nxt['expected'],
                                         violations=ep['violations']))
                break
            clean = clean and not step['violations']
    return dict(tasks=len(episodes), first_errors={f'{c}/{e}': n for (c, e), n in sorted(first_errors.items())},
                pair_patterns={c: dict(v) for c, v in pair_patterns.items()},
                refund_conditions={k: dict(v) for k, v in sorted(refund.items())},
                example_decision_errors=examples)


def audit_attribution(run):
    cfg = read(run / 'manifest.json')['config']
    saved = read(run / 'baseline/analysis.json')
    episodes = load_rows(run / 'baseline/discovery.jsonl')
    # Controlled ablation: same episodes, real TF-IDF vs saved Qwen analysis.
    tfidf = cluster_failures(episodes, None, cfg['cluster_threshold'])
    equal_weights = all(abs(tfidf['weights'][c] - saved['weights'][c]) < 1e-12 for c in CAPABILITIES)
    assert equal_weights
    assert allocate(cfg['train_pairs'], tfidf['weights']) == allocate(cfg['train_pairs'], saved['weights'])
    return dict(seed=cfg['seed'], configured_embedding=cfg['embedding'],
                recorded_embedding=saved['embedding'], qwen_clusters=len(saved['clusters']),
                tfidf_clusters=len(tfidf['clusters']),
                qwen_vs_tfidf_weights_identical=equal_weights,
                allocation_pairs=allocate(cfg['train_pairs'], saved['weights']),
                failure_rates=saved['failure_rates'], weights=saved['weights'],
                discovery=audit_episodes(episodes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a fresh audit directory: {args.output}')
    from transformers import AutoTokenizer
    runs = [ROOT / f'outputs/experiments/four_group_study_s{s}' for s in (42, 43, 44)]
    cfg = read(runs[0] / 'manifest.json')['config']
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    provenance = []
    for run in runs:
        checks = {}
        for name in ('model.py', 'discovery.py', 'run.py', 'tasks.py', 'world.py'):
            old = run / 'source/experiment' / name
            current = ROOT / 'experiment' / name
            # Archived copies use Windows line endings; normalize newlines only.
            checks[name] = hashlib.sha256(old.read_text(encoding='utf-8').encode()).hexdigest() == hashlib.sha256(current.read_text(encoding='utf-8').encode()).hexdigest()
        assert all(checks.values()), f'Archived code mismatch: {run.name}'
        provenance.append(dict(run=run.name, source_files_match=checks))
    result = dict(scope='post_hoc_diagnostic_not_new_method_evaluation', provenance=provenance,
                  attribution=[], training=[], evaluations=[])
    for run in runs:
        result['attribution'].append(audit_attribution(run))
        for arm in ('random', 'targeted', 'random_cf', 'targeted_cf'):
            row = audit_training(run, arm, tokenizer)
            result['training'].append(row)
            print(f"seed={row['seed']} {arm}: {row['unique_actions_seen']}/{row['action_rows']} actions; "
                  f"critical CF pairs={row['cf_pairs_both_critical_seen']}/{row['cf_pairs']}", flush=True)
        for arm in ('base', 'random', 'targeted', 'random_cf', 'targeted_cf'):
            for split in ('test', 'ood', 'composition'):
                result['evaluations'].append(dict(seed=read(run/'manifest.json')['config']['seed'],
                    arm=arm, split=split, **audit_episodes(load_rows(run/f'evaluations/{arm}_{split}.jsonl'))))
    dump(args.output / 'audit.json', result)
    lines = ['# 四组 study：归因与训练覆盖审计', '',
             '这是对已保存运行的事后诊断，不是新增训练或方法效果验证。原实验文件未改动。', '',
             '## 核验方式', '',
             '- 仅统一换行符后，核对三次运行归档的 model/discovery/run/tasks/world 与当前源码内容一致。',
             '- 用原始样本顺序和 Python RNG 重建实际抽样；逐组核对实际输入/监督 token 总量与保存记录相等。',
             '- 对140道题重放规则示范，逐条匹配保存的训练上下文与正确动作。',
             '- 在相同发现轨迹上重算 TF-IDF 聚类，验证预算是否变化。', '',
             '## 归因实现', '',
             '三次均使用Qwen embedding。当前归因是首次可观察错误提取及预设题型计数；没有按反事实对比较行为，也没有追加干预。',
             '聚类标签不进入预算公式。预算由题型失败率和错误签名多样性决定。',
             '记录名称 frozen_base_qwen_mean_pooling 容易引起误解：本次第一轮调用的是共同预热学生的 embed；该名称不证明使用的是未经预热的纯基座。', '',
             '| seed | Qwen簇数 | TF-IDF簇数 | 预算相同 |', '|---|---:|---:|---|']
    for a in result['attribution']:
        lines.append(f"| {a['seed']} | {a['qwen_clusters']} | {a['tfidf_clusters']} | {a['qwen_vs_tfidf_weights_identical']} |")
    lines += ['', '## 实际训练覆盖', '',
              '每组120道主要题+20道回放题；拆成动作示范后训练。80次更新×4条/更新=320条；rounds=1是一次诊断—训练循环，不是一个epoch。',
              '关键两侧覆盖：两条标准轨迹第一次出现不同正确动作时，两侧动作样本是否都被抽中。它只表示看过示范，不表示已学会。', '',
              '| seed | 组 | 实际动作/全部动作 | 等效epoch | 全轨迹覆盖题数/140 | CF关键两侧/60 | CF全部动作/60 |',
              '|---|---|---|---:|---:|---|---|']
    for r in result['training']:
        cp = f"{r['cf_pairs_both_critical_seen']}/60" if r['cf_pairs'] else '不适用'
        fp = f"{r['cf_pairs_all_actions_seen']}/60" if r['cf_pairs'] else '不适用'
        lines.append(f"| {r['seed']} | {r['arm']} | {r['unique_actions_seen']}/{r['action_rows']} | "
                     f"{r['effective_action_epochs']:.3f} | {r['complete_task_trajectories_seen']} | {cp} | {fp} |")
    lines += ['', '## 下一轮的判定顺序', '',
              '1. 先做固定正确查询上下文的退款因素探针，隔离流程失败。用同一订单/措辞交叉used和6/7/8天，并设价格变化控制；这只能支持特定干预下的行为证据，不能证明内部能力因果。',
              '2. 只有当控制有效，才把干预产生的差异作为能力缺口候选，加入未知/不确定类别；不得直接把原题型当作能力归因。',
              '3. 基于发现/开发任务设计新数据，测试集不参与预算。保持四组训练资源可比，记录完整epoch、实际动作及监督token；训练覆盖和归因方法分开消融。',
              '4. 小实验若稳定改善开发集且不过度拒绝，再进行新的多seed独立测试；本轮原测试已经用于诊断，不能再当盲测。', '',
              '详细逐题错误、退款条件分层、关键样本覆盖与预算见 audit.json。']
    (args.output / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(args.output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
