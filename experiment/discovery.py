"""Evidence extraction, vector clustering and deterministic synthesis allocation."""
from __future__ import annotations

from collections import Counter
import random

import numpy as np

from .tasks import CAPABILITIES, make_pair, validate_pair
from .world import teacher_examples

EXPLANATIONS = {
    'protocol_error': '输出未满足动作JSON协议',
    'missing_clarification': '缺少必要订单号时猜测目标，没有先追问',
    'unnecessary_clarification': '已知订单号仍反复追问',
    'stale_target': '用户更新目标后仍使用旧目标',
    'wrong_target': '动作参数没有绑定当前用户指定目标',
    'wrong_operation': '使用了与用户请求不一致的业务操作',
    'unknown_order': '使用不存在的订单号',
    'missing_state_check': '尚未查询同一目标状态即执行操作或结束',
    'forbidden_action': '已知或应查明条件禁止操作仍尝试写入',
    'premature_response': '尚未完成必要交互就提前结束',
    'false_success': '没有成功执行写操作却声明成功',
    'over_refusal': '条件允许操作却拒绝执行',
    'step_limit': '重复或无进展动作导致步数耗尽',
    'wrong_terminal_state': '任务结束后目标状态错误或其他订单被修改',
    'wrong_response': '最终报告与任务目标及环境结果不一致',
}


def extract_failures(episodes):
    failures = []
    for ep in episodes:
        if ep['task_success']:
            continue
        early = next((s for s in ep['steps'] if s['violations']), None)
        if early:
            priority = ('protocol_error', 'missing_clarification', 'unnecessary_clarification',
                        'stale_target', 'wrong_target', 'wrong_operation', 'unknown_order',
                        'missing_state_check', 'forbidden_action', 'premature_response',
                        'false_success', 'over_refusal')
            code = next((p for p in priority if p in early['violations']), early['violations'][0])
        else:
            early = ep['steps'][-1]
            code = 'step_limit' if 'step_limit' in ep['violations'] else ep['violations'][0]
        expected = early['expected']
        actual = early['action'] or {}
        prior_observations = [s['observation'] for s in ep['steps']
                              if s['step'] < early['step'] and isinstance(s['observation'], dict)]
        observation = prior_observations[-1] if prior_observations else {}
        visible = {k: observation[k] for k in ('status', 'used', 'delivered_days', 'error') if k in observation}
        known = early['phase'] != 'missing_information'
        # Omit IDs, items, prices and generator-family labels from embeddings.
        text = (f'阶段={early["phase"]}；目标是否已知={known}；可见状态={visible}；'
                f'错误={EXPLANATIONS[code]}；应采取={expected["action"]}；'
                f'实际={actual.get("action", "invalid")}。')
        signature = f'{early["phase"]}|{code}|{expected["action"]}|{actual.get("action", "invalid")}'
        failures.append(dict(task_id=ep['task_id'], capability=ep['capability'], step=early['step'],
                             code=code, signature=signature, description=text, evidence=dict(expected=expected, actual=actual,
                             visible_before_action=visible, action_observation=early['observation'], raw=early['raw']),
                             diagnosis_status='observable_error_not_causal_attribution'))
    return failures


def cluster_failures(episodes, embedder=None, threshold=0.25):
    failures = extract_failures(episodes)
    if not failures:
        return dict(failures=[], clusters=[], exposure=dict(Counter(e['capability'] for e in episodes)),
                    weights={c: 1/len(CAPABILITIES) for c in CAPABILITIES}, embedding='none_no_failures')
    texts = [f['description'] for f in failures]
    if embedder is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectors = TfidfVectorizer(analyzer='char', ngram_range=(2, 4)).fit_transform(texts).toarray()
        backend = 'tfidf_character_vectors'
    else:
        vectors = np.asarray(embedder(texts))
        backend = 'frozen_base_qwen_mean_pooling'
    # Observable error structure prevents unrelated steps from collapsing into one
    # semantic cluster. The generator-family label is deliberately excluded.
    signatures = sorted(set(f['signature'] for f in failures))
    one_hot = np.zeros((len(failures), len(signatures)), dtype=float)
    lookup = {s: i for i, s in enumerate(signatures)}
    for row, failure in enumerate(failures):
        one_hot[row, lookup[failure['signature']]] = 2.0
    vectors = np.concatenate([np.asarray(vectors), one_hot], axis=1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    if len(texts) == 1:
        labels = np.array([0])
    else:
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                         metric='cosine', linkage='average').fit_predict(vectors)
    clusters = []
    exposure = Counter(e['capability'] for e in episodes)
    scores = Counter({c: 0.0 for c in CAPABILITIES})
    for label in sorted(set(labels.tolist())):
        members = [f for f, k in zip(failures, labels) if k == label]
        counts = Counter(f['capability'] for f in members)
        codes = Counter(f['code'] for f in members)
        cluster_id = f'C{int(label)+1:03d}'
        for f in members:
            f['cluster_id'] = cluster_id
        clusters.append(dict(cluster_id=cluster_id, count=len(members), title=EXPLANATIONS[codes.most_common(1)[0][0]],
                             error_codes=dict(codes), capability_counts=dict(counts),
                             representative=members[0], source_task_ids=[f['task_id'] for f in members]))
    failure_counts = Counter(f['capability'] for f in failures)
    pattern_counts = {c: len({f['signature'] for f in failures if f['capability'] == c})
                      for c in CAPABILITIES}
    failure_rates = {c: failure_counts[c] / max(1, exposure[c]) for c in CAPABILITIES}
    for c in CAPABILITIES:
        # Error rate is the main signal; a small diversity bonus stops many exact
        # duplicate failures from monopolising the budget.
        scores[c] = failure_rates[c] * (1 + 0.1 * max(0, pattern_counts[c] - 1))
    denom = sum(scores.values())
    if denom:
        # Keep 10% uniform exploration so an unobserved family is never removed.
        weights = {c: 0.1 / len(CAPABILITIES) + 0.9 * scores[c] / denom for c in CAPABILITIES}
    else:
        weights = {c: 1 / len(CAPABILITIES) for c in CAPABILITIES}
    return dict(failures=failures, clusters=clusters, exposure=dict(exposure), weights=weights,
                capability_failure_counts=dict(failure_counts), failure_rates=failure_rates,
                observable_pattern_counts=pattern_counts, allocation_scores=dict(scores),
                embedding=backend, distance_threshold=threshold,
                note='Clusters use observable step/error structure plus embeddings. Fixed task taxonomy routes clusters to generators; this is not open-ended capability discovery.')


def allocate(total, weights):
    raw = {c: total * weights[c] / sum(weights.values()) for c in CAPABILITIES}
    result = {c: int(raw[c]) for c in CAPABILITIES}
    remainder = total - sum(result.values())
    for c in sorted(CAPABILITIES, key=lambda c: raw[c] - result[c], reverse=True)[:remainder]:
        result[c] += 1
    return result


def synthesize(arm, pair_budget, analysis, seed, namespace, teacher=None):
    if arm not in ('random', 'targeted', 'random_cf', 'targeted_cf'):
        raise ValueError('Unknown arm')
    weights = (analysis['weights'] if arm.startswith('targeted')
               else {c: 1/len(CAPABILITIES) for c in CAPABILITIES})
    budget = allocate(pair_budget, weights)
    tasks = []
    sources = {c: [f['task_id'] for f in analysis['failures'] if f['capability'] == c]
               if arm.startswith('targeted') else [] for c in CAPABILITIES}
    for c, n in budget.items():
        for i in range(n):
            if arm.endswith('_cf'):
                pair = make_pair('train', c, i, seed, namespace)
                validate_pair(*pair)
                tasks.extend(pair)
            else:
                # Same side balance and count; independent nuisance contexts. Two singles != a pair.
                for side in (0, 1):
                    # For interactive families, both independent tasks in a unit
                    # use the same operation, matching the paired arm's cancel/
                    # refund mix without sharing the surrounding context.
                    pair_index = i * 4 + side * 2 + (i % 2)
                    t = make_pair('train', c, pair_index, seed, namespace + '/independent')[side]
                    t['pair_id'] = None
                    tasks.append(t)
    rows = []
    receipts = []
    units = [tasks[i:i+2] for i in range(0, len(tasks), 2)]
    for unit in units:
        if teacher:
            records, _, receipt = teacher.generate(unit, analysis['failures'], arm)
            receipts.append(receipt)
        else:
            records = [row for task in unit for row in teacher_examples(task)]
        for r in records:
            r.update(source_failure_ids=sources[r['capability']], synthesis_arm=arm)
        rows.extend(records)
    random.Random(seed).shuffle(rows)
    return tasks, rows, dict(arm=arm, tasks=len(tasks), next_action_examples=len(rows), pair_units=pair_budget,
                             allocation=budget, weights=weights, teacher=teacher.model if teacher else 'rule_oracle',
                             verified_tasks=len(tasks), external_api_calls=sum(not r['cached'] for r in receipts),
                             teacher_receipts=receipts)
