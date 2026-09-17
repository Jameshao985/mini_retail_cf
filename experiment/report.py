from __future__ import annotations

from collections import Counter, defaultdict
import csv
import random
from pathlib import Path

from .io import dump
from .tasks import CAPABILITIES

ARM_NAMES = {
    'base': '共同预热起点（base）',
    'random': '普通合成（random）',
    'targeted': '定向普通合成（targeted）',
    'random_cf': '普通反事实合成（random_cf）',
    'targeted_cf': '定向反事实合成（targeted_cf）',
}


def mean(values):
    return sum(values) / len(values) if values else None


def metrics(episodes):
    pairs = defaultdict(list)
    for e in episodes:
        if e['pair_id']:
            pairs[e['pair_id']].append(e)
    complete = [p for p in pairs.values() if len(p) == 2]
    if len(complete) != len(pairs):
        raise ValueError('Evaluation includes an incomplete or duplicated pair')
    allowed = [e for e in episodes if e['gold_outcome'] in ('cancelled', 'refunded')]
    forbidden = [e for e in episodes if e['gold_outcome'] == 'refused']
    return dict(tasks=len(episodes), successes=sum(e['task_success'] for e in episodes),
                success_rate=mean([e['task_success'] for e in episodes]), pairs=len(complete),
                pair_both_correct=mean([all(e['task_success'] for e in p) for p in complete]),
                forbidden_action_rate=mean(['forbidden_action' in e['violations'] for e in episodes]),
                forbidden_attempt_given_forbidden=mean(['forbidden_action' in e['violations'] for e in forbidden]),
                allowed_task_success=mean([e['task_success'] for e in allowed]),
                over_refusal_rate=mean(['over_refusal' in e['violations'] for e in allowed]),
                protocol_error_rate=mean(['protocol_error' in e['violations'] for e in episodes]),
                by_capability={c: mean([e['task_success'] for e in episodes if e['capability'] == c]) for c in CAPABILITIES},
                failure_counts=dict(Counter(v for e in episodes for v in e['violations'])))


def paired_interval(base, trained, seed=42, repeats=2000):
    """Bootstrap task groups (not individual counterfactual sides)."""
    b, t = {e['task_id']: e for e in base}, {e['task_id']: e for e in trained}
    if b.keys() != t.keys():
        raise ValueError('Cannot compare different evaluation tasks')
    groups = defaultdict(list)
    for k in b:
        groups[b[k]['pair_id'] or b[k]['task']['group_id']].append(k)
    groups = list(groups.values())
    if not groups:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        sampled = [k for g in rng.choices(groups, k=len(groups)) for k in g]
        values.append(sum(int(t[k]['task_success']) - int(b[k]['task_success']) for k in sampled) / len(sampled))
    values.sort()
    return {'delta': sum(int(t[k]['task_success']) - int(b[k]['task_success']) for k in b)/len(b),
            'ci95_low': values[int(repeats*.025)], 'ci95_high': values[int(repeats*.975)],
            'resampling_unit': 'pair_group', 'repeats': repeats}


def write_report(root, config, evaluations, training):
    root = Path(root)
    records, comparisons = [], {}
    for arm, splits in evaluations.items():
        for split, episodes in splits.items():
            m = metrics(episodes)
            dump(root / 'metrics' / f'{arm}_{split}.json', m)
            records.append(dict(arm=arm, split=split, tasks=m['tasks'], success_rate=m['success_rate'],
                                pair_both_correct=m['pair_both_correct'], allowed_task_success=m['allowed_task_success'],
                                forbidden_action_rate=m['forbidden_action_rate'],
                                protocol_error_rate=m['protocol_error_rate'], **m['by_capability']))
            if arm != 'base':
                comparisons[f'{arm}_vs_base/{split}'] = paired_interval(evaluations['base'][split], episodes, config['seed'])
    factorial = (('targeted', 'random'), ('random_cf', 'random'),
                 ('targeted_cf', 'targeted'), ('targeted_cf', 'random_cf'))
    for treatment, control in factorial:
        if treatment in evaluations and control in evaluations:
            for split in evaluations[control]:
                comparisons[f'{treatment}_vs_{control}/{split}'] = paired_interval(
                    evaluations[control][split], evaluations[treatment][split], config['seed'])
    dump(root / 'comparisons.json', comparisons)
    with (root / 'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    pct = lambda x: 'N/A' if x is None else f'{100*x:.1f}%'
    teacher_description = ('DeepSeek '+config['teacher_model']+'，经环境回放后用于主要合成数据；共享回放仍为规则教师。'
                           if config.get('teacher') == 'deepseek' else '确定性规则程序；无外部 API。')
    lines = ['# 本地实验实测报告', '',
             f'- 配置：{config["profile"]}；训练seed={config["seed"]}；固定任务seed={config["task_seed"]}；轮数={config["rounds"]}',
             '- Student：本地 Qwen3-0.6B；教师：'+teacher_description,
             f'- base表示完成所有组共享的协议预热后、尚未接受组别训练的共同起点；预热任务={config["warmup_tasks"]}，更新={config["warmup_steps"]}。',
             '- 所有数值来自实际模型逐步生成、环境执行与程序验证；没有用教师替代 Student。',
             '- 这是有限模板的零售微环境。能力类别和生成器为预设；聚类用于聚合观察错误与分配预算。', '',
             '| 模型/训练组 | 测试集 | 任务数 | 完整成功率 | 两侧均正确 | 允许任务成功率 | 违规尝试率 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {ARM_NAMES.get(r["arm"], r["arm"])} | {r["split"]} | {r["tasks"]} | {pct(r["success_rate"])} | {pct(r["pair_both_correct"])} | {pct(r["allowed_task_success"])} | {pct(r["forbidden_action_rate"])} |')
    lines += ['', '## 与原模型相比的成功率变化', '', '95% 区间按完整样本对重采样；不包含不同训练随机种子的方差。', '']
    for key, value in comparisons.items():
        lines.append(f'- {key}：{value["delta"]*100:+.1f} 个百分点，区间 [{value["ci95_low"]*100:+.1f}, {value["ci95_high"]*100:+.1f}]。')
    lines += ['', '## 训练成本', '', '| 组/轮 | 更新数 | 训练动作样本 | 实际输入token | 实际监督token | 训练秒数 |', '|---|---:|---:|---:|---:|---:|']
    for name, s in training.items():
        lines.append(f'| {name} | {s["optimizer_updates"]} | {s["examples"]} | {s["seen_input_tokens"]} | {s["seen_supervised_tokens"]} | {s["elapsed_seconds"]:.1f} |')
    lines += ['', '各组控制合成任务数、LoRA配置和更新数；序列长度不同，实际token并非严格相等，见上表。',
              '普通合成同样覆盖各类正反条件并使用相同规则教师；共享回放使用独立普通样本，不含反事实对。',
              '', '## 如何判断是否值得继续', '',
              '- smoke只验证工程链路；训练步数和测试样本太少，不支持方法有效性的结论。',
              '- 优先比较targeted_cf与targeted，不能仅用微调模型胜过base来证明反事实策略有效。',
              '- 若普通SFT和反事实SFT都接近满分，说明当前任务太简单，应扩大模板和规则变化。',
              '- 若提升主要来自JSON协议错误减少，应先补充等量协议预热，再检验决策能力差异。',
              '- ood只是同一零售工具环境内的新措辞、商品与规则组合，不是跨领域迁移。',
              '- invariance两侧只改无关价格；以两侧均正确衡量，不奖励“两侧都错但一致”。',
              '- composition让另一条退货条件先失效，两侧都应拒绝，用于检查模型是否同时执行多条规则。',
              '- 正式结果需要更多任务、至少3个训练seed和独立外部任务。',
              '', '## 失败案例与产物', '',
              '原始记录见 evaluations/*.jsonl；聚类、代表错误及预算见各轮analysis.json；',
              '训练数据见各轮sft.jsonl；适配器见各轮train/adapter；源配置及代码指纹见manifest.json。']
    ledger_path = root / 'teacher_cache' / 'ledger.json'
    if ledger_path.is_file():
        from .io import read
        attempts = read(ledger_path)['attempts']
        lines += ['', '## 外部教师实际调用', '',
                  f'- 已记录尝试次数：{len(attempts)}；通过回放：{sum(a["status"] == "accepted" for a in attempts)}。',
                  f'- API报告prompt tokens：{sum(a.get("usage", {}).get("prompt_tokens", 0) for a in attempts)}。',
                  f'- API报告completion tokens：{sum(a.get("usage", {}).get("completion_tokens", 0) for a in attempts)}。',
                  '- 失败请求可能计费但没有usage，以上token可能不含失败请求。详情见teacher_cache/ledger.json。',
                  '- 任务结构、关键变量和用户措辞仍由程序生成；强教师在本版本中生成动作轨迹，没有自由改写任务。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Small standalone SVG uses only local measured values (no plotting dependency).
    test_rows = [r for r in records if r['split'] == 'test']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="320" viewBox="0 0 800 320">',
           '<rect width="800" height="320" fill="#f5f7fb"/>',
           '<text x="24" y="30" font-size="18" font-family="sans-serif">Measured task success / pair both correct</text>']
    for i, r in enumerate(test_rows):
        y = 58 + i * 47
        svg += [f'<text x="24" y="{y+17}" font-family="sans-serif" font-size="13">{r["arm"]}</text>',
                f'<rect x="155" y="{y}" width="{450*r["success_rate"]}" height="16" fill="#167d9a"/>',
                f'<rect x="155" y="{y+18}" width="{450*(r["pair_both_correct"] or 0)}" height="10" fill="#edaa42"/>',
                f'<text x="620" y="{y+18}" font-size="13" font-family="sans-serif">{pct(r["success_rate"])} / {pct(r["pair_both_correct"])}</text>']
    svg.append('</svg>')
    (root / 'results.svg').write_text('\n'.join(svg), encoding='utf-8')
    return records
