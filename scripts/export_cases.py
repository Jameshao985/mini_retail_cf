"""Export matched, actually recorded before/after trajectories from a completed run."""
import argparse
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def export_cases(root):
    root = Path(root)
    if not (root / 'COMPLETE.json').is_file():
        raise ValueError('Only completed runs can be exported')
    base = {r['task_id']: r for r in read_rows(root / 'evaluations' / 'base_test.jsonl')}
    arms = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))['config']['arms']
    lines = ['# 实际轨迹案例', '', '以下案例从正式test记录中按固定顺序选取，每组最多2个改善案例和2个残留失败。',
             '这不是额外测试，也不是Teacher生成的示范；选择案例仅用于解释，组间比较以全量指标为准。', '']
    for arm in arms:
        rows = read_rows(root / 'evaluations' / f'{arm}_test.jsonl')
        buckets = [('改善案例', [r for r in rows if r['task_success'] and not base[r['task_id']]['task_success']]),
                   ('残留失败', [r for r in rows if not r['task_success']])]
        for label, selected in buckets:
            lines += [f'## {arm}：{label}', '']
            if not selected:
                lines += ['该组在此测试集中没有此类案例。', '']
            for row in selected[:2]:
                lines += [f'### {row["task_id"]}', '', f'- 用户请求：{row["task"]["request"]}',
                          f'- 任务类别：{row["capability"]}', f'- 干预变量：{row["task"]["intervention_variable"]}',
                          f'- 当前组完整成功：{row["task_success"]}',
                          f'- 当前组违规：{", ".join(row["violations"]) or "无"}', '']
                for label2, episode in [('底座', base[row['task_id']]), ('训练后', row)]:
                    lines += [f'**{label2}的实际动作：**', '', '```text']
                    for step in episode['steps']:
                        lines.append(f'{step["step"]}. {step["raw"]}')
                        if step.get('user_continuation'):
                            lines.append('   用户后续消息：' + step['user_continuation'])
                    lines += ['```', '']
    path = root / '训练前后的真实轨迹案例.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(export_cases(args.run_dir))
