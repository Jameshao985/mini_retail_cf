"""Aggregate completed, compatible multi-seed experiments; never creates model scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def aggregate(directories):
    expected, expected_source, seen_seeds = None, None, set()
    observations = {}
    for folder in directories:
        folder = Path(folder)
        if not (folder / 'COMPLETE.json').is_file():
            raise ValueError(f'Incomplete experiment: {folder}')
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        config = dict(manifest['config'])
        seed = config.pop('seed')
        if seed in seen_seeds:
            raise ValueError(f'Duplicate seed {seed}; repeats must use independent seeds')
        seen_seeds.add(seed)
        if expected is None:
            expected, expected_source = config, manifest['source_hash']
        if config != expected or manifest['source_hash'] != expected_source:
            raise ValueError('Different configurations/source versions cannot be pooled')
        run_values = {}
        with (folder / 'results.csv').open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                for metric in ('success_rate', 'pair_both_correct', 'allowed_task_success',
                               'forbidden_action_rate', 'protocol_error_rate'):
                    if row[metric]:
                        value = float(row[metric])
                        run_values[(row['arm'], row['split'], metric)] = value
                        observations.setdefault((row['arm'], row['split'], metric), []).append(value)
        contrasts = (('targeted', 'random'), ('random_cf', 'random'),
                     ('targeted_cf', 'targeted'), ('targeted_cf', 'random_cf'))
        for treatment, control in contrasts:
            for split in ('test', 'ood', 'invariance', 'composition'):
                for metric in ('success_rate', 'pair_both_correct', 'forbidden_action_rate'):
                    t_key, c_key = (treatment, split, metric), (control, split, metric)
                    if t_key in run_values and c_key in run_values:
                        name = f'delta_{treatment}_vs_{control}'
                        observations.setdefault((name, split, metric), []).append(
                            run_values[t_key] - run_values[c_key])
    if len(seen_seeds) < 2:
        raise ValueError('At least two independent completed seeds are required')
    rows = []
    for (arm, split, metric), values in sorted(observations.items()):
        if len(values) != len(seen_seeds):
            raise ValueError('Missing metric in one seed')
        rows.append(dict(arm=arm, split=split, metric=metric, seeds=len(values),
                         mean=statistics.mean(values), sample_std=statistics.stdev(values)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    rows = aggregate(args.runs)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}; choose a new output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(args.output)


if __name__ == '__main__':
    main()
