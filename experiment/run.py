"""python -m experiment.run --profile smoke|pilot [--run-dir ...]"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import os
from pathlib import Path
import platform
import time

# Student inference/training remain offline; only the explicitly selected teacher uses an API.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .io import digest, dump, jsonl, read
from .tasks import (CAPABILITIES, assert_disjoint, make_composition, make_invariance,
                    make_protocol_tasks, make_split)
from .world import rollout, teacher_examples

PROFILES = {
    'smoke': dict(discovery_pairs=1, dev_pairs=1, test_pairs=1, ood_pairs=1,
                  invariance_pairs=1, train_pairs=10, replay_pairs=1, train_steps=3, grad_accum=2,
                  warmup_tasks=3, warmup_steps=2, composition_pairs=1),
    'pilot': dict(discovery_pairs=6, dev_pairs=3, test_pairs=8, ood_pairs=4,
                  invariance_pairs=2, train_pairs=60, replay_pairs=2, train_steps=60, grad_accum=4,
                  warmup_tasks=12, warmup_steps=12, composition_pairs=4),
    # 300 fixed main-test tasks: 30 pairs x 5 generator families x 2 sides.
    'study': dict(discovery_pairs=12, dev_pairs=6, test_pairs=30, ood_pairs=10,
                  invariance_pairs=4, train_pairs=60, replay_pairs=2, train_steps=80, grad_accum=4,
                  warmup_tasks=20, warmup_steps=20, composition_pairs=10),
}
ARMS = ('random', 'targeted', 'targeted_cf', 'random_cf')


def load_model_path(explicit=None):
    # Independent from legacy config.py, which may contain a user's API credential.
    base = Path(explicit or os.environ.get('MODEL_PATH') or r'D:\hf_cache\hub\models--Qwen--Qwen3-0.6B').expanduser()
    candidates = [base]
    if (base / 'snapshots').is_dir():
        candidates += sorted((base / 'snapshots').iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if (p / 'config.json').is_file() and (p / 'tokenizer_config.json').is_file() and list(p.glob('*.safetensors')):
            index = p / 'model.safetensors.index.json'
            if index.is_file() and any(not (p / name).is_file() for name in set(read(index)['weight_map'].values())):
                continue
            if read(p / 'config.json').get('model_type') != 'qwen3':
                raise ValueError(f'Expected Qwen3, found another architecture: {p}')
            return p.resolve()
    raise FileNotFoundError(f'No complete offline Qwen3 snapshot in {base}; pass --model-path.')


def doctor(path, device):
    import torch
    import peft  # Fail here rather than after expensive discovery rollouts.
    import sklearn
    info = dict(python=platform.python_version(), model_path=str(path), requested_device=device,
                packages={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'accelerate', 'scikit-learn')},
                cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_total_gb=torch.cuda.get_device_properties(0).total_memory/2**30 if torch.cuda.is_available() else None)
    print(info, flush=True)
    return info


def evaluate(tasks, student, output):
    """Atomically checkpoint after each completed episode; resumable within a split."""
    output = Path(output)
    existing = []
    if output.is_file():
        import json
        existing = [json.loads(s) for s in output.read_text(encoding='utf-8').splitlines() if s]
    by_id = {e['task_id']: e for e in existing}
    if len(by_id) != len(existing) or not set(by_id).issubset({t['task_id'] for t in tasks}):
        raise ValueError('Duplicate or mismatched evaluation cache')
    for t in tasks:
        if t['task_id'] in by_id and digest(by_id[t['task_id']]['task']) != digest(t):
            raise ValueError('Task changed since cached rollout')
    start = time.monotonic()
    for i, task in enumerate(tasks):
        if task['task_id'] not in by_id:
            by_id[task['task_id']] = rollout(task, student)
            jsonl(output, [by_id[t['task_id']] for t in tasks if t['task_id'] in by_id])
        if (i + 1) % 6 == 0 or i + 1 == len(tasks):
            print(f'  {output.stem}: {i+1}/{len(tasks)} episodes ({time.monotonic()-start:.0f}s)', flush=True)
    return [by_id[t['task_id']] for t in tasks]


def load_episodes(path):
    import json
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]


def prepare(root, cfg):
    splits = {name: make_split(name, cfg[name + '_pairs'], cfg['task_seed'])
              for name in ('discovery', 'dev', 'test', 'ood')}
    splits['invariance'] = make_invariance(cfg['invariance_pairs'], cfg['task_seed'])
    splits['composition'] = make_composition(cfg['composition_pairs'], cfg['task_seed'])
    warmup = make_protocol_tasks(cfg['warmup_tasks'], cfg['task_seed'])
    assert_disjoint(warmup, *splits.values())
    for name, tasks in splits.items():
        for t in tasks:
            if not rollout(t, None)['task_success']:
                raise ValueError('Oracle control failed')
        dump(root / 'tasks' / f'{name}.json', tasks)
    for task in warmup:
        if not rollout(task, None)['task_success']:
            raise ValueError('Warmup oracle control failed')
    dump(root / 'tasks' / 'common_warmup.json', warmup)
    return splits, warmup


def pipeline(root, cfg, stage):
    from .discovery import cluster_failures, synthesize
    from .model import Student, train_lora
    from .report import write_report, metrics
    splits, warmup_tasks = prepare(root, cfg)
    if stage == 'prepare':
        return
    warmup_adapter = root / 'common_warmup' / 'train' / 'adapter'
    if stage in ('all', 'baseline', 'train') and not (root / 'common_warmup' / 'train' / 'training_summary.json').is_file():
        print('Stage: common protocol warmup', flush=True)
        warmup_rows = [row for task in warmup_tasks for row in teacher_examples(task)]
        jsonl(root / 'common_warmup' / 'sft.jsonl', warmup_rows)
        train_lora(cfg['model_path'], warmup_rows, root / 'common_warmup' / 'train',
                   dict(cfg, train_steps=cfg['warmup_steps']))
    if not warmup_adapter.is_dir():
        raise RuntimeError('Missing common warmup adapter; run --stage baseline or --stage all first.')
    if stage in ('all', 'baseline'):
        print('Stage: warmed common-start student failure discovery', flush=True)
        student = Student(cfg['model_path'], cfg['device'], cfg['seed'], cfg['max_new_tokens'], warmup_adapter)
        try:
            episodes = evaluate(splits['discovery'], student, root / 'baseline' / 'discovery.jsonl')
            dev = evaluate(splits['dev'], student, root / 'baseline' / 'dev.jsonl')
            dump(root / 'baseline' / 'dev_metrics.json', metrics(dev))
            if not (root / 'baseline' / 'analysis.json').is_file():
                analysis = cluster_failures(episodes, student.embed if cfg['embedding'] == 'qwen' else None, cfg['cluster_threshold'])
                dump(root / 'baseline' / 'analysis.json', analysis)
        finally:
            student.close()
    if stage == 'baseline':
        return
    if not (root / 'baseline' / 'analysis.json').is_file():
        raise RuntimeError('Run --stage baseline first, or use --stage all.')
    if stage in ('all', 'train'):
        teacher = None
        if cfg.get('teacher', 'rule') == 'deepseek':
            from .teacher import DeepSeekTeacher
            teacher = DeepSeekTeacher(root / 'teacher_cache', cfg['teacher_model'], cfg['teacher_max_calls'],
                                     cfg['teacher_timeout'], cfg['use_legacy_key'])
        all_train_tasks = list(warmup_tasks)
        for arm in cfg['arms']:
            previous = warmup_adapter
            analysis = read(root / 'baseline' / 'analysis.json')
            for round_id in range(1, cfg['rounds'] + 1):
                folder = root / arm / f'round_{round_id}'
                folder.mkdir(parents=True, exist_ok=True)
                print(f'Stage: {arm}, round {round_id}/{cfg["rounds"]}', flush=True)
                if round_id > 1:
                    prev = root / arm / f'round_{round_id-1}'
                    # Re-diagnose the newly trained Student, using discovery data only.
                    if not (prev / 'next_analysis.json').is_file():
                        probe = Student(cfg['model_path'], cfg['device'], cfg['seed'], cfg['max_new_tokens'], previous)
                        try:
                            failures = evaluate(splits['discovery'], probe, prev / 'rediscovery.jsonl')
                        finally:
                            probe.close()
                        frozen = Student(cfg['model_path'], cfg['device'], cfg['seed'], cfg['max_new_tokens']) if cfg['embedding'] == 'qwen' else None
                        try:
                            analysis = cluster_failures(failures, frozen.embed if frozen else None, cfg['cluster_threshold'])
                            dump(prev / 'next_analysis.json', analysis)
                        finally:
                            if frozen:
                                frozen.close()
                    analysis = read(prev / 'next_analysis.json')
                dump(folder / 'analysis.json', analysis)
                tasks, rows, synthesis = synthesize(arm, cfg['train_pairs'], analysis, cfg['seed'],
                                                    f'{arm}/round{round_id}', teacher)
                # The same small general-task replay set goes into all arms, generated once per round.
                replay_tasks, replay_rows, _ = synthesize('random', cfg['replay_pairs'] * len(CAPABILITIES),
                                 analysis, cfg['seed'], f'shared_replay/round{round_id}')
                assert_disjoint(tasks, replay_tasks, *splits.values())
                if all_train_tasks:
                    assert_disjoint(tasks, all_train_tasks)
                all_train_tasks += tasks
                dump(folder / 'synthesis.json', dict(synthesis, replay_tasks=len(replay_tasks)))
                dump(folder / 'tasks.json', tasks + replay_tasks)
                jsonl(folder / 'sft.jsonl', rows + replay_rows)
                if not (folder / 'train' / 'training_summary.json').is_file():
                    train_lora(cfg['model_path'], rows + replay_rows, folder / 'train', cfg, previous)
                previous = folder / 'train' / 'adapter'
                student = Student(cfg['model_path'], cfg['device'], cfg['seed'], cfg['max_new_tokens'], previous)
                try:
                    dev = evaluate(splits['dev'], student, folder / 'dev.jsonl')
                    dump(folder / 'dev_metrics.json', metrics(dev))
                finally:
                    student.close()
    if stage == 'train':
        return
    # Held-out evaluation only after all training rounds, never used by synthesis or budget allocation.
    evaluations, training = {}, {}
    for arm in ['base'] + cfg['arms']:
        adapter = warmup_adapter if arm == 'base' else root / arm / f'round_{cfg["rounds"]}' / 'train' / 'adapter'
        if adapter and not adapter.is_dir():
            raise RuntimeError(f'Missing adapter {adapter}; run --stage train first.')
        print(f'Stage: final held-out evaluation, {arm}', flush=True)
        student = Student(cfg['model_path'], cfg['device'], cfg['seed'], cfg['max_new_tokens'], adapter)
        try:
            evaluations[arm] = {name: evaluate(splits[name], student, root / 'evaluations' / f'{arm}_{name}.jsonl')
                                for name in ('test', 'ood', 'invariance', 'composition')}
        finally:
            student.close()
        if arm != 'base':
            for r in range(1, cfg['rounds'] + 1):
                training[f'{arm}/round{r}'] = read(root / arm / f'round_{r}' / 'train' / 'training_summary.json')
        else:
            training['common_warmup'] = read(root / 'common_warmup' / 'train' / 'training_summary.json')
    write_report(root, cfg, evaluations, training)
    dump(root / 'COMPLETE.json', dict(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status='complete'))
    print(f'Completed. Report: {root / "REPORT.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='pilot')
    parser.add_argument('--model-path')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task-seed', type=int, default=20260914,
                        help='Fixed task/test seed; keep constant when repeating training seeds.')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--arms', nargs='+', choices=ARMS,
                        default=['random', 'targeted', 'random_cf', 'targeted_cf'])
    parser.add_argument('--embedding', choices=('qwen', 'tfidf'), default='qwen')
    parser.add_argument('--teacher', choices=('rule', 'deepseek'), default='rule')
    parser.add_argument('--teacher-model', default='deepseek-v4-pro')
    parser.add_argument('--teacher-max-calls', type=int, default=200)
    parser.add_argument('--teacher-timeout', type=int, default=90)
    parser.add_argument('--use-legacy-key', action='store_true')
    parser.add_argument('--cluster-threshold', type=float, default=0.25)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--stage', choices=('all', 'prepare', 'baseline', 'train', 'report'), default='all')
    parser.add_argument('--doctor', action='store_true')
    for name in PROFILES['study']:
        parser.add_argument('--' + name.replace('_', '-'), type=int)
    args = parser.parse_args()
    cfg = dict(PROFILES[args.profile])
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k not in ('run_dir', 'stage', 'doctor')})
    for key in ('rounds', 'max_length', 'max_new_tokens', 'lora_r', 'teacher_max_calls', 'teacher_timeout', *PROFILES['study']):
        if cfg[key] < 1:
            parser.error(f'{key} must be positive')
    if not 0 < cfg['cluster_threshold'] <= 2 or cfg['learning_rate'] <= 0:
        parser.error('Invalid cluster threshold or learning rate')
    if len(set(cfg['arms'])) != len(cfg['arms']):
        parser.error('Duplicate arms')
    if cfg['teacher'] == 'deepseek':
        needed = cfg['train_pairs'] * len(cfg['arms']) * cfg['rounds']
        if needed > cfg['teacher_max_calls']:
            parser.error(f'This run needs up to {needed} teacher calls; increase --teacher-max-calls or reduce the experiment.')
        if args.stage in ('all', 'train') and not args.doctor:
            from .teacher import get_key
            get_key(cfg['use_legacy_key'])  # Check availability before expensive student rollout.
    cfg['model_path'] = str(load_model_path(args.model_path))
    info = doctor(cfg['model_path'], cfg['device'])
    if args.doctor:
        return
    root = (args.run_dir or Path('outputs/experiments') / f'{args.profile}_{dt.datetime.now():%Y%m%d_%H%M%S}_s{args.seed}').resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_hash = digest({p.name: p.read_text(encoding='utf-8') for p in sorted(Path(__file__).parent.glob('*.py'))})
    model_files = {p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                   for p in Path(cfg['model_path']).iterdir() if p.is_file()}
    fingerprint = digest(dict(config=cfg, source_hash=source_hash, model_files=model_files, packages=info['packages']))
    manifest = root / 'manifest.json'
    if manifest.is_file():
        if read(manifest)['fingerprint'] != fingerprint:
            raise ValueError('Run configuration/code/model/dependencies changed. Choose a NEW --run-dir; old results will not be overwritten.')
    else:
        if any(root.iterdir()):
            raise ValueError('Output directory is nonempty and has no experiment manifest. Choose a new directory.')
        dump(manifest, dict(config=cfg, environment=info, fingerprint=fingerprint, source_hash=source_hash,
                            model_files=model_files, created_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        for source in Path(__file__).parent.glob('*.py'):
            target = root / 'source' / 'experiment' / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
    try:
        pipeline(root, cfg, args.stage)
    except BaseException as exc:
        dump(root / 'LAST_ERROR.json', dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
