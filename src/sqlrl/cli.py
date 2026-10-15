"""Offline benchmarks, adversarial probes, curriculum audit and RL diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .grpo import describe_groups, group_relative_advantages
from .policy import BaselinePolicy
from .reward import score_sql
from .robust import score_sql_robust
from .tasks import load_tasks


def _write_json(dest: str, payload: dict) -> None:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def dataset_hash(path: str) -> str:
    """Exact dataset bytes, for matching comparison reports (no secret in model prompt)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluation(data_path: str, output: str, *, use_model: bool, robust: bool) -> dict:
    if use_model:
        from .agent import run_episode
        if not (os.environ.get('AGL_OPENAI_BASE_URL') or os.environ.get('SQLRL_OPENAI_BASE_URL')):
            raise SystemExit('Set SQLRL_OPENAI_BASE_URL to a running OpenAI-compatible model server')
    policy = BaselinePolicy()
    outcomes = []
    for task in load_tasks(data_path):
        if use_model:
            sql, _, attempts = run_episode(task)
        else:
            sql, attempts = policy.complete(task), 1
        score = score_sql_robust(task, sql) if robust else score_sql(task, sql)
        outcomes.append({'task_id': task.id, 'database': task.database,
                         'sql': sql, 'attempts': attempts, 'score': score.to_dict()})
    if not outcomes:
        raise ValueError('Empty benchmark')
    dbs = sorted({v['database'] for v in outcomes})
    summary = {name: {'n': sum(o['database'] == name for o in outcomes),
                      'accuracy': sum(o['score']['correct'] for o in outcomes if o['database'] == name) /
                          sum(o['database'] == name for o in outcomes)} for name in dbs}
    payload = {
        'run_type': ('model_eval' if use_model else 'baseline_eval') + ('_robust' if robust else '_single'),
        'evaluation_protocol': 'three_seed_strict' if robust else 'one_seed',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'split_name': Path(data_path).stem,
        'dataset_fingerprint': dataset_hash(data_path),
        'model': os.environ.get('SQLRL_MODEL', 'unspecified') if use_model else 'deterministic_nonlearning',
        'n': len(outcomes),
        'accuracy': sum(o['score']['correct'] for o in outcomes)/len(outcomes),
        'mean_reward': sum(o['score']['reward'] for o in outcomes)/len(outcomes),
        'by_domain': summary, 'results': outcomes,
    }
    _write_json(output, payload)
    return payload


def smoke(data_path: str, output: str) -> dict:
    return _evaluation(data_path, output, use_model=False, robust=False)


def robust_eval(data_path: str, output: str, *, use_model: bool = False) -> dict:
    return _evaluation(data_path, output, use_model=use_model, robust=True)


def simulate_grpo(data_path: str, output: str) -> dict:
    """Contrastive groups demonstrate reward signal, WITHOUT model updates."""
    policy = BaselinePolicy()
    groups = []
    for task in load_tasks(data_path):
        candidates = [policy.complete(task), 'SELECT COUNT(*) FROM orders',
                      'SELECT COUNT(*) FROM customers', 'DROP TABLE customers']
        # The answer from the trusted oracle is used only by this offline diagnostic
        # to provide a known-positive control. NEVER send it to a policy as a hint.
        candidates[-2] = task.oracle_sql
        scored = [score_sql_robust(task, sql) for sql in candidates]
        stats = group_relative_advantages([s.reward for s in scored])
        groups.append({'task_id': task.id, 'rewards': [s.reward for s in scored],
                       'advantages': list(stats.advantages), 'zero_variance': stats.zero_variance,
                       'correct': [s.correct for s in scored]})
    diag = describe_groups([g['rewards'] for g in groups])
    payload = {'run_type': 'cpu_group_diagnostics_NO_WEIGHT_UPDATES',
               'n_groups': len(groups), 'zero_variance_groups': diag['zero_variance'],
               'group_stats': diag, 'groups': groups}
    _write_json(output, payload)
    return payload


def model_eval(data_path: str, output: str) -> dict:
    return _evaluation(data_path, output, use_model=True, robust=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'simulate-grpo', 'model-eval',
                         'robust-eval', 'robust-model-eval', 'audit-splits',
                         'curriculum', 'attack-suite', 'compare', 'signal-sweep'])
    parser.add_argument('--data', default='data/train.jsonl')
    parser.add_argument('--output', default=None)
    parser.add_argument('--baseline', help='For compare: reference JSON')
    parser.add_argument('--candidate', help='For compare: candidate JSON')
    parser.add_argument('--resamples', type=int, default=5000)
    args = parser.parse_args()
    output = args.output or f'artifacts/{args.command.replace("-", "_")}.json'
    if args.command == 'smoke':
        payload = smoke(args.data, output)
    elif args.command == 'simulate-grpo':
        payload = simulate_grpo(args.data, output)
    elif args.command == 'model-eval':
        payload = model_eval(args.data, output)
    elif args.command in ('robust-eval', 'robust-model-eval'):
        payload = robust_eval(args.data, output, use_model=args.command == 'robust-model-eval')
    elif args.command == 'audit-splits':
        from .curriculum import split_audit
        payload = split_audit({p: f'data/{p}.jsonl' for p in ('train', 'val', 'test', 'ood')})
    elif args.command == 'curriculum':
        from .curriculum import curriculum_manifest
        payload = curriculum_manifest(load_tasks(args.data))
    elif args.command == 'attack-suite':
        from .adversarial import run_attack_suite
        payload = run_attack_suite(load_tasks(args.data))
    elif args.command == 'signal-sweep':
        from .signal import signal_sweep
        payload = signal_sweep()
    else:
        from .experiment import compare_evaluations
        if not args.baseline or not args.candidate:
            parser.error('compare requires --baseline and --candidate')
        payload = compare_evaluations(args.baseline, args.candidate, resamples=args.resamples)
    _write_json(output, payload)
    compact = {k: v for k, v in payload.items() if k in (
        'n', 'accuracy', 'n_groups', 'zero_variance_groups', 'counts', 'schemas',
        'ok', 'n', 'guard_bypasses', 'promotion_gate_passed', 'ci_95', 'absolute_accuracy_gain')}
    print(json.dumps(compact, indent=2))
    print(f'Artifact: {output}')


if __name__ == '__main__':
    main()
