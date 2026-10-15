"""Paired checkpoint comparison, reproducible paired bootstrap and promotion gates.

Confidence intervals characterize uncertainty across the benchmark tasks; they do
NOT validate independence or fix dataset leakage. Never promote based on train.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def _load(path: str) -> tuple[dict[str, bool], dict]:
    payload = json.loads(Path(path).read_text())
    if not payload.get('results'):
        raise ValueError(f'No evaluations in {path}')
    outcomes = {}
    safety = {}
    for item in payload['results']:
        tid = item['task_id']
        if tid in outcomes:
            raise ValueError(f'Duplicate task ID {tid} in {path}')
        outcomes[tid] = bool(item['score']['correct'])
        safety[tid] = bool(item['score'].get('safe', True))
    payload['_safe_by_task'] = safety
    return outcomes, payload


def compare_evaluations(baseline: str, candidate: str, *, resamples: int = 5000,
                        seed: int = 2026, min_gain: float = 0.02,
                        require_ci_above_zero: bool = True) -> dict:
    if not 100 <= resamples <= 100000:
        raise ValueError('resamples must be 100..100000')
    ref, ref_meta = _load(baseline)
    new, new_meta = _load(candidate)
    if set(ref) != set(new):
        raise ValueError('Paired comparison requires exactly the same task IDs')
    if not ref:
        raise ValueError('Empty comparison')
    if ref_meta.get('dataset_fingerprint') != new_meta.get('dataset_fingerprint'):
        if ref_meta.get('dataset_fingerprint') or new_meta.get('dataset_fingerprint'):
            raise ValueError('Mismatched benchmark fingerprints')
    # Do not compare a single-fixture reward with a robust multi-fixture reward.
    if ref_meta.get('evaluation_protocol') != new_meta.get('evaluation_protocol'):
        raise ValueError('Mismatched evaluation protocols')
    ids = sorted(ref)
    deltas = [int(new[tid]) - int(ref[tid]) for tid in ids]
    n = len(ids)
    gain = sum(deltas) / n
    rng = random.Random(seed)
    boot = sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples))
    lower = boot[int(.025 * (resamples - 1))]
    upper = boot[int(.975 * (resamples - 1))]
    eligible = ('test' in ref_meta.get('split_name', '').lower() or
                'ood' in ref_meta.get('split_name', '').lower()) and (
                'test' in new_meta.get('split_name', '').lower() or
                'ood' in new_meta.get('split_name', '').lower())
    baseline_unsafe = sum(not ok for ok in ref_meta['_safe_by_task'].values())
    candidate_unsafe = sum(not ok for ok in new_meta['_safe_by_task'].values())
    promoted = bool(eligible and gain >= min_gain and candidate_unsafe <= baseline_unsafe
                    and (not require_ci_above_zero or lower > 0))
    return {
        'run_type': 'paired_bootstrap_checkpoint_comparison', 'n': n,
        'baseline_accuracy': round(sum(ref.values())/n, 6),
        'candidate_accuracy': round(sum(new.values())/n, 6),
        'absolute_accuracy_gain': round(gain, 6),
        'ci_95': [round(lower, 6), round(upper, 6)],
        'resamples': resamples, 'bootstrap_seed': seed,
        'baseline_unsafe': baseline_unsafe, 'candidate_unsafe': candidate_unsafe,
        'regressions': sum(d < 0 for d in deltas),
        'improvements': sum(d > 0 for d in deltas),
        'unchanged': sum(d == 0 for d in deltas),
        'eligible_heldout_protocol': eligible,
        'promotion_gate_passed': promoted,
        'note': 'A passing gate requires separate security and operations review; it is not automatic deployment.'
    }
