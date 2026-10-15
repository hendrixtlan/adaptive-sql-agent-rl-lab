"""When does a group of rollouts carry no GRPO learning signal?

A diagnostic Bernoulli correctness model, NOT a simulation of neural training.
Zero-variance probability is p**k + (1-p)**k for binary outcome rewards.
"""
from __future__ import annotations

import random

from .grpo import group_relative_advantages


def signal_sweep(*, probabilities: tuple[float, ...] = (.05, .1, .25, .5, .75, .9, .95),
                 group_sizes: tuple[int, ...] = (2, 4, 8, 16),
                 n_groups: int = 2500, seed: int = 2026) -> dict:
    if n_groups < 1 or any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid probability or group count')
    rng = random.Random(seed)
    results = []
    for p in probabilities:
        for n in group_sizes:
            if n < 2:
                raise ValueError('Group size must be >=2')
            n_flat = 0
            for _ in range(n_groups):
                rewards = [1.0 if rng.random() < p else -1.0 for _ in range(n)]
                n_flat += group_relative_advantages(rewards).zero_variance
            results.append({
                'correctness_probability': p,
                'group_size': n,
                'observed_zero_variance_rate': round(n_flat/n_groups, 5),
                'theoretical_zero_variance_rate': round(p**n + (1-p)**n, 5),
            })
    return {
        'run_type': 'synthetic_binary_reward_signal_DIAGNOSTIC_NO_TRAINING',
        'n_groups_per_case': n_groups, 'seed': seed, 'results': results,
        'note': 'A Bernoulli simplification; real model rollouts and multi-valued rewards behave differently.'
    }
