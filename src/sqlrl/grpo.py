"""CPU-only GRPO reward/advantage DIAGNOSTICS; not a policy gradient trainer.

The real gradient updates occur only through Agent Lightning + verl on GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupStats:
    advantages: tuple[float, ...]
    reward_mean: float
    reward_std: float
    zero_variance: bool


def group_relative_advantages(rewards: list[float], epsilon: float = 1e-6) -> GroupStats:
    if len(rewards) < 2:
        raise ValueError("GRPO requires grouped samples n >= 2")
    if not all(math.isfinite(x) for x in rewards):
        raise ValueError("Rewards must be finite")
    mean = sum(rewards) / len(rewards)
    variance = sum((x - mean) ** 2 for x in rewards) / len(rewards)
    std = math.sqrt(variance)
    # Zero-variance groups contain no ranking signal and produce zero advantages.
    return GroupStats(
        tuple(0.0 if std <= epsilon else (x - mean) / (std + epsilon) for x in rewards),
        mean,
        std,
        std <= epsilon,
    )


def describe_groups(groups: list[list[float]]) -> dict:
    """Detect groups where no relative policy-learning signal is present."""
    if not groups:
        raise ValueError('At least one group required')
    results = [group_relative_advantages(g) for g in groups]
    return {
        'total_groups': len(results),
        'zero_variance': sum(g.zero_variance for g in results),
        'zero_variance_rate': sum(g.zero_variance for g in results) / len(results),
        'mean_group_reward': sum(g.reward_mean for g in results)/len(results),
        'mean_group_std': sum(g.reward_std for g in results)/len(results),
        'warning': 'No policy update in this CPU diagnostics module',
    }
