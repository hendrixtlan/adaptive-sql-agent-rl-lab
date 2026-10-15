"""Version 0.4 CPU counterfactual scheduler for GRPO collection efficiency.

This module DOES NOT train model weights or implement a verl rollout hook.
Each strategy sees the same immutable (task, attempt) reward tape and may
only request the next attempt for a task; unseen future rewards are never
used for selection. Unused budget is accounted for, not silently ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import random
from typing import Sequence, Protocol

from .grpo import group_relative_advantages


@dataclass(frozen=True)
class TaskProfile:
    id: str
    family: str
    p_correct: float
    p_unsafe: float = 0.01

    def __post_init__(self) -> None:
        if not self.id or not self.family:
            raise ValueError('Task IDs and families must be nonempty')
        if not 0.0 <= self.p_correct <= 1.0 or not 0.0 <= self.p_unsafe <= 1.0:
            raise ValueError('All probabilities must lie in [0, 1]')


@dataclass(frozen=True)
class BudgetConfig:
    budget: int
    initial: int = 2
    group_size: int = 4
    max_per_task: int = 8
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.budget < 2 or self.initial < 2 or self.group_size < 2:
            raise ValueError('Budget and group sizes must be >= 2')
        if self.max_per_task < max(self.initial, self.group_size):
            raise ValueError('max_per_task too small')


@dataclass
class TaskState:
    id: str
    family: str
    rewards: list[float] = field(default_factory=list)

    @property
    def informative(self) -> bool:
        return len(self.rewards) >= 2 and not group_relative_advantages(self.rewards).zero_variance

    @property
    def flat(self) -> bool:
        return len(self.rewards) >= 2 and not self.informative


class RewardTape:
    """Independent per-task/attempt draws; same tape for paired strategies.

    Attempt-specific SHA256 seeds permit order-invariant reproducibility.
    Probabilities are hidden from the sampler; only this evaluation fixture
    knows them. Do not confuse this with actual model rollout recordings.
    """

    def __init__(self, profiles: Sequence[TaskProfile], seed: int):
        self.profiles = {p.id: p for p in profiles}
        if len(self.profiles) != len(profiles):
            raise ValueError('Duplicate task IDs')
        self.seed = seed

    def sample(self, task_id: str, attempt: int) -> float:
        if attempt < 0:
            raise ValueError('Attempt index must be nonnegative')
        p = self.profiles[task_id]
        digest = hashlib.sha256(f'v04:{self.seed}:{task_id}:{attempt}'.encode()).digest()
        draw = random.Random(int.from_bytes(digest[:16], 'big')).random()
        if draw < p.p_unsafe:
            return -1.0
        return 1.0 if draw < p.p_unsafe + (1.0 - p.p_unsafe) * p.p_correct else 0.0


class RolloutTape(Protocol):
    def sample(self, task_id: str, attempt: int) -> float: ...


def run_strategy(profiles: Sequence[TaskProfile], config: BudgetConfig, strategy: str,
                 *, tape: RolloutTape | None = None) -> dict:
    """Compare fixed-n, fixed-2n and pilot→expand sampling at identical cap.

    All strategies consume complete initial groups in reproducible task order.
    Adaptive expansion allocates one sample at a time only to flat groups,
    rotating fairly rather than reading unknown future values.
    """
    if strategy not in ('fixed', 'large_fixed', 'adaptive'):
        raise ValueError(f'Unknown strategy {strategy}')
    if not profiles:
        raise ValueError('At least one profile is required')
    if len({p.id for p in profiles}) != len(profiles):
        raise ValueError('Duplicate task IDs')
    tape = tape if tape is not None else RewardTape(profiles, config.seed)
    rng = random.Random(config.seed)
    ordered = list(profiles)
    rng.shuffle(ordered)
    group_n = config.initial if strategy == 'adaptive' else (
        config.group_size * 2 if strategy == 'large_fixed' else config.group_size
    )
    if group_n > config.max_per_task:
        raise ValueError('Max attempts must allow the fixed group size')
    states: dict[str, TaskState] = {}
    spent = 0
    # Fully complete each initial group; no partial groups enter GRPO analysis.
    for profile in ordered:
        if config.budget - spent < group_n:
            break
        s = TaskState(profile.id, profile.family)
        for _ in range(group_n):
            s.rewards.append(tape.sample(profile.id, len(s.rewards)))
            spent += 1
        states[profile.id] = s
    if strategy == 'adaptive':
        # A pre-committed round robin, with no hidden task probability access.
        # A group stops expanding immediately when it becomes informative.
        while spent < config.budget:
            pending = [s for s in states.values() if s.flat and len(s.rewards) < config.max_per_task]
            if not pending:
                break
            for s in pending:
                if spent >= config.budget:
                    break
                s.rewards.append(tape.sample(s.id, len(s.rewards)))
                spent += 1

    informative = sum(s.informative for s in states.values())
    flat = sum(s.flat for s in states.values())
    families = sorted({p.family for p in profiles})
    correctness_signal = sum(1 in s.rewards and (0 in s.rewards or -1 in s.rewards) for s in states.values())
    unsafe_only_signal = sum(s.informative and 1 not in s.rewards for s in states.values())
    per_family = {}
    for family in families:
        fs = [s for s in states.values() if s.family == family]
        per_family[family] = {
            'covered': len(fs), 'informative': sum(s.informative for s in fs),
            'rollouts': sum(len(s.rewards) for s in fs),
        }
    result = {
        'strategy': strategy, 'seed': config.seed,
        'budget': config.budget, 'rollouts_used': spent,
        'unused_rollouts': config.budget - spent,
        'task_count': len(profiles), 'task_coverage': len(states),
        'informative_tasks': informative, 'flat_tasks': flat,
        'correctness_signal_tasks': correctness_signal,
        'unsafe_without_correctness_signal_tasks': unsafe_only_signal,
        'informative_per_100_rollouts': round(100 * informative / spent, 5) if spent else 0,
        'informative_share_of_covered': round(informative / len(states), 6) if states else 0,
        'unsafe_rate': round(sum(r == -1 for s in states.values() for r in s.rewards) / spent, 6) if spent else 0,
        'family_metrics': per_family,
        'rollout_lengths': {s.id: len(s.rewards) for s in states.values()},
        'rollout_trace_digest': hashlib.sha256(json.dumps(
            {k: v.rewards for k, v in sorted(states.items())}, sort_keys=True
        ).encode()).hexdigest(),
        'interpretation': 'synthetic reward collection only; no GRPO update or model accuracy',
    }
    return result


def build_profiles(n: int = 144, seed: int = 93,
                   scenario: str = "imbalanced") -> list[TaskProfile]:
    """Deliberately imbalanced tasks with hard, medium, easy, and unsafe groups."""
    if n < 4:
        raise ValueError('At least four profiles required')
    scenarios = {
        'imbalanced': [('low_success', .04, .025), ('mid_success', .40, .015),
                       ('high_success', .94, .01), ('unsafe_pressure', .20, .25)],
        'uniform_mid': [('mid_a', .45, .02), ('mid_b', .55, .02),
                        ('mid_c', .50, .02), ('mid_d', .40, .02)],
        'mostly_wrong': [('hard_a', .005, .005), ('hard_b', .01, .005),
                         ('hard_c', .02, .005), ('hard_d', .03, .005)],
    }
    if scenario not in scenarios:
        raise ValueError('Unknown scenario')
    groups = scenarios[scenario]
    rng = random.Random(seed)
    profiles = []
    for i in range(n):
        family, p, unsafe = groups[i % len(groups)]
        jitter = rng.uniform(-.02, .02)
        profiles.append(TaskProfile(f'task-{i:04d}', family,
                                    min(.999, max(.001, p + jitter)), unsafe))
    return profiles


def paired_sweep(n: int = 144, *, budget_per_task: int = 4, repetitions: int = 40,
                 seed: int = 2026, scenario: str = "imbalanced") -> dict:
    if not 1 <= repetitions <= 500:
        raise ValueError('repetitions must be 1..500')
    if budget_per_task < 2:
        raise ValueError('budget_per_task must be >= 2')
    profiles = build_profiles(n, scenario=scenario)
    arms = ['fixed', 'large_fixed', 'adaptive']
    all_runs = {a: [] for a in arms}
    for i in range(repetitions):
        cfg = BudgetConfig(budget=n * budget_per_task, seed=seed + i)
        for arm in arms:
            all_runs[arm].append(run_strategy(profiles, cfg, arm))
    summary = {}
    for arm in arms:
        rr = all_runs[arm]
        summary[arm] = {key: round(sum(run[key] for run in rr) / repetitions, 4)
                        for key in ('rollouts_used', 'unused_rollouts', 'task_coverage',
                                    'informative_tasks', 'correctness_signal_tasks',
                                    'unsafe_without_correctness_signal_tasks',
                                    'informative_per_100_rollouts',
                                    'informative_share_of_covered', 'unsafe_rate')}
    paired = []
    for base, candidate in [('fixed', 'adaptive'), ('large_fixed', 'adaptive')]:
        differences = [c['informative_tasks'] - b['informative_tasks']
                       for b, c in zip(all_runs[base], all_runs[candidate])]
        mean = sum(differences) / repetitions
        # Descriptive repeated-simulation quantiles, NOT CI of model-performance lift.
        values = sorted(differences)
        paired.append({'candidate': candidate, 'baseline': base,
                       'mean_informative_task_difference': round(mean, 4),
                       'empirical_95pct_range': [values[int(.025 * (repetitions - 1))],
                                                 values[int(.975 * (repetitions - 1))]],
                       'candidate_wins': sum(d > 0 for d in differences),
                       'ties': sum(d == 0 for d in differences)})
    return {
        'experiment': 'v04_frozen_reward_tape_CPU_DIAGNOSTICS_ONLY',
        'profiles': n, 'scenario': scenario, 'rollout_budget_each': n * budget_per_task,
        'repetitions': repetitions, 'seed': seed,
        'strategy_means': summary, 'paired_differences': paired,
        'disclaimer': ('Rewards are generated from fixed latent probabilities, not a language model. '
                       'Variable-length groups are NOT wired into AGL/verl. '
                       'This does not show policy learning, convergence or downstream accuracy.'),
    }
