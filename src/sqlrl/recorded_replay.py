"""Strict, train-only replay of externally recorded *frozen-policy* rollout rewards.

For counterfactual sampling each task must have >= max_per_task attempts already
recorded. Absence of an attempt is NEVER filled by an invented model output.
Scores must come from a single policy checkpoint and decoding configuration.
No connection to the AGL event stream is claimed or implemented here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .adaptive_sampling import BudgetConfig, TaskProfile, run_strategy


ALLOWED_FIELDS = {'task_id', 'family', 'split', 'attempt', 'reward', 'policy_id', 'decode_id'}


class RecordedTape:
    def __init__(self, rewards: dict[str, tuple[float, ...]]):
        self.rewards = rewards

    def sample(self, task_id: str, attempt: int) -> float:
        if attempt < 0:
            raise ValueError('Negative attempt')
        return self.rewards[task_id][attempt]


def load_reward_replay(path: str | Path, *, required_attempts: int = 8
                       ) -> tuple[list[TaskProfile], RecordedTape, dict]:
    if required_attempts < 2 or required_attempts > 32:
        raise ValueError('required_attempts must be 2..32')
    rows: dict[str, dict[int, float]] = {}
    families: dict[str, str] = {}
    policies: set[str] = set()
    decodes: set[str] = set()
    count = 0
    raw = Path(path).read_bytes()
    for line_no, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != ALLOWED_FIELDS:
            raise ValueError(f'Line {line_no}: invalid keys (never include gold SQL or secrets)')
        if row['split'] != 'train':
            raise ValueError('Replay refuses validation, test and OOD rollouts')
        tid, family = row['task_id'], row['family']
        if not isinstance(tid, str) or not tid or not isinstance(family, str) or not family:
            raise ValueError(f'Line {line_no}: task id/family invalid')
        if type(row['attempt']) is not int or not 0 <= row['attempt'] < required_attempts:
            raise ValueError(f'Line {line_no}: attempt out of range')
        reward = row['reward']
        if type(reward) not in (int, float) or not math.isfinite(reward) or not -1 <= reward <= 1:
            raise ValueError(f'Line {line_no}: invalid reward')
        if not all(isinstance(row[k], str) and row[k] for k in ('policy_id', 'decode_id')):
            raise ValueError(f'Line {line_no}: policy/decode identifiers required')
        policies.add(row['policy_id'])
        decodes.add(row['decode_id'])
        if tid in families and families[tid] != family:
            raise ValueError(f'Line {line_no}: contradictory family')
        families[tid] = family
        attempts = rows.setdefault(tid, {})
        if row['attempt'] in attempts:
            raise ValueError(f'Line {line_no}: duplicate task/attempt')
        attempts[row['attempt']] = float(reward)
        count += 1
    if not rows:
        raise ValueError('Empty replay')
    if len(policies) != 1 or len(decodes) != 1:
        raise ValueError('Replay must contain one frozen model and decoding policy')
    full = {}
    for task, attempts in rows.items():
        if set(attempts) != set(range(required_attempts)):
            raise ValueError(f'Task {task} missing attempts; cannot invent counterfactual rollouts')
        full[task] = tuple(attempts[i] for i in range(required_attempts))
    profiles = [TaskProfile(tid, families[tid], .5, 0) for tid in sorted(families)]
    policy_id = next(iter(policies))
    is_demo = policy_id.startswith('SYNTHETIC_DEMO_')
    return profiles, RecordedTape(full), {
        'source': ('synthetic_demo_not_a_model' if is_demo else
                   'externally_supplied_frozen_policy_recorded_reward_replay'),
        'file_sha256': hashlib.sha256(raw).hexdigest(),
        'policy_id': policy_id, 'decode_id': next(iter(decodes)),
        'tasks': len(profiles), 'records': count,
        'counterfactual_limit': ('The recorded model behavior is frozen; no intervention '
                                 'or policy learning is estimated.'),
    }


def compare_recorded(path: str | Path, *, budget_per_task: int = 4,
                     max_per_task: int = 8, seed: int = 2026) -> dict:
    profiles, tape, metadata = load_reward_replay(path, required_attempts=max_per_task)
    config = BudgetConfig(budget=len(profiles) * budget_per_task,
                          max_per_task=max_per_task, seed=seed)
    strategies = {arm: run_strategy(profiles, config, arm, tape=tape)
                  for arm in ('fixed', 'large_fixed', 'adaptive')}
    return {'experiment': 'v04_frozen_policy_reward_tape_REPLAY_NO_TRAINING',
            'input': metadata, 'strategies': strategies,
            'warning': ('Reward replay requires real recorded rollouts produced separately. '
                        'It cannot establish policy improvement or predict on-policy GRPO outcomes.')}
