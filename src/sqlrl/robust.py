"""Multi-instance execution verification to discourage hardcoded-result reward hacking.

Trusted evaluator ONLY. Does not execute arbitrary OS commands and is not a secure
container boundary for hostile code. The candidate sees only schema/question.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from .sandbox import QueryRejected, equivalent_rows, execute_readonly
from .tasks import Task


@dataclass(frozen=True)
class RobustScore:
    task_id: str
    database: str
    reward: float
    correct: bool
    safe: bool
    executable: bool
    correct_instances: int
    total_instances: int
    errors: tuple[str, ...]
    mean_latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def verification_seeds(task: Task, n: int = 3) -> tuple[int, ...]:
    """Deterministic disjoint seeds (base fixture + independently seeded mutations).

    These seeds are NEVER included in the policy prompt. More seeds are a defense
    in depth, not proof of semantic equivalence over every possible database.
    """
    if not 1 <= n <= 12:
        raise ValueError("n must be 1..12")
    seeds = [task.seed]
    for i in range(1, n):
        digest = hashlib.sha256(f"sqlrl-v02:{task.id}:{task.database}:{task.seed}:{i}".encode()).digest()
        proposed = 10_000 + int.from_bytes(digest[:4], "big") % 1_000_000
        while proposed in seeds:
            proposed += 1
        seeds.append(proposed)
    return tuple(seeds)


def score_sql_robust(task: Task, proposed_sql: str, *, n_instances: int = 3) -> RobustScore:
    correct = 0
    latency = 0.0
    executed = 0
    errors = []
    safe = True
    for seed in verification_seeds(task, n_instances):
        gold = execute_readonly(task.oracle_sql, seed=seed, database=task.database)
        try:
            candidate = execute_readonly(proposed_sql, seed=seed, database=task.database)
        except QueryRejected as exc:
            safe = False
            errors.append(str(exc))
            break
        latency += candidate.latency_ms
        executed += 1
        correct += equivalent_rows(candidate.rows, gold.rows)
    if not safe:
        # Never reward a candidate if it fails on even one verification instance.
        return RobustScore(task.id, task.database, -1.0, False, False, False,
                           correct, n_instances, tuple(errors), latency / max(1, executed))
    all_correct = correct == n_instances
    # Partial credit gives a gradient only when the candidate runs safely, while
    # exact full credit requires success across all independent instances.
    reward = 1.0 if all_correct else 0.05 + 0.65 * (correct / n_instances)
    return RobustScore(task.id, task.database, round(reward, 6), all_correct, True, True,
                       correct, n_instances, (), latency / n_instances)
