"""Deterministic, executable rewards. Never use the model to judge itself."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .sandbox import QueryRejected, equivalent_rows, execute_readonly
from .tasks import Task


@dataclass(frozen=True)
class Score:
    task_id: str
    reward: float
    correct: bool
    executable: bool
    safe: bool
    predicted_rows: int
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def score_sql(task: Task, proposed_sql: str) -> Score:
    # The oracle comes only from our trusted benchmark; it is NEVER sent to the model.
    gold = execute_readonly(task.oracle_sql, seed=task.seed, database=task.database)
    try:
        candidate = execute_readonly(proposed_sql, seed=task.seed, database=task.database)
    except QueryRejected as exc:
        return Score(task.id, -1.0, False, False, False, 0, 0.0, str(exc))
    correct = equivalent_rows(candidate.rows, gold.rows)
    # Reward deliberately favors correctness over token cost. Wall-clock latency
    # is logged but is not part of reward: noisy microbenchmarks are gameable.
    reward = 1.0 if correct else 0.05
    return Score(task.id, reward, correct, True, True, len(candidate.rows), candidate.latency_ms)
