"""Benchmark records; gold SQL is never placed in an agent prompt."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    oracle_sql: str
    database: str = "retail"
    seed: int = 42
    difficulty: str = "basic"

    @classmethod
    def from_dict(cls, value: dict) -> "Task":
        from .fixtures import SCHEMAS
        if value.get("database", "retail") not in SCHEMAS:
            raise ValueError("Unsupported database")
        return cls(
            id=str(value["id"]),
            question=str(value["question"]),
            oracle_sql=str(value["oracle_sql"]),
            database=str(value.get("database", "retail")),
            seed=int(value.get("seed", 42)),
            difficulty=str(value.get("difficulty", "basic")),
        )


def load_tasks(path: str | Path) -> list[Task]:
    tasks = [Task.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task identifiers")
    return tasks
