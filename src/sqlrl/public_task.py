"""Minimal rollout envelope: NEVER serialize gold query, evaluator seeds or reward secret."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .fixtures import SCHEMAS
from .tasks import Task


@dataclass(frozen=True)
class PublicTask:
    id: str
    question: str
    database: str

    @classmethod
    def from_task(cls, task: Task) -> "PublicTask":
        return cls(id=task.id, question=task.question, database=task.database)

    @classmethod
    def from_dict(cls, data: dict) -> "PublicTask":
        if set(data) != {"id", "question", "database"}:
            raise ValueError("Rollout payload must contain only id, question, database")
        if data["database"] not in SCHEMAS:
            raise ValueError("Unsupported public schema")
        return cls(id=str(data["id"]), question=str(data["question"]), database=data["database"])

    def to_dict(self) -> dict:
        return asdict(self)
