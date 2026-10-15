"""Agent Lightning v1.0 local-controller entrypoint (no legacy LitAgent API)."""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from .policy import policy_from_env
from .reward import Score, score_sql
from .tasks import Task


def run_episode(task: Task, *, max_attempts: int = 2) -> tuple[str, Score, int]:
    policy = policy_from_env()
    feedback = None
    for attempt in range(1, max_attempts + 1):
        sql = policy.complete(task, feedback=feedback)
        score = score_sql(task, sql)
        # Only execution errors yield actionable feedback. Correctness is held
        # out: giving a gold-derived signal here would contaminate evaluation.
        if score.executable or attempt == max_attempts:
            return sql, score, attempt
        feedback = score.error
    raise AssertionError("Unreachable")


def emit_agl_reward(value: float) -> None:
    """Agent Lightning v1 event endpoint; failure must propagate, not be hidden."""
    event_url = os.environ["AGL_EVENT_URL"]
    key = os.environ["AGL_KEY"]
    req = urllib.request.Request(
        event_url,
        data=json.dumps({"event_type": "reward", "data": {"value": value}}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        if not (200 <= response.status < 300):
            raise RuntimeError(f"Agent Lightning event rejected: HTTP {response.status}")


class Agent:
    """Imported by agl-controller runner_type=local (Agent.run is awaited)."""

    async def run(self) -> None:
        # TASK_JSON comes from the trainer's env_map, not from the model.
        task = Task.from_dict(json.loads(os.environ["TASK_JSON"]))
        sql, score, attempts = await asyncio.to_thread(run_episode, task)
        from .robust import score_sql_robust
        robust = await asyncio.to_thread(score_sql_robust, task, sql)
        print(json.dumps({"task_id": task.id, "sql": sql, "attempts": attempts,
                          "score": score.to_dict(), "robust_score": robust.to_dict()}), flush=True)
        await asyncio.to_thread(emit_agl_reward, robust.reward)


if __name__ == "__main__":
    asyncio.run(Agent().run())
