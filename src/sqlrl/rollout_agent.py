"""Agent Lightning v1 agent rollout with NO gold SQL inside the task envelope.

One LLM completion per rollout, one trusted reward HTTP request. The model never
receives grader feedback; the AGL event emits the scalar to the RL trainer only.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os

from .agl_events import emit_agl_reward
from .oracle_client import request_reward
from .policy import policy_from_env
from .public_task import PublicTask


class OracleSeparatedAgent:
    async def run(self) -> None:
        public = PublicTask.from_dict(json.loads(os.environ["PUBLIC_TASK_JSON"]))
        policy = policy_from_env()
        sql = await asyncio.to_thread(policy.complete, public)
        reward = await asyncio.to_thread(request_reward,
                                         os.environ["SQLRL_REWARD_URL"],
                                         os.environ["SQLRL_REWARD_TOKEN"],
                                         public.id, sql)
        await asyncio.to_thread(emit_agl_reward, reward)
        # Avoid dumping SQL or evaluator responses into logs accidentally.
        print(json.dumps({"task_id": public.id,
                          "candidate_sha256": hashlib.sha256(sql.encode()).hexdigest()[:16],
                          "status": "reward_emitted"}), flush=True)


if __name__ == "__main__":
    asyncio.run(OracleSeparatedAgent().run())
