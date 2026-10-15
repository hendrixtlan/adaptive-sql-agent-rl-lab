"""Minimal authenticated reward client. Never includes gold SQL or expected rows."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


def request_reward(endpoint: str, token: str, task_id: str, sql: str,
                   *, timeout: float = 30.0) -> float:
    if not endpoint.startswith("http://127.0.0.1:") and not endpoint.startswith("http://localhost:"):
        raise ValueError("Prototype reward client only supports a loopback evaluator")
    if len(token) < 24:
        raise ValueError("Missing or weak reward token")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/reward",
        data=json.dumps({"task_id": task_id, "sql": sql}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    if set(data) != {"task_id", "reward"} or data["task_id"] != task_id:
        raise RuntimeError("Unexpected reward service response")
    reward = data["reward"]
    if not isinstance(reward, (int, float)) or not -1 <= reward <= 1:
        raise RuntimeError("Invalid reward scalar")
    return float(reward)
