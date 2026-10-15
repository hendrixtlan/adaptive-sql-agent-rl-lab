"""AGL event emission independent of trusted SQL evaluation libraries."""
from __future__ import annotations
import json
import os
import urllib.request


def emit_agl_reward(value: float) -> None:
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
