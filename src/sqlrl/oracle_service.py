"""Trusted HTTP reward service holding private oracle SQL and secret verifier seeds.

Threat model: protects against accidental gold-in-rollout leakage. NOT a hardened
network sandbox. Use a private network and process/container isolation; do NOT
mount data/v03 or this package's oracle files in a hostile agent container.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock
from urllib.parse import urlsplit

from .robust import score_sql_with_seeds
from .tasks import Task, load_tasks


class TrustedOracle:
    def __init__(self, manifests: list[str | Path], *, token: str, seed_key: str,
                 n_instances: int = 5, max_scoring_per_task: int = 5000):
        if len(token) < 24 or len(seed_key) < 24:
            raise ValueError("Independent service token and verifier seed key must be >=24 characters")
        if not (2 <= n_instances <= 12):
            raise ValueError("n_instances must be 2..12")
        self.token = token
        self.seed_key = seed_key.encode()
        self.n_instances = n_instances
        self.max_scoring_per_task = max_scoring_per_task
        self._lock = Lock()
        self._counts: dict[str, int] = {}
        self.tasks: dict[str, Task] = {}
        for path in manifests:
            split = Path(path).stem
            # Never expose held-out gold to a trainer-accessible reward service.
            if split not in {"train", "val"}:
                raise ValueError("Training reward service accepts train/val manifests ONLY")
            for task in load_tasks(path):
                if task.id in self.tasks:
                    raise ValueError("Duplicate oracle task ID")
                self.tasks[task.id] = task
        if not self.tasks:
            raise ValueError("Oracle task set empty")

    def seeds(self, task_id: str) -> tuple[int, ...]:
        task = self.tasks[task_id]
        seeds = []
        for i in range(self.n_instances):
            label = f"v03-verifier:{task.id}:{task.database}:{task.seed}:{i}".encode()
            digest = hmac.new(self.seed_key, label, hashlib.sha256).digest()
            value = 100_000 + int.from_bytes(digest[:8], "big") % 1_000_000_000
            while value in seeds:
                value += 1
            seeds.append(value)
        return tuple(seeds)

    def score(self, task_id: str, sql: str) -> dict:
        if task_id not in self.tasks:
            raise KeyError("Unknown or unavailable task")
        if not isinstance(sql, str) or len(sql) > 8000:
            raise ValueError("SQL must be a string <=8000 chars")
        with self._lock:
            count = self._counts.get(task_id, 0) + 1
            if count > self.max_scoring_per_task:
                raise RuntimeError("Task scoring budget exceeded")
            self._counts[task_id] = count
        result = score_sql_with_seeds(self.tasks[task_id], sql, self.seeds(task_id))
        # Only a scalar reward is returned; never return gold, target rows or seed values.
        return {"task_id": task_id, "reward": result.reward}


def build_http_server(oracle: TrustedOracle, *, host: str = "127.0.0.1",
                      port: int = 8765) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only loopback binding supported by this local prototype")

    class Handler(BaseHTTPRequestHandler):
        limiter = BoundedSemaphore(6)

        def log_message(self, fmt: str, *args) -> None:
            # Avoid logging task content or auth credentials.
            return

        def _respond(self, status: int, body: dict) -> None:
            result = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        def do_GET(self) -> None:
            if urlsplit(self.path).path == "/healthz":
                return self._respond(200, {"status": "ok"})
            return self._respond(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/v1/reward":
                return self._respond(404, {"error": "not_found"})
            authorization = self.headers.get("Authorization", "")
            if not hmac.compare_digest(authorization, f"Bearer {oracle.token}"):
                return self._respond(401, {"error": "unauthorized"})
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self._respond(415, {"error": "unsupported_media_type"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._respond(400, {"error": "invalid_content_length"})
            if not 0 < n <= 12288:
                return self._respond(413, {"error": "invalid_body_size"})
            if not self.limiter.acquire(blocking=False):
                return self._respond(429, {"error": "too_many_requests"})
            try:
                data = json.loads(self.rfile.read(n).decode("utf-8"))
                if not isinstance(data, dict) or set(data) != {"task_id", "sql"}:
                    raise ValueError("Invalid reward payload")
                result = oracle.score(data["task_id"], data["sql"])
                self._respond(200, result)
            except KeyError:
                self._respond(404, {"error": "unknown_task"})
            except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
                self._respond(400, {"error": "bad_request"})
            except RuntimeError:
                self._respond(429, {"error": "scoring_budget_exceeded"})
            finally:
                self.limiter.release()

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/v03/train.jsonl")
    parser.add_argument("--val", default="data/v03/val.jsonl")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--instances", type=int, default=5)
    args = parser.parse_args()
    token = os.environ.get("SQLRL_REWARD_TOKEN", "")
    secret = os.environ.get("SQLRL_VERIFIER_SECRET", "")
    oracle = TrustedOracle([args.train, args.val], token=token, seed_key=secret, n_instances=args.instances)
    server = build_http_server(oracle, port=args.port)
    print(f"Training oracle listening on loopback port {server.server_port}; {len(oracle.tasks)} train/val tasks", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
