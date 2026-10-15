"""Pluggable policies: deterministic baseline and OpenAI-compatible model gateway."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from .fixtures import schema_for
from .tasks import Task


class Policy(Protocol):
    def complete(self, task: Task, feedback: str | None = None) -> str: ...


SYSTEM_PROMPT = """You are a SQLite text-to-SQL assistant. Output exactly one read-only SELECT query,
without explanations or Markdown. Use only these tables and fields:
{schema}
Never use external data, write statements, ATTACH, PRAGMA or extension functions.
Do not assume missing columns. Corrected SQL must still follow all restrictions.
"""


def build_messages(task: Task, feedback: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema_for(task.database))},
        {"role": "user", "content": task.question},
    ]
    if feedback:
        # Feedback only contains parser/execution diagnostics, never oracle results.
        messages.append({"role": "user", "content": "Your previous query could not execute. Error: "
                         + feedback[:160] + ". Return a corrected SELECT query."})
    return messages


class OpenAICompatiblePolicy:
    def __init__(self, base_url: str, api_key: str, model: str = "auto", timeout: float = 90) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) endpoint")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, task: Task, feedback: str | None = None) -> str:
        payload = {"model": self.model, "messages": build_messages(task, feedback), "max_tokens": 384}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                body = json.load(res)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Model gateway HTTP {exc.code}") from exc
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("No completions returned")
        content = choices[0]["message"].get("content")
        if not isinstance(content, str):
            raise RuntimeError("Expected a text-only completion")
        return content


class BaselinePolicy:
    """Intentionally limited non-learning control policy for a baseline, not an oracle."""

    def complete(self, task: Task, feedback: str | None = None) -> str:
        question = task.question.lower()
        if "number of completed orders" in question or "count completed orders" in question:
            return "SELECT COUNT(*) FROM orders WHERE status='completed'"
        if "completed revenue" in question or "total revenue" in question:
            return ("SELECT ROUND(SUM(oi.quantity*oi.unit_price),2) "
                    "FROM orders o JOIN order_items oi ON o.order_id=oi.order_id "
                    "WHERE o.status='completed'")
        if "orders by status" in question:
            return "SELECT status, COUNT(*) FROM orders GROUP BY status"
        if "customers by region" in question:
            return "SELECT region, COUNT(*) FROM customers GROUP BY region"
        return "SELECT COUNT(*) FROM orders"  # intentionally poor generalization


def policy_from_env() -> Policy:
    base_url = os.environ.get("AGL_OPENAI_BASE_URL") or os.environ.get("SQLRL_OPENAI_BASE_URL")
    if not base_url:
        if os.environ.get("AGL_EVENT_URL"):
            raise RuntimeError("Agent Lightning rollout is missing AGL_OPENAI_BASE_URL")
        return BaselinePolicy()
    api_key = os.environ.get("AGL_KEY") or os.environ.get("SQLRL_API_KEY", "unused")
    return OpenAICompatiblePolicy(base_url, api_key, os.environ.get("SQLRL_MODEL", "auto"))
