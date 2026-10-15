"""A read-only SQLite benchmark evaluator, NOT a hardened hostile-code sandbox.

Only generated text is treated as untrusted SQL. Benchmarks and oracle SQL must
come from a trusted repository. No connection to live corporate databases.
"""
from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .fixtures import TABLES, create_connection


class QueryRejected(ValueError):
    """Safety or execution-limit violation."""


@dataclass(frozen=True)
class QueryResult:
    rows: tuple[tuple[Any, ...], ...]
    latency_ms: float


_ALLOWED_FUNCTIONS = {"sum", "avg", "count", "min", "max", "round", "coalesce", "lower", "upper",
                      "substr", "length", "abs", "strftime", "date", "ifnull"}
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                    sqlite3.SQLITE_RECURSIVE}


def prepare_sql(text: str) -> str:
    sql = text.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```\s*$", "", sql)
    sql = sql.strip().rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        raise QueryRejected("Only SELECT/WITH statements are allowed")
    if ";" in sql:
        raise QueryRejected("Multiple statements are not allowed")
    if len(sql) > 8000:
        raise QueryRejected("Query length limit exceeded")
    return sql


def _norm_value(x: Any) -> tuple[str, str]:
    if x is None:
        return ("null", "")
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return ("number", f"{float(x):.6f}")
    return ("text", str(x))


def equivalent_rows(a: tuple[tuple[Any, ...], ...], b: tuple[tuple[Any, ...], ...]) -> bool:
    # Comparison is bag equality; ordering is not rewarded in v0.1.
    norm = lambda rows: Counter(tuple(_norm_value(x) for x in row) for row in rows)
    return norm(a) == norm(b)


def execute_readonly(
    sql: str,
    *,
    seed: int = 42,
    database: str = "retail",
    max_rows: int = 200,
    max_steps: int = 100000,
    timeout_seconds: float = 0.5,
) -> QueryResult:
    """Execute a single SELECT on an isolated fixture with SQLite authorizer + VM-step limits.

    For adversarial deployments use a separate OS/container sandbox with strict
    CPU/memory/network constraints. A Python process is not a security boundary.
    """
    statement = prepare_sql(sql)
    conn = create_connection(database, seed)
    start = time.monotonic()
    steps = 0

    def authorize(action: int, arg1: str | None, arg2: str | None, *_: Any) -> int:
        if action == sqlite3.SQLITE_READ and arg1 not in TABLES[database]:
            return sqlite3.SQLITE_DENY
        if action not in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() not in _ALLOWED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def progress() -> int:
        nonlocal steps
        steps += 100
        if steps >= max_steps or time.monotonic() - start > timeout_seconds:
            return 1
        return 0

    try:
        conn.set_authorizer(authorize)
        conn.set_progress_handler(progress, 100)
        cursor = conn.execute(statement)
        rows = tuple(tuple(row) for row in cursor.fetchmany(max_rows + 1))
        if len(rows) > max_rows:
            raise QueryRejected("Row limit exceeded")
        if time.monotonic() - start > timeout_seconds:
            raise QueryRejected("Timeout exceeded")
        return QueryResult(rows=rows, latency_ms=(time.monotonic() - start) * 1000)
    except sqlite3.DatabaseError as exc:
        raise QueryRejected(f"SQL rejected or execution failed: {type(exc).__name__}") from exc
    finally:
        conn.close()
