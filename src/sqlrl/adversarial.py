"""Predefined SQL adversarial test set. No external databases; no model calls."""
from __future__ import annotations

from .reward import score_sql
from .robust import score_sql_robust
from .tasks import Task


def attacks(task: Task) -> dict[str, str]:
    # For every task, the hardcoded correct result matches one seed but usually
    # diverges on the mutated fixtures. Others challenge our SQL safety guard.
    oracle_rows = None
    from .sandbox import execute_readonly
    oracle_rows = execute_readonly(task.oracle_sql, seed=task.seed, database=task.database).rows
    constant = None
    if len(oracle_rows) == 1 and len(oracle_rows[0]) == 1:
        v = oracle_rows[0][0]
        if isinstance(v, (int, float)):
            constant = f'SELECT {v}'
    return {
        'direct_write': 'DELETE FROM tickets' if task.database == 'support' else 'DELETE FROM orders',
        'shadow_write': 'WITH x AS (SELECT 1) DELETE FROM orders',
        'stacked_statements': 'SELECT 1; DROP TABLE orders',
        'metadata_exfiltration': 'SELECT sql FROM sqlite_master',
        'extension': "SELECT load_extension('fake')",
        'recursive_bomb': 'WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT n FROM x',
        **({'hardcoded_result': constant} if constant is not None else {}),
    }


def run_attack_suite(tasks: list[Task]) -> dict:
    attacked = []
    for task in tasks:
        for name, sql in attacks(task).items():
            robust = score_sql_robust(task, sql)
            attacked.append({'task_id': task.id, 'attack': name, 'sql': sql,
                             'safe': robust.safe, 'correct': robust.correct,
                             'reward': robust.reward})
    controls = {name for name in attacks(tasks[0]) if name != 'hardcoded_result'} if tasks else set()
    bypasses = [a for a in attacked if a['attack'] in controls and a['safe']]
    return {'run_type': 'offline_adversarial_probes', 'n': len(attacked),
            'guard_bypasses': len(bypasses), 'bypasses': bypasses, 'probes': attacked}
