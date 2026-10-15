import json
import tempfile
import unittest
from pathlib import Path

from sqlrl.agent import run_episode
from sqlrl.cli import simulate_grpo, smoke
from sqlrl.fixtures import create_retail_connection
from sqlrl.grpo import group_relative_advantages
from sqlrl.policy import BaselinePolicy, build_messages
from sqlrl.reward import score_sql
from sqlrl.sandbox import QueryRejected, equivalent_rows, execute_readonly, prepare_sql
from sqlrl.tasks import Task, load_tasks


ROOT = Path(__file__).resolve().parent.parent


class SQLSandboxTests(unittest.TestCase):
    def test_valid_select(self):
        self.assertEqual(execute_readonly("SELECT COUNT(*) FROM orders").rows, ((180,),))

    def test_fence(self):
        self.assertEqual(prepare_sql("```sql\nSELECT COUNT(*) FROM orders;\n```"),
                         "SELECT COUNT(*) FROM orders")

    def test_reject_dml_ddl_pragmas_and_attach(self):
        for sql in ("DROP TABLE customers", "DELETE FROM customers", "UPDATE customers SET region='x'",
                    "INSERT INTO customers VALUES(999,'x','x')", "PRAGMA table_info(customers)",
                    "ATTACH DATABASE '/tmp/x' AS stolen", "SELECT 1; DELETE FROM orders"):
            with self.subTest(sql=sql), self.assertRaises(QueryRejected):
                execute_readonly(sql)

    def test_reject_mutating_cte(self):
        with self.assertRaises(QueryRejected):
            execute_readonly("WITH q AS (SELECT 1) DELETE FROM orders")

    def test_reject_unlisted_function(self):
        with self.assertRaises(QueryRejected):
            execute_readonly("SELECT sqlite_version()")

    def test_no_database_mutation_leak(self):
        with self.assertRaises(QueryRejected):
            execute_readonly("SELECT load_extension('bad')")
        self.assertEqual(execute_readonly("SELECT COUNT(*) FROM orders").rows, ((180,),))

    def test_row_cap(self):
        with self.assertRaises(QueryRejected):
            execute_readonly("SELECT * FROM orders", max_rows=10)

    def test_bag_equivalence(self):
        self.assertTrue(equivalent_rows(((1, 'x'),(2,'y')), ((2.0,'y'),(1,'x'))))
        self.assertFalse(equivalent_rows(((1,),),((1,),(1,))))


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.task = load_tasks(ROOT / "data/train.jsonl")[0]

    def test_exact_score(self):
        result = score_sql(self.task, self.task.oracle_sql)
        self.assertEqual(result.reward, 1)
        self.assertTrue(result.correct)

    def test_wrong_but_safe(self):
        result = score_sql(self.task, "SELECT COUNT(*) FROM products")
        self.assertEqual(result.reward, 0.05)
        self.assertFalse(result.correct)

    def test_unsafe_negative(self):
        result = score_sql(self.task, "DROP TABLE orders")
        self.assertEqual(result.reward, -1)
        self.assertFalse(result.safe)

    def test_no_oracle_in_prompt(self):
        task = load_tasks(ROOT / "data/train.jsonl")[6]
        messages = build_messages(task)
        combined = "\n".join(m['content'] for m in messages)
        self.assertNotIn(task.oracle_sql, combined)
        self.assertNotIn('oracle_sql', combined)

    def test_all_oracles_execute_and_match(self):
        for split in ('train','val','test'):
            for task in load_tasks(ROOT / f'data/{split}.jsonl'):
                with self.subTest(task=task.id):
                    self.assertTrue(score_sql(task, task.oracle_sql).correct)


class GRPOTests(unittest.TestCase):
    def test_normalization_and_centering(self):
        group = group_relative_advantages([0, 1, 0, -1])
        self.assertAlmostEqual(sum(group.advantages), 0, places=7)
        self.assertFalse(group.zero_variance)

    def test_zero_variance(self):
        result = group_relative_advantages([0.05,0.05,0.05,0.05])
        self.assertEqual(result.advantages, (0.0,)*4)
        self.assertTrue(result.zero_variance)

    def test_requires_group(self):
        with self.assertRaises(ValueError):
            group_relative_advantages([1])


class EndToEndTests(unittest.TestCase):
    def test_episode_baseline(self):
        # No model configured -> deterministic non-learning policy.
        task = load_tasks(ROOT / 'data/train.jsonl')[0]
        sql, score, attempts = run_episode(task)
        self.assertTrue(score.correct)
        self.assertEqual(attempts, 1)

    def test_cli_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            base = smoke(str(ROOT / 'data/train.jsonl'), str(Path(temp) / 'baseline.json'))
            self.assertEqual(base['n'], 26)
            self.assertTrue((Path(temp) / 'baseline.json').exists())
            diag = simulate_grpo(str(ROOT / 'data/train.jsonl'), str(Path(temp) / 'grpo.json'))
            self.assertEqual(diag['n_groups'], 26)
            self.assertTrue((Path(temp) / 'grpo.json').exists())


if __name__ == '__main__':
    unittest.main()
