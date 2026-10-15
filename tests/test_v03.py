"""v0.3 scientific workflow: held-out families, gold isolation, sealed rewards."""
from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from sqlrl.curriculum import split_audit
from sqlrl.eval_v03 import compare_suite, evaluate_frozen, hidden_eval_seeds
from sqlrl.oracle_client import request_reward
from sqlrl.oracle_service import TrustedOracle, build_http_server
from sqlrl.public_task import PublicTask
from sqlrl.robust import score_sql_with_seeds
from sqlrl.tasks import load_tasks
from sqlrl.v03_benchmark import generate, write_benchmark, TRAIN_FAMILIES, TEST_FAMILIES, OOD_FAMILIES
from training.train_agl_v1 import training_rows

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "v03-local-test-only-token-aaaaaaaaaaaaa"
SECRET = "v03-local-test-only-evaluator-seed-bbbbb"


class BenchmarkTests(unittest.TestCase):
    def test_generated_files_are_reproducible_and_nonleaking(self):
        with tempfile.TemporaryDirectory() as path:
            x = write_benchmark(path)
            y = write_benchmark(path)
            self.assertEqual(x["sha256"], y["sha256"])
            self.assertTrue(x["audit"]["ok"])
            self.assertEqual(x["counts"], {"train": 144, "val": 48, "test": 48, "ood": 60})
            self.assertFalse(set(TRAIN_FAMILIES) & set(TEST_FAMILIES))
            self.assertFalse(set(TRAIN_FAMILIES) & set(OOD_FAMILIES))
            self.assertEqual(set(x["audit"]["schemas"]["ood"]), {"inventory"})

    def test_all_300_gold_queries_evaluate_on_hidden_seeds(self):
        for split in ("train", "val", "test", "ood"):
            for task in load_tasks(ROOT / "data" / "v03" / (split + ".jsonl")):
                with self.subTest(task=task.id):
                    seeds = hidden_eval_seeds(task.id, task.database, task.seed, SECRET)
                    r = score_sql_with_seeds(task, task.oracle_sql, seeds)
                    self.assertTrue(r.correct, (task.id, r.errors))

    def test_training_rows_and_public_tasks_have_no_gold_or_secret(self):
        items = training_rows(str(ROOT / 'data/v03/train.jsonl'))
        self.assertEqual(len(items), 144)
        for r in items:
            self.assertNotIn("oracle_sql", json.dumps(r))
            self.assertNotIn("seed", json.dumps(r))
            data = json.loads(r["public_task_json"])
            self.assertEqual(set(data), {"id", "question", "database"})
            self.assertEqual(PublicTask.from_dict(data).to_dict(), data)
        with self.assertRaises(ValueError):
            PublicTask.from_dict({"id": "t", "question": "q", "database": "retail", "oracle_sql": "SELECT 1"})


class OracleServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = TrustedOracle([ROOT / "data/v03/train.jsonl", ROOT / "data/v03/val.jsonl"],
                                   token=TOKEN, seed_key=SECRET, n_instances=5,
                                   max_scoring_per_task=100)
        cls.server = build_http_server(cls.oracle, port=0)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_auth_success_and_constant_cheat(self):
        task = load_tasks(ROOT / 'data/v03/train.jsonl')[0]
        score = request_reward(self.url, TOKEN, task.id, task.oracle_sql)
        self.assertEqual(score, 1.0)
        self.assertNotEqual(request_reward(self.url, TOKEN, task.id, "SELECT 999999"), 1)
        self.assertEqual(request_reward(self.url, TOKEN, task.id, "DELETE FROM orders"), -1)

    def test_private_oracle_seeds_change_with_key(self):
        task_id = next(iter(self.oracle.tasks))
        a = self.oracle.seeds(task_id)
        alt = TrustedOracle([ROOT / 'data/v03/train.jsonl'], token=TOKEN,
                            seed_key="independent-seed-key-for-testing-00").seeds(task_id)
        self.assertNotEqual(a, alt)
        self.assertEqual(len(a), len(set(a)))

    def test_unauthenticated_request_and_heldout_task_rejected(self):
        task = load_tasks(ROOT / 'data/v03/train.jsonl')[0]
        req = Request(self.url + '/v1/reward', data=json.dumps({'task_id': task.id, 'sql': task.oracle_sql}).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
        with self.assertRaises(HTTPError) as exc:
            urlopen(req, timeout=3)
        self.assertEqual(exc.exception.code, 401)
        heldout = load_tasks(ROOT / 'data/v03/test.jsonl')[0]
        with self.assertRaises(HTTPError) as exc:
            request_reward(self.url, TOKEN, heldout.id, heldout.oracle_sql)
        self.assertEqual(exc.exception.code, 404)

    def test_training_oracle_cannot_be_constructed_with_test_or_ood(self):
        for name in ('test', 'ood'):
            with self.assertRaises(ValueError):
                TrustedOracle([ROOT / f'data/v03/{name}.jsonl'], token=TOKEN, seed_key=SECRET)

    def test_full_public_rollout_emits_reward_event_without_gold(self):
        import asyncio
        from sqlrl.rollout_agent import OracleSeparatedAgent
        task = load_tasks(ROOT / 'data/v03/train.jsonl')[2]
        public = PublicTask.from_task(task)
        self.assertNotIn('oracle_sql', json.dumps(public.to_dict()))
        emitted = []
        class FixedPolicy:
            def complete(self, inp, feedback=None):
                self.__class__.seen = inp.to_dict()
                return task.oracle_sql  # TEST DOUBLE ONLY: not available to the production agent
        env = {'PUBLIC_TASK_JSON': json.dumps(public.to_dict()),
               'SQLRL_REWARD_URL': self.url, 'SQLRL_REWARD_TOKEN': TOKEN}
        with patch.dict('os.environ', env), \
             patch('sqlrl.rollout_agent.policy_from_env', return_value=FixedPolicy()), \
             patch('sqlrl.rollout_agent.emit_agl_reward', side_effect=emitted.append):
            asyncio.run(OracleSeparatedAgent().run())
        self.assertEqual(emitted, [1.0])
        self.assertEqual(set(FixedPolicy.seen), {'id', 'question', 'database'})

    def test_only_scalar_reward_is_returned(self):
        task = load_tasks(ROOT / 'data/v03/train.jsonl')[1]
        req = Request(self.url + '/v1/reward', data=json.dumps({'task_id': task.id, 'sql': task.oracle_sql}).encode(),
                      headers={'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json'}, method='POST')
        with urlopen(req, timeout=5) as resp:
            data = json.load(resp)
        self.assertEqual(set(data), {'task_id', 'reward'})


class FrozenComparisonTests(unittest.TestCase):
    def test_frozen_baseline_results_are_repeatable(self):
        from sqlrl.policy import BaselinePolicy
        path = ROOT / 'data/v03/test.jsonl'
        a = evaluate_frozen(path, BaselinePolicy(), checkpoint_id='rules-v1', secret=SECRET)
        b = evaluate_frozen(path, BaselinePolicy(), checkpoint_id='rules-v1', secret=SECRET)
        self.assertEqual(a['accuracy'], b['accuracy'])
        self.assertEqual(a['protocol_hash'], b['protocol_hash'])
        self.assertEqual(a['n'], 48)
        self.assertEqual(a['evaluation_protocol'], 'five_secret_seed_strict_v03')
        self.assertNotIn(SECRET, json.dumps(a))

    def test_rejects_non_holdout_evaluation(self):
        from sqlrl.policy import BaselinePolicy
        with self.assertRaises(ValueError):
            evaluate_frozen(ROOT / 'data/v03/train.jsonl', BaselinePolicy(), checkpoint_id='x', secret=SECRET)

    def test_two_axis_suite_enforces_protocol_consistency(self):
        from sqlrl.policy import BaselinePolicy
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            for split in ('test', 'ood'):
                for label, ident, key in [('base', 'rules-v1', SECRET),
                                          ('candidate', 'rules-v2', SECRET)]:
                    evaluate_frozen(ROOT / f'data/v03/{split}.jsonl', BaselinePolicy(),
                                    checkpoint_id=ident, secret=key, output=d / f'{label}_{split}.json')
            report = compare_suite(*(str(d / f'{v}_{split}.json') for split in ('test', 'ood')
                                     for v in ('base', 'candidate')), resamples=100)
            self.assertFalse(report['research_gate_passed'])
            self.assertEqual(report['test']['absolute_accuracy_gain'], 0)
            corrupted = json.loads((d / 'candidate_ood.json').read_text())
            corrupted['protocol_hash'] = 'fake'
            (d / 'candidate_ood.json').write_text(json.dumps(corrupted))
            with self.assertRaises(ValueError):
                compare_suite(*(str(d / f'{v}_{split}.json') for split in ('test', 'ood')
                                for v in ('base', 'candidate')), resamples=100)


if __name__ == '__main__':
    unittest.main()
