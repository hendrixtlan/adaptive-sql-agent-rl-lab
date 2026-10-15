"""v0.2 invariants, including negative tests against reward hacking."""
import json
from contextlib import closing
import tempfile
import unittest
from pathlib import Path

from sqlrl.adversarial import attacks, run_attack_suite
from sqlrl.cli import robust_eval, simulate_grpo
from sqlrl.curriculum import curriculum_manifest, split_audit
from sqlrl.experiment import compare_evaluations
from sqlrl.fixtures import SCHEMAS, create_connection
from sqlrl.grpo import describe_groups
from sqlrl.policy import build_messages
from sqlrl.reward import score_sql
from sqlrl.robust import score_sql_robust, verification_seeds
from sqlrl.sandbox import execute_readonly, QueryRejected
from sqlrl.tasks import load_tasks

ROOT = Path(__file__).resolve().parent.parent


class CrossSchemaTests(unittest.TestCase):
    def test_all_gold_questions_across_domains_and_mutations(self):
        for split in ('train', 'val', 'test', 'ood'):
            for task in load_tasks(ROOT / f'data/{split}.jsonl'):
                with self.subTest(task=task.id):
                    self.assertTrue(score_sql_robust(task, task.oracle_sql).correct)

    def test_schema_prompt_is_domain_specific_and_gold_hidden(self):
        for split in ('train', 'ood'):
            for task in load_tasks(ROOT / f'data/{split}.jsonl'):
                prompt = json.dumps(build_messages(task))
                self.assertIn(task.database, prompt)
                self.assertNotIn('oracle_sql', prompt)
                self.assertNotIn(task.oracle_sql, prompt)

    def test_sqlite_metadata_denied_everywhere(self):
        for db in SCHEMAS:
            with self.subTest(db=db), self.assertRaises(QueryRejected):
                execute_readonly('SELECT name FROM sqlite_master', database=db)

    def test_domain_mixing_denied(self):
        with self.assertRaises(QueryRejected):
            execute_readonly('SELECT * FROM tickets', database='retail')
        with self.assertRaises(QueryRejected):
            execute_readonly('SELECT * FROM orders', database='support')

    def test_fixture_counts_change_and_no_external_data(self):
        for db, table in [('retail','orders'), ('support','tickets'), ('inventory','shipments')]:
            counts = {execute_readonly(f'SELECT COUNT(*) FROM {table}', database=db, seed=s).rows
                      for s in (11, 23, 44)}
            self.assertGreaterEqual(len(counts), 2)

    def test_no_variability_of_schema_identity_with_seed(self):
        for db in SCHEMAS:
            with closing(create_connection(db, 111)) as conn:
                self.assertGreater(len(conn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()), 0)


class RobustRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_tasks(ROOT / 'data/train.jsonl')[0]
        cls.support_task = load_tasks(ROOT / 'data/train.jsonl')[-10]

    def test_seeds_deterministic_distinct_and_bounded(self):
        seeds = verification_seeds(self.task, 5)
        self.assertEqual(seeds, verification_seeds(self.task, 5))
        self.assertEqual(len(set(seeds)), 5)
        with self.assertRaises(ValueError):
            verification_seeds(self.task, 0)
        with self.assertRaises(ValueError):
            verification_seeds(self.task, 13)

    def test_oracle_scores_full(self):
        for task in [self.task, self.support_task]:
            score = score_sql_robust(task, task.oracle_sql)
            self.assertEqual(score.reward, 1)
            self.assertEqual(score.correct_instances, 3)

    def test_hardcoded_single_instance_fools_old_reward_not_robust(self):
        from sqlrl.sandbox import execute_readonly
        for task in [self.task, self.support_task]:
            correct_constant = execute_readonly(task.oracle_sql, seed=task.seed, database=task.database).rows[0][0]
            cheat = f'SELECT {correct_constant}'
            self.assertTrue(score_sql(task, cheat).correct)
            robust = score_sql_robust(task, cheat)
            self.assertFalse(robust.correct)
            self.assertLess(robust.reward, 1)

    def test_unsafe_fails_closed(self):
        for sql in ['DELETE FROM orders', 'SELECT sql FROM sqlite_master',
                    'SELECT 1; UPDATE orders SET status=\'x\'']:
            with self.subTest(sql=sql):
                outcome = score_sql_robust(self.task, sql)
                self.assertFalse(outcome.safe)
                self.assertEqual(outcome.reward, -1)


class CurriculumAndSecurityTests(unittest.TestCase):
    def test_schema_holdout_and_duplicates(self):
        report = split_audit({s: ROOT / f'data/{s}.jsonl' for s in ('train','val','test','ood')})
        self.assertTrue(report['ok'], report)
        self.assertEqual(set(report['schemas']['ood']), {'inventory'})
        self.assertNotIn('inventory', report['schemas']['train'])

    def test_detect_leakage_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'train.jsonl', Path(tmp)/'ood.jsonl'
            row = (ROOT / 'data/train.jsonl').read_text().splitlines()[0]
            a.write_text(row+'\n')
            b.write_text(row+'\n')
            report = split_audit({'train': a, 'ood': b})
            self.assertFalse(report['ok'])
            self.assertGreaterEqual(len(report['collisions']), 1)

    def test_curriculum_is_deterministic_and_grows(self):
        tasks = load_tasks(ROOT/'data/train.jsonl')
        a = curriculum_manifest(tasks)
        self.assertEqual(a, curriculum_manifest(tasks))
        lengths = [len(stage['task_ids']) for stage in a['stages']]
        self.assertEqual(lengths, sorted(lengths))
        self.assertEqual(lengths[-1], len(tasks))

    def test_adversarial_bypasses_zero(self):
        result = run_attack_suite(load_tasks(ROOT/'data/ood.jsonl')[:3])
        self.assertEqual(result['guard_bypasses'], 0)
        self.assertGreater(result['n'], 0)

    def test_group_diagnostics_warn_on_flat_rewards(self):
        result = describe_groups([[1,1,1,1], [-1,.05,1,1]])
        self.assertEqual(result['zero_variance'], 1)
        self.assertEqual(result['zero_variance_rate'], .5)


class ExperimentalGateTests(unittest.TestCase):
    def test_cpu_ood_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'eval.json')
            payload = robust_eval(str(ROOT/'data/ood.jsonl'), path)
            self.assertEqual(payload['n'], 10)
            self.assertEqual(payload['evaluation_protocol'], 'three_seed_strict')
            self.assertIn('inventory', payload['by_domain'])

    def test_paired_bootstrap_rejects_cheating_with_different_task_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/'a.json', Path(tmp)/'b.json'
            base = {'results':[{'task_id':'h1','score':{'correct':False}}],
                    'split_name':'test','evaluation_protocol':'three_seed_strict', 'dataset_fingerprint':'same'}
            a.write_text(json.dumps(base))
            base['results'][0]['task_id']='wrong-id'
            b.write_text(json.dumps(base))
            with self.assertRaises(ValueError):
                compare_evaluations(str(a), str(b))

    def test_no_promotion_on_train_or_non_significant_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/'a.json', Path(tmp)/'b.json'
            def payload(split, correct):
                return {'split_name':split,'evaluation_protocol':'three_seed_strict',
                        'dataset_fingerprint':'same',
                        'results':[{'task_id':f'h{i}', 'score':{'correct':v}} for i,v in enumerate(correct)]}
            a.write_text(json.dumps(payload('train',[False]*20)))
            b.write_text(json.dumps(payload('train',[True]*20)))
            self.assertFalse(compare_evaluations(str(a),str(b))['promotion_gate_passed'])
            a.write_text(json.dumps(payload('test',[False]*20)))
            b.write_text(json.dumps(payload('test',[False]*20)))
            self.assertFalse(compare_evaluations(str(a),str(b))['promotion_gate_passed'])

    def test_large_heldout_gain_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/'a.json', Path(tmp)/'b.json'
            def payload(correct):
                return {'split_name':'test','evaluation_protocol':'three_seed_strict',
                        'dataset_fingerprint':'same',
                        'results':[{'task_id':f'h{i}', 'score':{'correct':v}} for i,v in enumerate(correct)]}
            a.write_text(json.dumps(payload([False]*30)))
            b.write_text(json.dumps(payload([True]*30)))
            report=compare_evaluations(str(a),str(b),resamples=200)
            self.assertTrue(report['promotion_gate_passed'])
            self.assertEqual(report['ci_95'],[1,1])

if __name__ == '__main__':
    unittest.main()

class SignalSweepTests(unittest.TestCase):
    def test_theoretical_and_observed_group_signal(self):
        from sqlrl.signal import signal_sweep
        report = signal_sweep(probabilities=(.1,.5), group_sizes=(2,8), n_groups=1000,seed=3)
        self.assertEqual(len(report['results']), 4)
        for item in report['results']:
            self.assertLess(abs(item['observed_zero_variance_rate'] -
                                item['theoretical_zero_variance_rate']), .06)
        self.assertGreater(report['results'][0]['theoretical_zero_variance_rate'],
                           report['results'][1]['theoretical_zero_variance_rate'])
