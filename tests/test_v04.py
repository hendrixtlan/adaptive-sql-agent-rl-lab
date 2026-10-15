"""Regression tests for rollout-budget counterfactuals and trainer config safety."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sqlrl.adaptive_sampling import (
    BudgetConfig, RewardTape, TaskProfile, build_profiles, paired_sweep, run_strategy,
)
from training.environment_manifest import environment_manifest


class TestAdaptiveSampling(unittest.TestCase):
    def setUp(self):
        self.profiles = build_profiles(48)
        self.config = BudgetConfig(budget=192, seed=23)

    def test_config_rejects_bad_sizes(self):
        for kw in ({'budget': 1}, {'budget': 8, 'initial': 1},
                   {'budget': 8, 'group_size': 1},
                   {'budget': 8, 'max_per_task': 3}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                BudgetConfig(**kw)

    def test_duplicate_profiles_fail(self):
        with self.assertRaises(ValueError):
            RewardTape([self.profiles[0], self.profiles[0]], 5)

    def test_probability_validation(self):
        with self.assertRaises(ValueError):
            TaskProfile('x', 'f', 1.2)
        with self.assertRaises(ValueError):
            TaskProfile('x', 'f', .2, -0.1)

    def test_tape_is_order_independent_and_reproducible(self):
        tape = RewardTape(self.profiles, 987)
        a = tape.sample('task-0001', 7)
        tape.sample('task-0008', 3)
        self.assertEqual(a, tape.sample('task-0001', 7))
        self.assertEqual(a, RewardTape(self.profiles[::-1], 987).sample('task-0001', 7))

    def test_unknown_task_and_negative_attempt_fail(self):
        tape = RewardTape(self.profiles, 8)
        with self.assertRaises(KeyError):
            tape.sample('absent', 0)
        with self.assertRaises(ValueError):
            tape.sample('task-0001', -1)

    def test_per_strategy_budget_caps_and_accounting(self):
        for strategy in ('fixed', 'large_fixed', 'adaptive'):
            with self.subTest(strategy=strategy):
                result = run_strategy(self.profiles, self.config, strategy)
                self.assertLessEqual(result['rollouts_used'], self.config.budget)
                self.assertEqual(result['rollouts_used'] + result['unused_rollouts'],
                                 self.config.budget)
                self.assertEqual(sum(result['rollout_lengths'].values()), result['rollouts_used'])
                self.assertEqual(result['informative_tasks'] + result['flat_tasks'],
                                 result['task_coverage'])
                self.assertTrue(all(n <= self.config.max_per_task
                                    for n in result['rollout_lengths'].values()))
                self.assertLessEqual(result['correctness_signal_tasks'], result['informative_tasks'])
                self.assertEqual(result['unsafe_without_correctness_signal_tasks'] +
                                 result['correctness_signal_tasks'], result['informative_tasks'])

    def test_fixed_n_does_not_peek_beyond_n(self):
        result = run_strategy(self.profiles, self.config, 'fixed')
        self.assertEqual(result['task_coverage'], 48)
        self.assertTrue(all(n == 4 for n in result['rollout_lengths'].values()))

    def test_adaptive_never_exceeds_max_attempts_and_keeps_pilot(self):
        result = run_strategy(self.profiles, self.config, 'adaptive')
        self.assertEqual(result['task_coverage'], 48)
        self.assertTrue(all(2 <= n <= 8 for n in result['rollout_lengths'].values()))

    def test_deterministic_strategy_result(self):
        self.assertEqual(run_strategy(self.profiles, self.config, 'adaptive'),
                         run_strategy(self.profiles, self.config, 'adaptive'))

    def test_small_budget_does_not_create_incomplete_group(self):
        config = BudgetConfig(budget=10, seed=10)
        a = run_strategy(self.profiles, config, 'fixed')
        self.assertEqual(a['rollouts_used'], 8)
        self.assertEqual(a['unused_rollouts'], 2)
        self.assertEqual(a['task_coverage'], 2)

    def test_zero_variance_when_all_rewards_identical(self):
        profiles = [TaskProfile(f'x{i}', 'all-correct', 1, 0) for i in range(5)]
        for s in ('fixed', 'adaptive'):
            result = run_strategy(profiles, BudgetConfig(budget=20), s)
            self.assertEqual(result['informative_tasks'], 0)
            self.assertEqual(result['correctness_signal_tasks'], 0)

    def test_unknown_strategy_and_scenario(self):
        with self.assertRaises(ValueError):
            run_strategy(self.profiles, self.config, 'magic')
        with self.assertRaises(ValueError):
            build_profiles(scenario='magic')

    def test_paired_sweep_reproducible(self):
        a = paired_sweep(n=24, repetitions=3)
        b = paired_sweep(n=24, repetitions=3)
        self.assertEqual(a, b)
        self.assertEqual(a['rollout_budget_each'], 96)
        self.assertEqual(a['repetitions'], 3)
        self.assertIn('paired_differences', a)

    def test_sensitivity_scenarios_execute(self):
        for s in ('imbalanced', 'uniform_mid', 'mostly_wrong'):
            with self.subTest(s=s):
                out = paired_sweep(n=24, repetitions=2, scenario=s)
                self.assertEqual(out['scenario'], s)

    def test_sweep_validates_rep_count(self):
        with self.assertRaises(ValueError):
            paired_sweep(repetitions=0)

    def test_does_not_fabricate_model_performance(self):
        output = paired_sweep(n=16, repetitions=2)
        self.assertIn('NO', output['experiment'].upper() + output['disclaimer'].upper())
        self.assertIn('not a language model', output['disclaimer'])

    def test_native_gpu_filter_dryrun_is_explicit(self):
        proc = subprocess.run([sys.executable, 'training/train_agl_v1.py', '--dry-run',
                               '--native-group-filter', '--max-gen-batches', '3'],
                              check=True, capture_output=True, text=True)
        out = json.loads(proc.stdout)
        self.assertTrue(out['native_group_filter'])
        self.assertEqual(out['max_gen_batches'], 3)
        self.assertFalse(out['adaptive_cpu_scheduler_connected'])

    def test_zero_generation_cap_rejected(self):
        proc = subprocess.run([sys.executable, 'training/train_agl_v1.py', '--dry-run',
                               '--max-gen-batches', '0'], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('must be >= 1', proc.stderr)

    def test_environment_provenance_omits_secrets(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'data.json'
            p.write_text('{"version":4}')
            os.environ['SQLRL_VERIFIER_SECRET'] = 'DO_NOT_COPY_ME'
            result = environment_manifest([str(p)])
            self.assertEqual(len(result['file_sha256'][str(p)]), 64)
            self.assertNotIn('DO_NOT_COPY_ME', json.dumps(result))
            self.assertIsNone(result['model_weights_sha256'])
            del os.environ['SQLRL_VERIFIER_SECRET']

    def test_provenance_missing_file_fails(self):
        with self.assertRaises(FileNotFoundError):
            environment_manifest(['/not/a/file/manifest.json'])


class TestRecordedReplay(unittest.TestCase):
    def write(self, rows) -> str:
        # Persist under temporary resource owned by the test; caller cleans up.
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        for row in rows:
            tmp.write(json.dumps(row) + '\n')
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    @staticmethod
    def sample_rows(tasks=4, attempts=8):
        return [dict(task_id=f'task-{t}', family='basic', split='train', attempt=a,
                     reward=1 if (a+t) % 3 == 0 else 0,
                     policy_id='weights-sha256-checkpoint-X', decode_id='temp-0.8-top-p-0.9')
                for t in range(tasks) for a in range(attempts)]

    def test_replay_is_deterministic(self):
        from sqlrl.recorded_replay import compare_recorded
        path = self.write(self.sample_rows())
        a = compare_recorded(path)
        b = compare_recorded(path)
        self.assertEqual(a, b)
        self.assertEqual(a['input']['tasks'], 4)
        self.assertGreaterEqual(a['strategies']['adaptive']['rollouts_used'], 8)
        self.assertLessEqual(a['strategies']['adaptive']['rollouts_used'], 16)
        self.assertIn('NO_TRAINING', a['experiment'])

    def test_no_partial_counterfactual_tape(self):
        from sqlrl.recorded_replay import load_reward_replay
        with self.assertRaisesRegex(ValueError, 'missing attempts'):
            load_reward_replay(self.write(self.sample_rows()[:-1]))

    def test_holdout_not_accepted(self):
        from sqlrl.recorded_replay import load_reward_replay
        rows = self.sample_rows()
        rows[0]['split'] = 'ood'
        with self.assertRaisesRegex(ValueError, 'refuses'):
            load_reward_replay(self.write(rows))

    def test_mixed_model_checkpoints_not_accepted(self):
        from sqlrl.recorded_replay import load_reward_replay
        rows = self.sample_rows()
        rows[0]['policy_id'] = 'DIFFERENT'
        with self.assertRaisesRegex(ValueError, 'one frozen'):
            load_reward_replay(self.write(rows))

    def test_missing_or_extra_gold_sql_key_not_accepted(self):
        from sqlrl.recorded_replay import load_reward_replay
        rows = self.sample_rows()
        rows[0]['gold_sql'] = 'SELECT SECRET'
        with self.assertRaisesRegex(ValueError, 'invalid keys'):
            load_reward_replay(self.write(rows))

    def test_duplicate_or_nonfinite_reward_not_accepted(self):
        from sqlrl.recorded_replay import load_reward_replay
        rows = self.sample_rows()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_reward_replay(self.write(rows + [rows[0]]))
        rows[0]['reward'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid reward'):
            load_reward_replay(self.write(rows))

    def test_cross_group_policy_version_mixture_not_accepted(self):
        from sqlrl.recorded_replay import load_reward_replay
        rows = self.sample_rows()
        rows[-1]['decode_id'] = 'different decoding'
        with self.assertRaisesRegex(ValueError, 'one frozen'):
            load_reward_replay(self.write(rows))


if __name__ == '__main__':
    unittest.main()
