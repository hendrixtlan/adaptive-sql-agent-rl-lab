# Evaluation design and acceptance conditions — v0.2

**Datasets:** 26 train (16 retail / 10 support), 10 validation (6 retail / 4 support), 10 held-out test (6 retail / 4 support), 10 additional OOD inventory. All data are deterministic, synthetic and isolated; OOD schema is excluded from train/val. Train tasks contain basic, join and advanced query classes.

**Training:** three curriculum phases selected through `--stage foundation|joins|reasoning` (13 / 23 / 26 training tasks). The `curriculum` command also exports deterministically ordered stage manifests. Training stages are **separate experimental runs** in v0.2; checkpoint-to-checkpoint resume, sequential weight carryover and production curriculum scheduling are NOT implemented.

**Reward:** exact bag equality of actual SQL execution results across 3 fixture instances; all-seed success required for +1.0; safe partial correctness gets 0.05 + 0.65*(k/3); unsafe/rejected gets -1.0. Metric includes end-to-end exact accuracy, not only execution success. Preserve the same evaluator protocol for both model checkpoints.

**Baselines:** deterministic non-learning policy (4/26 train, 0/10 test, 0/10 OOD). The local GPU adapter has not yet been executed, so there is no model accuracy or learned lift available.

**Comparison:** paired bootstrap with fixed seed and identical ordered task IDs, fingerprint and scoring protocol; reports 95% bootstrap percentile interval, improvements, regressions, unsafe counts. A promotion flag is enabled only for held-out test/OOD evidence, sufficient positive lower CI and no added unsafe candidates. A pass is a statistical *research* gate, not deployment authorization.

**Experimental design:** run frozen base, SFT-only and GRPO checkpoints on unseen questions, held-out seeds, and ideally unseen schema names; evaluate multiple independent training seeds and identical inference settings. Add hundreds of independent tasks to make percentile uncertainty credible. Never claim cross-schema generalization from zero-OOD accuracy or from a small, repeatedly inspected OOD suite.

**GRPO note:** CPU `signal-sweep` assumes independent binary rewards; zero-variance theoretical rate is `p^k + (1-p)^k`. Live agent rollouts are correlated, may receive continuous rewards and create multiple trace rows per rollout; measure real reward distribution and trainer diagnostics instead of using the Bernoulli result to predict observed gradient quality.
