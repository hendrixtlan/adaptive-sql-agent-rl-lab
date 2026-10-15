# v0.3 Scientific Experiment Protocol

## Pre-register the design before training

Primary hypothesis: given the same Qwen initial weights and synthetic 144-task training family pool, Agent Lightning v1 + verl GRPO can improve accuracy on structurally novel SQL families compared with the frozen base checkpoint. Secondary hypothesis: these gains extend to an unseen inventory schema without an increase in rejected SQL. Null hypotheses: no statistically identifiable improvement on either axis.

Ablation arms in `experiments/v03_ablation_matrix.json`:
1. `base`: frozen pretrained instruct model, no optimization.
2. `sft`: full-model supervised fine-tuning on train-only gold examples.
3. `grpo-from-base`: Agent Lightning rollouts + private reward + verl GRPO, initialized from base.
4. `sft-then-grpo`: initialize from SFT checkpoint, then GRPO; requires a fresh run.

The original GRPO assignment is `algorithm.adv_estimator=grpo`, `actor_rollout_ref.rollout.n=4`, `algorithm.enable_rollout_level_advantage=true` and a per-rollout mean policy loss. Reward is 1.0 for safe correct across 5 private fixtures; -1.0 for rejected SQL; intermediate for partially correct. For production scientific comparison, keep the reward policy fixed across runs and record its code revision and seed-secret fingerprint.

## Independent holdouts

- Validation (48): tune and choose checkpoints. Same 8 template families as train, shifted parameters.
- Test (48): 4 query families completely absent from train and validation, same retail/support schemas.
- OOD (60): inventory schema and four query families not used anywhere else.

The test and OOD gold SQL MUST NOT be mounted inside rollout containers. The training reward service refuses test/OOD manifests. Checkpoint evaluation is a separate post-training offline process with a different secret verifier key.

## Required recording

Track for each run: immutable source Git SHA, base model revision and weights digest, PyTorch/CUDA/driver/verl/AGL versions, GPU model/count, hyperparameters, training seed, sampled completions, reward histogram, nonzero-variance rollout share, KL to reference, optimizer steps, output truncation, AGL dropped samples, elapsed GPU hours, checkpoint location and SHA-256, service logs (excluding secrets), exact benchmark/protocol hashes.

For every frozen checkpoint, call `python -m sqlrl.eval_v03 evaluate --split test` and `--split ood` using the same heldout private secret and inference parameters. Never pick the best model by repeatedly optimizing against those test results. `compare-suite` calculates a task-paired bootstrap for the same heldout split and guards against two-axis regressions. Test-family correlations mean ordinary task-level intervals may be overconfident; follow with family-level uncertainty or genuinely independent benchmark families before paper-level claims.

## Decision rule

- Positive test accuracy delta with bootstrapped 95% lower bound > 0, minimum 2 percentage points as implemented by the v0.2 inherited gate.
- OOD accuracy delta >= 0 and no extra rejected/unsafe SQL on either holdout.
- Multi-seed consistency and cost/latency review required separately; automated research gate does not replace them.
- If a treatment improves the validation pool but fails test, treat as possible memorization or policy overfitting, not success.
- If all GRPO group rewards are the same, investigate collection/group sizes rather than asserting that changing the group size guarantees improvement.

## Known validity limits

Only 16 hand-authored template families. Across individual templates, parameterized instances share structure: 300 row-level tasks do not imply 300 independent ideas. SQLite bag-based comparison does not reward ordering, supports only allowed SQL functions, and cannot prove semantic equivalence on arbitrary databases. Inference may remain nondeterministic even at temperature zero on certain GPU hardware. The checkpoint label is operator-supplied; log weight digests outside the repo.
