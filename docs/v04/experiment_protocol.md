# v0.4 experiment protocol — signal-aware rollout collection

## Claim under test

**Research hypothesis (not proven):** Given the same total generation effort, targeted rollout allocation or verl's native reward-variance group filtering may increase the quantity of useful GRPO updates and improve held-out SQL accuracy relative to fixed-n sampling without increasing unsafe-query incidence or cost per correct query.

**CPU experiment:** Measures *reward diversity, not gradients*. All policies consume the same deterministic (task, attempt) reward tape. It simulates an imbalanced population of tasks and deliberately includes two sensitivity regimes: mostly medium-difficulty and near-impossible. It reports coverage and rollouts consumed. The CPU adaptive pilot-then-expand scheduler returns **variable-length** groups, so it cannot be passed unchanged to verl's fixed `rollout.n` training code. The existing GPU GRPO entrypoint only exposes **upstream verl group filtering** as an optional configuration. There is no live adaptive sampler hook.

## Arms and fairness constraints

| Arm | CPU | GPU | Description |
|---|---|---|---|
| Fixed-4 | yes | yes (`--group-size 4`) | Constant 4 completions per query; paired common random numbers in CPU experiment |
| Fixed-8 | yes | possible (`--group-size 8`) | Same rollout budget, half as many covered tasks; a trade-off, not a direct accuracy claim |
| Adaptive 2→8 | **only** | **no** | Two pilot samples, then round-robin expansion of flat groups until reward differs or cap reaches 8 |
| Native verl filter | no CPU equivalent | **configurable, unverified** | `algorithm.filter_groups` with `metric=reward`, bounded `max_num_gen_batches` |

CPU comparison always reports *coverage*, *used/unused rollout budget*, *informative groups*, *informative groups per 100 rollouts*, *groups with an exact-reward success contrast*, *groups containing only unsafe-vs-wrong contrasts*, and *unsafe rate*. The latter two are crucial: variance generated solely by unsafe SQL is not proof of improved semantic accuracy. Adaptive sampling is not assumed to dominate under all difficulty regimes.

Paired analysis uses identical task probability profiles and attempt-indexed draws for all arms. Deterministic SHA-256 seeds make generated tapes independent of sample request order. Groups cannot inspect future attempts. This design compares **sampling efficiency** under a frozen stochastic model, but does not estimate how a learning policy changes its own future response distribution.

## Recorded model rewards

`python training/replay_v04.py --input ...` supports a strict JSONL interface for rewards previously collected from **one frozen model checkpoint** with **one decoding configuration**. All records must be train split, with 8 attempts per task by default. `policy_id` and `decode_id` must not change across the file. Missing attempts, duplicate attempts, invalid scalars, and additional keys (including gold SQL) cause validation failures rather than silently filled counterfactual outputs.

Example record (illustrative only):

```json
{"task_id":"v3_train_r_status_band_02","family":"retail_status","split":"train","attempt":0,"reward":0.0,"policy_id":"sha256-of-frozen-model-weights","decode_id":"temp=0.8;top_p=0.9;max_tokens=512"}
```

The replay source is **not** connected directly to AGL logs in this version. The operator must capture task-level rewards and map them into the contract without exposing private gold, verifier keys, or sample SQL. Recording eight completions for every task has a real compute cost; the offline replay does not eliminate that cost. Replaying the same answers cannot measure how a policy would change under different training.

## GPU experiment — blocked until compatible stack is installed

1. Pin the exact Agent Lightning 1.x, verl, vLLM, PyTorch, CUDA, tokenizer and model revisions and record the hashes in your experiment registry; follow upstream setup instructions.
2. Capture environment provenance: `python training/environment_manifest.py` (never stores secrets or claims to hash remote model weights).
3. Run separate *from-identical-checkpoint* `train_agl_v1.py` experiments with fixed four and native group filtering. Seed each training run independently, but hold the evaluation protocol and model initialization invariant.
4. For native filtering use `--native-group-filter --max-gen-batches 4`. This maps to upstream `algorithm.filter_groups` and is not equivalent to pilot→expand adaptive sampling. Monitor actual generated groups, wasted groups, model tokens, wall-clock and GPU hours; maximum generation *batches* is not a hard token budget.
5. Evaluate frozen checkpoints on `test` and `ood` using `sqlrl.eval_v03`. Never use test/OOD oracle gold in any training worker.
6. A release claim requires positive held-out lift relative to base and fixed GRPO, no extra unsafe SQL, matched compute reporting, and replication across multiple training seeds. Negative or inconclusive results must be published too.

## Threats to validity

- The synthetic Bernoulli reward generator is deliberately simplistic. Real models have structured, correlated errors and graded rewards.
- Diverse group rewards can come from *unsafe-vs-wrong* contrasts, not correct SQL. Distinguish correctness signals explicitly.
- Reusing a fixed candidate tape is off-policy for a learner whose weights change. Don't splice trajectories from different policy versions.
- Dataset-level independence is limited by only 16 SQL template families in v0.3. Task-level confidence intervals can underestimate uncertainty.
- The SQLite sandbox and local process separation are research components, not hardened production isolation.
- Group filtering can induce selection bias and increase generation spend. Compare final checkpoint effectiveness **per GPU-hour** and **per token**, not just the count of retained rollouts.
