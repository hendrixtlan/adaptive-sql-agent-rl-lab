# Adaptive SQL Agent — v0.4: Signal-Aware GRPO Laboratory

**Agent Lightning v1 · verl/GRPO · Sealed Oracle · Adaptive Rollout Budget Diagnostics · Frozen-Policy Replay**

> **Verified status:** 77 CPU tests pass. The v0.4 CPU experiments and the strict recorded-reward replay parser execute locally. The v0.3 authenticated reward oracle and 300-task SQL benchmark remain intact. **No NVIDIA/CUDA training or Qwen weight updates were performed in this build.** The new native verl filtering switch is an experimental configuration path, not a validated trained checkpoint. Offline variable-size adaptive rollout groups are not connected to the GPU optimizer.

## The v0.4 question

Can we collect more *informative GRPO rollout groups per unit of inference compute* by changing which tasks receive additional samples? And does this translate into robust SQL accuracy on unseen query families, or does a high reward variance merely reflect unsafe-versus-wrong completions?

For context, GRPO produces a zero relative advantage when all rewards in a group are identical. **verl already offers upstream group filtering**; we do not claim to have invented it. v0.4 adds a reproducible cost-aware pilot→expand *CPU counterfactual scheduler* and the strict replay protocol needed to evaluate it on externally recorded frozen model rollouts. It adds an experimental `--native-group-filter` switch to the existing AGL/verl configuration, but does **not** equate that switch with the variable-group CPU algorithm.

## New in this release

- **Three-arm CPU ablation:** fixed 4, fixed 8, adaptive pilot-2→8; exactly the same maximum rollout budget, deterministic reward tapes and run seeds. Reports actual usage and task coverage.
- **Three sensitivity regimes:** mixed difficulty, mostly medium difficulty, and nearly all incorrect. Document the failure case, not just favorable averages.
- **Safety-aware signal attribution:** separate diverse rewards containing a correct answer from diversity caused only by unsafe and wrong answers.
- **Frozen-policy replay:** accept complete train-only JSONL reward tapes containing one checkpoint and one decoding configuration; reject missing attempts, holdouts, secret/gold extra fields, duplicates, invalid rewards and cross-checkpoint mixing. Replay is **not** model training or on-policy inference.
- **Upstream GPU config option:** verl `algorithm.filter_groups.enable=true`, `metric=reward`, bounded `max_num_gen_batches`; inherited from verl, unverified with CUDA here.
- **Non-secret environment manifest** and preregistered matched-compute GPU ablation matrix.
- All original v0.3 SQL benchmark, oracle, agents, SFT and evaluation pipelines retained.

## Quick start (CPU; Python 3.11+)

```bash
pip install -e .
python -W error::ResourceWarning -m unittest discover -s tests -q
python training/sweep_v04.py --scenario imbalanced --repetitions 100
python training/sweep_v04.py --scenario uniform_mid --repetitions 100 --output reports/v04_cpu/uniform_mid.json
python training/sweep_v04.py --scenario mostly_wrong --repetitions 100 --output reports/v04_cpu/mostly_wrong.json
python training/train_agl_v1.py --dry-run --native-group-filter --max-gen-batches 4
```

See `reports/v04_cpu/*.json` for captured results and `VALIDATION.md` for exact claims. Try the **synthetic** replay smoke test with `python training/replay_v04.py --input data/v04/sample_SYNTHETIC_reward_tape.jsonl`. For real frozen-model recorded-rollout analysis use `python training/replay_v04.py --input YOUR_JSONL`; the contract is documented in [v0.4 runbook](docs/v04/runbook.md). The GPU integration and full research protocol are in [experiment design](docs/v04/experiment_protocol.md); no model optimization occurs in any dry run.

## Architecture

```text
                               v0.3 SEALED ORACLE BOUNDARY
 Task/question/schema → Agent Lightning gateway → SQL completion
                                        │
                                        └── private reward service (gold + HMAC seeds)
                                                       │
                                                scalar reward [-1, 1]
                                                       │
                          ┌────────────────────────────┴─────────────────────────┐
                          ▼                                                      ▼
         v0.4 CPU SIGNAL-AWARE RESEARCH                           GPU TRAINER (EXPERIMENTAL)
         ├─ frozen synthetic reward tape                          Agent Lightning v1 + verl
         ├─ fixed-4 / fixed-8 / pilot→expand                      ├─ GRPO fixed rollout.n
         ├─ actual frozen-policy tape replay                      └─ optional NATIVE filter_groups
         ├─ signal / safety / compute report                              │
         └─ NO model weight updates                                      ▼
                                                                model checkpoint → heldouts
```

**Important:** The CPU pilot→expand algorithm returns variable group sizes, whereas the GPU pipeline retains a fixed `rollout.n` and lets upstream verl filter whole groups. These are distinct research arms; an integrated, on-policy adaptive sampler would require a separately validated trainer hook, policy-version tracking and budget controls. Neither a replay nor a CPU simulation demonstrates an accuracy gain on unseen SQL schemas.

## v0.3 foundation

The 300 tasks remain split 144 train, 48 validation, 48 unseen-query-family test and 60 unseen-schema OOD. The private oracle runs safe SQL against five seed-distinct fixtures, with scalar-only authenticated reward responses and a public task DTO that excludes gold SQL. See `docs/v03/` and `data/v03/MANIFEST.json`.

## Sources and research constraints

- [Agent Lightning v1 trainer configuration](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/)
- [verl PPO/GRPO configuration](https://github.com/verl-project/verl/blob/main/verl/trainer/config/ppo_trainer.yaml)
- [verl-recipe DAPO dynamic sampling](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md)
- [TRL SFT](https://huggingface.co/docs/trl/main/en/sft_trainer)

Independent educational research project; not an official Microsoft, verl, or Hugging Face release.

