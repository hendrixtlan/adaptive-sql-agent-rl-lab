# v0.4 Validation Record — CPU diagnostics and integration dry-runs

**Scope:** Python 3.13, standard-library-only test execution; no NVIDIA GPU, CUDA/verl GPU trainer, vLLM runtime or trained Qwen checkpoint in this session.

## Verified execution

- `PYTHONPATH=src:. python -W error::ResourceWarning -m unittest discover -s tests -q`: **77 tests passed, 0 failures** (50 inherited and 27 new v0.4 checks). Includes SQL safety/oracle tests, adaptive-budget invariants, reproducibility, exact cap/usage accounting, group attribution, replay validation and leakage rejection, native-filter CLI dry run, and provenance-secret omission.
- `PYTHONPATH=src:. python training/sweep_v04.py --scenario ... --repetitions 100` for three regimes: all finished with reproducible seeded common-random-number rewards. Each uses **144 synthetic task profiles** and a **576 maximum rollout budget** per strategy per repetition.
- `PYTHONPATH=src:. python training/replay_v04.py --input data/v04/sample_SYNTHETIC_reward_tape.jsonl`: validated a clearly labeled **synthetic** 24-task tape, each with eight pre-recorded synthetic attempts. It is not real model output.
- `PYTHONPATH=src:. python training/train_agl_v1.py --dry-run --native-group-filter --max-gen-batches 4`: verified public-only training input counts and explicit upstream filtering arguments. Actual Hydra merge and optimizer/GPU execution remain untested.
- `PYTHONPATH=src:. python training/environment_manifest.py`: wrote package version and dataset fingerprint record, without tokens or environment variables.
- `bash -n training/run_gpu_local.sh` and `python -m compileall -q src training tests` succeeded.

## Measured CPU allocation efficiency (100 deterministic seeds per regime)

All values below are **means**. Informativeness counts sampled tasks where at least two distinct rewards were observed. This is neither gradient quality nor SQL accuracy.

| Synthetic regime | Strategy | Mean rollouts used (max 576) | Mean covered tasks | Mean informative groups | Mean groups with full-correct contrast |
|---|---|---:|---:|---:|---:|
| Mixed difficulty | fixed-4 | 576.00 | 144 | 79.56 | 62.01 |
| Mixed difficulty | fixed-8 | 576.00 | 72 | 51.15 | 44.06 |
| Mixed difficulty | adaptive 2→8 | 576.00 | 144 | **92.82** | **67.75** |
| Mostly medium | fixed-4 | 576.00 | 144 | 125.56 | 124.03 |
| Mostly medium | adaptive 2→8 | 430.46 | 144 | **142.69** | **138.52** |
| Nearly impossible | fixed-4 | 576.00 | 144 | 11.87 | 9.16 |
| Nearly impossible | adaptive 2→8 | 576.00 | 144 | **12.15** | **9.29** |

For the mixed-difficulty regime, adaptive vs fixed-4 produced an average difference of **+13.26 informative tasks** (100/100 paired synthetic runs higher; descriptive simulation 95% empirical difference range +7 to +20). In the nearly-impossible regime, the mean difference was **+0.28**, with **76/100 ties**, so adaptive gains nearly vanish. These simulation ranges are **not** confidence intervals for neural training lift. Check the JSON reports for safety-only signals, utilization and per-family differences.

## Artifacts

- `reports/v04_cpu/imbalanced.json` — 100-seed mixed-difficulty simulated ablation.
- `reports/v04_cpu/uniform_mid.json` — sensitivity simulation (medium).
- `reports/v04_cpu/mostly_wrong.json` — sensitivity simulation (near-zero correctness).
- `reports/v04_cpu/synthetic_demo_replay.json` — replay parser smoke result, synthetic rewards.
- `reports/v04_cpu/agl_fixed_dry_run.json` and `agl_native_filter_dry_run.json` — argument-level verification only.
- `reports/v04_cpu/environment.json` — non-secret environment provenance.
- `experiments/v04/ablation_matrix.json` — preregistered comparison plan, no results.

## NOT demonstrated

- **No language-model weight updates.** Not on CPU and not on GPU. No real model inference, no benchmark improvement or OOD generalization is claimed.
- **No true adaptive live sampler hook in Agent Lightning.** The CPU pilot→expand policy uses variable groups while the experimental GPU path only enables verl's existing whole-group dynamic filtering.
- **No end-to-end CUDA validation.** The native group-filter CLI maps config fields consistent with current verl documentation; it has not been exercised with a verified AGL/verl pinned stack.
- **No real model replay evidence.** The repo contains an executable parser and a *synthetic* example tape, not harvested AGL model rollouts.
- **No general sandbox security guarantee.** As in v0.3, the local Oracle provides accidental gold isolation, not a malicious-hostile-process boundary.

The proposed next acceptance gate is **a GPU-backed, compute-matched checkpoint study (fixed-4 vs native-filter GRPO) starting from identical SFT checkpoints**, with real model trace collection, three independent training seeds and frozen held-out SQL evaluation. See `docs/v04/experiment_protocol.md`.
