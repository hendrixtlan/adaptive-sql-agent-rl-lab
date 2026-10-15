# Adaptive SQL Agent — Adversarial Reinforcement Learning Lab

**v0.2.0 · Agent Lightning v1.0 · verl · GRPO · SQLite · Python 3.11+**

> **Status:** the multi-domain CPU environment, robust reward, split audit, adversarial probes, curriculum builder, GRPO-signal experiment and checkpoint-comparison gate were run and tested without GPUs. Agent Lightning v1 + verl training integration is implemented but **real GPU training has NOT been run**, no model checkpoint has been produced, and no RL accuracy improvement is claimed. This is a research prototype, not a hostile-code sandbox.

## What changed from v0.1

| Capability | v0.1 | v0.2 |
|---|---|---|
| Data schemas | Retail only | **Retail + support + held-out inventory** |
| Dataset | 28 tasks | **56 tasks**: 26 train / 10 val / 10 test / 10 OOD |
| Reward | Execution equivalence on 1 fixture | **Equivalence on 3 independent fixture seeds** |
| Hardcoded-answer defense | None | **Oracle result checked on changed data cardinalities** |
| SQL safety | Read-only authorizer | + **Explicit table whitelist** (e.g. deny `sqlite_master`) |
| RL curriculum | None | **Three stages**, deterministic manifest & training `--stage` option |
| GRPO diagnostics | Example advantages | + **Group-collapse probability sweep** for k=2/4/8/16 |
| Evaluation | Exact-set accuracy | **Per-domain + heldout + paired bootstrap CI + security-aware promotion gate** |
| Verification | 18 unit tests | **38 unit tests**; offline adversarial suites |
| Packaging | Python entrypoints | + **CPU-only Docker Compose** (not run in build environment) |

## 1. Quickstart (CPU, no credentials)

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m sqlrl.cli audit-splits
python -m sqlrl.cli robust-eval --data data/test.jsonl
python -m sqlrl.cli robust-eval --data data/ood.jsonl
python -m sqlrl.cli attack-suite --data data/ood.jsonl
python -m sqlrl.cli curriculum
python -m sqlrl.cli signal-sweep
python training/train_agl_v1.py --dry-run --stage foundation
```

Run unit tests in a resource-limited, network-isolated Docker Compose container (optional, **Docker path not executed during packaging**):

```bash
docker compose -f compose.cpu.yml run --build --rm cpu-validation
```

## 2. Actual CPU validation snapshot (October 14, 2026)

All figures below were obtained using this repository's **deliberately weak, non-learning rule policy**, NOT a pretrained/trained LLM:

| Dataset | Tasks | Rule-policy strict 3-seed accuracy |
|---|---:|---:|
| Training (retail + support) | 26 | 4 / 26 (15.38%) |
| Held-out test (retail + support) | 10 | 0 / 10 (0%) |
| Inventory unseen-schema OOD | 10 | 0 / 10 (0%) |

A one-seed fixed scalar can pass `score_sql(task, "SELECT <number>")` for a scalar task, while **the same candidate fails multi-seed scoring** when fixture cardinality changes. Tests explicitly demonstrate this counterexample. The offline suites observed **0 allowed-schema guard bypasses among 63 predefined OOD probes**, which is narrowly about these probes, **not a security certification**.

Validation artifacts live in `artifacts/` locally. `VALIDATION.md` records the exact commands and outcomes. The folder is Git-ignored except `.gitkeep`.

## 3. Reward contract

Candidate SQL only sees schema and question. Oracle SQL is retained by the trusted local evaluator and NOT sent as an LLM prompt. Every reward evaluation executes the candidate and private oracle on three deterministic, distinct synthetic fixtures (same schema, independently different rows/cardinalities):

```
unsafe or execution rejected on ANY instance:          -1.00
safe and correct on all 3 instances:                   +1.00
safe and correct on k < 3 instances:                    0.05 + 0.65*(k/3)
```

The original one-seed reward remains available for backward-compatible diagnostics. **Real AGL agent rollouts now submit `score_sql_robust` reward**, not the legacy reward. The 3-instance verifier is still imperfect: a hardcoded answer could coincidentally match all chosen fixtures. Further hardening requires hidden evaluator seeds, randomized schema/columns and a separate score-serving trust boundary.

## 4. Genuine model training (optional, GPU, not validated here)

Agent Lightning v1.0 uses its **API Gateway**, **local/Kubernetes Rollout Controller**, and **verl-backed Trainer**. It is not compatible with old `LitAgent` examples. The repo's GPU route sends LLM requests through `AGL_OPENAI_BASE_URL` and posts robust scalar rewards through `AGL_EVENT_URL`. The training config uses GRPO (`algorithm.adv_estimator=grpo`), group sampling (`rollout.n=4`), rollout-level advantage and per-rollout policy-loss normalization.

First dry-run all curriculum stages without installing training packages:

```bash
python training/train_agl_v1.py --dry-run --stage foundation
python training/train_agl_v1.py --dry-run --stage joins
python training/train_agl_v1.py --dry-run --stage reasoning
```

After installing upstream Agent Lightning v1.0 + the appropriate verl/CUDA stack **in Linux/WSL2 with a supported NVIDIA GPU**:

```bash
python -m pip install -e '.[train]'
export AGL_KEY="$(openssl rand -hex 24)"
export SQLRL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
bash training/run_gpu_local.sh --stage foundation
# Later explicitly start a fresh run for --stage joins and --stage reasoning.
```

**Important:** this launcher was not GPU smoke-tested in the current environment, and the default 26 training records are insufficient for claims of research-quality RL generalization. Extend the train split substantially before serious comparisons. See [`docs/gpu_training.md`](docs/gpu_training.md).

## 5. Checkpoint evaluation and promotion

Serve each **frozen** base/trained checkpoint separately via an OpenAI-compatible API with the **same** sampling settings and prompt. Run each on exactly the same manifest:

```bash
export SQLRL_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export SQLRL_MODEL=Qwen-base-checkpoint
python -m sqlrl.cli robust-model-eval --data data/test.jsonl --output artifacts/base_test.json
# Change the served checkpoint and SQLRL_MODEL to the trained checkpoint.
python -m sqlrl.cli robust-model-eval --data data/test.jsonl --output artifacts/trained_test.json
python -m sqlrl.cli compare --baseline artifacts/base_test.json --candidate artifacts/trained_test.json
```

A checkpoint comparison **rejects mismatched datasets/protocols**, calculates paired bootstrap uncertainty and checks held-out split eligibility + no increase in unsafe query rate. The approval flag is a *research gate*, not automatic deployment. A significant numerical gain on just 10 test tasks would remain fragile; evaluate many more held-out tasks and multiple training seeds.

## 6. Why study zero-variance GRPO groups?

When the group consists of independent binary reward samples with chance `p` of being correct and group size `k`, the theoretical probability that **all rewards are identical** is:

`P(zero variance) = p^k + (1-p)^k`

For `p=0.10, k=4`, about **65.62%** of groups carry no relative outcome signal. The `signal-sweep` command compares this formula with seeded Monte Carlo for group sizes 2, 4, 8 and 16. It **does not** train a network or simulate token-level policy gradients.

## Architecture

```text
                           ┌─────────────────────────────┐
training / validation ────►│ SQL agent / policy (LLM)    │
question + schema only     │ AGL OpenAI-compatible proxy │
                           └──────────────┬──────────────┘
                                          │ candidate SQL
                                          ▼
                           ┌─────────────────────────────┐
                           │ Trusted local reward code   │
                           │ 3 seeds × domain fixtures   │
                           │ Authorizer / row / VM caps  │
                           └──────────────┬──────────────┘
                                          │ robust reward event
                                          ▼
                            Agent Lightning v1 Gateway
                                          │
                             Rollout Controller / traces
                                          │
                                          ▼
                          verl / GRPO (GPU, not executed)
                                          │
                           trained checkpoint (future)
                                          │
                        ┌─────────────────┴─────────────┐
                        ▼                               ▼
                    heldout test                   inventory OOD
                        └──────────►paired bootstrap◄──┘
```

`oracle_sql` is currently still passed to the agent **process** via a trusted local environment mapping, although never to the model prompt. This is a documented threat-model limitation and is **not** suitable for adversarial tool-using agents. Separate the evaluator from the rollout process before exposing arbitrary agent tools or real data. SQLite is not a hardened sandbox.

## Main code map

- `src/sqlrl/fixtures.py`: retail, support and inventory synthetic SQLite generators.
- `src/sqlrl/sandbox.py`: SQL read-only evaluator, table whitelist and resource limits.
- `src/sqlrl/robust.py`: multi-seed scoring and deterministic verification seeds.
- `src/sqlrl/adversarial.py`: attack probes including result memorization.
- `src/sqlrl/curriculum.py`: split-leakage audit and deterministic curriculum stages.
- `src/sqlrl/grpo.py` + `signal.py`: CPU reward-group diagnostics, no gradients.
- `src/sqlrl/experiment.py`: paired comparison and conservative promotion gate.
- `training/train_agl_v1.py`: experimental AGL v1 + verl GPU adapter and curriculum flag.
- `tests/`: 38 CPU unit tests covering v0.1 and v0.2 behaviors.
- `docs/`: technical design, security model and GPU setup.

## References

- [Agent Lightning v1.0 docs](https://microsoft.github.io/agent-lightning/stable/)
- [AGL v1 trainer configuration](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/)
- [AGL v1 controller configuration](https://microsoft.github.io/agent-lightning/stable/30-controller-configuration/)
- [verl GRPO implementation](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)

This is an independently authored research lab under the MIT license, not an official Microsoft or verl repository.
