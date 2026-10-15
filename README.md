# Adaptive SQL Agent — v0.3: Sealed-Oracle Agentic RL Lab

**Agent Lightning v1.0 · verl / GRPO · Hugging Face TRL SFT · Python 3.11+ · SQLite**

> **Verification status:** 50 CPU tests pass; 300 synthetic SQL tasks were generated, their trusted reference SQL executed correctly against five distinct hidden fixture seeds, the authenticated local reward service and the public-rollout integration were tested, and a weak rule baseline was evaluated. **NO CUDA / Agent Lightning / verl model training was performed in this environment; no trained model or RL lift is claimed.** The GPU integration remains experimental and requires upstream dependency validation.

v0.3 turns the previous 56-task adversarial prototype into a reproducible **Base vs SFT vs GRPO** experiment design, with a separated reward process, a 300-task synthetic benchmark, and frozen evaluation on *held-out query families* and *unseen inventory schema*. The v0.2 code, 56 original tasks, and old tests are kept for reproducibility.

## Why this is the next level

| | v0.2 | v0.3 |
|---|---|---|
| Synthetic SQL tasks | 56 | **300** |
| Holdouts | 10 test / 10 OOD | **48 unseen-query-family test / 60 inventory OOD** |
| Gold data in AGL rollout payload | **Yes** (`oracle_sql`) | **No** (`id`, `question`, `database` only) |
| Reward | Three deterministic fixtures in agent process | **Five HMAC-derived, evaluator-private fixtures via authenticated HTTP service** |
| Control groups | Weak rule policy | **Rule policy + SFT training entrypoint + AGL GRPO entrypoint** |
| Frozen evaluation | Single paired comparison | **Two-axis protocol-locked comparison, secret eval verifier** |
| CPU checks | 38 tests | **50 tests, HTTP integration included** |
| Actual model weight updates | Not run | **Not run** — CUDA stack necessary |

## Directory map

```text
src/sqlrl/
  v03_benchmark.py       Deterministic 300-task generator with template-family holdout
  public_task.py         Strict agent task DTO with no gold SQL/seed
  oracle_service.py      Trusted train/val reward service, private gold, HMAC verifier seeds
  oracle_client.py       Authenticated HTTP reward client (scalar only)
  rollout_agent.py       Agent Lightning local-controller entrypoint (one model request, one reward)
  agl_events.py          Agent Lightning event publisher
  eval_v03.py            Immutable-prompt, hidden-seed frozen evaluation and dual gate
training/
  train_sft.py           Experimental TRL full-model SFT control arm, dry-run by default
  train_agl_v1.py        Agent Lightning v1 + verl / GRPO integration (dry-run first)
  run_gpu_local.sh       Experimental CUDA launcher, starts private oracle, AGL gateway, controller
experiments/
  v03_ablation_matrix.json   Four frozen checkpoints × three training seeds, precommitted arms
data/v03/
  train.jsonl            144 tasks, 8 SQL families; labeled evaluator input only
  val.jsonl              48 tasks, same families but shifted parameters
  test.jsonl             48 tasks, 4 entirely new SQL families
  ood.jsonl              60 tasks, 4 new inventory query families
  MANIFEST.json          Dataset hashes, declared splits, leakage audit
```

## Run the verified CPU path (stdlib; no network access / no GPU)

From this repository root:

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
PYTHONPATH=src python -m sqlrl.v03_benchmark
PYTHONPATH=src:. python training/train_sft.py
PYTHONPATH=src:. python training/train_agl_v1.py --dry-run --stage reasoning
```

If you prefer installing the package, create a virtual environment and install in editable mode with `pip install -e .` **in a Python environment with local build dependencies available**. The above `PYTHONPATH` commands work without installing any packages in the current verification environment.

### Local private reward server (trusted internal process)

```bash
export SQLRL_REWARD_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export SQLRL_VERIFIER_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
PYTHONPATH=src python -m sqlrl.oracle_service
# Server listens ONLY on 127.0.0.1:8765. In a second shell:
curl -sS http://127.0.0.1:8765/healthz
```

The scorer loads only the `train` and `val` gold files and **refuses to load `test` or `ood`**. Agent rollouts receive just a task ID, question and database; they never need oracle SQL. A separate verifier secret determines the five hidden scoring seeds. The reward HTTP response includes only `task_id` and a numeric `reward`. The verifier secret must **not** be passed to agent processes.

**Not a hardened boundary:** local-controller subprocesses may still read files belonging to the same host user, and a token-holding client can attempt oracle probing. For adversarial or tool-executing agents, use a dedicated restricted container/pod without labeled datasets or evaluator files, enforce request quotas and network policy, and consider separate grader credentials per rollout. SQLite safety checks are not a general hostile-code sandbox. [Threat model](docs/v03/threat_model.md).

## Synthetic benchmark protocol

| Split | Tasks | Domains | Template families | Accessible to training? |
|---|---:|---|---:|---|
| Train | 144 | Retail, Support | 8 | Yes, labeled for SFT and private rewards |
| Validation | 48 | Retail, Support | Same 8, new parameters | Yes, for selection |
| Test | 48 | Retail, Support | **4 novel families** | **No** |
| OOD | 60 | Inventory | **4 novel families and schema** | **No** |

These families are synthetic templates, NOT independent natural-language tasks or human-written enterprise workloads. They are useful for falsifying weak systems, but optimistic certainty can arise from within-family correlations. Use multiple independent training seeds and a larger, separately curated human benchmark for research claims.

## Frozen model evaluation and comparison

Evaluate every frozen checkpoint separately through an OpenAI-compatible endpoint. Use the **same private `SQLRL_EVAL_SECRET`** for every checkpoint; set it independently from the training verifier key. Keep this secret off the model server/rollout environment.

```bash
export SQLRL_EVAL_SECRET="<STORE_A_PRIVATE_RANDOM_32_BYTE_KEY_OUTSIDE_THE_REPO>"
export SQLRL_OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export SQLRL_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# Base checkpoint; model served at the endpoint above
PYTHONPATH=src python -m sqlrl.eval_v03 evaluate --split test \
  --checkpoint-id base-frozen --output artifacts/base_test.json
PYTHONPATH=src python -m sqlrl.eval_v03 evaluate --split ood \
  --checkpoint-id base-frozen --output artifacts/base_ood.json

# Serve the trained checkpoint separately, then rerun with --checkpoint-id grpo-frozen
PYTHONPATH=src python -m sqlrl.eval_v03 evaluate --split test \
  --checkpoint-id grpo-frozen --output artifacts/grpo_test.json
PYTHONPATH=src python -m sqlrl.eval_v03 evaluate --split ood \
  --checkpoint-id grpo-frozen --output artifacts/grpo_ood.json

PYTHONPATH=src python -m sqlrl.eval_v03 compare-suite \
  --baseline-test artifacts/base_test.json --candidate-test artifacts/grpo_test.json \
  --baseline-ood artifacts/base_ood.json --candidate-ood artifacts/grpo_ood.json
```

The model evaluation sets `temperature=0`, `max_tokens=384`, one attempt per task and uses **five HMAC-derived secret evaluation fixtures**, distinct from training rewards. Reports include a dataset SHA-256 fingerprint, prompt fingerprint, decoding settings and verifier-key identifier. Compare-suite refuses different protocol hashes, mismatched task sets or inconsistent model labels. The checkpoint ID is currently supplied by the operator and **not cryptographically verified against weights**; record the actual weight hashes separately for any publishable result.

The research gate requires positive held-out test lift with a paired-bootstrap lower bound above zero, no OOD accuracy decrease, and no increase in unsafe SQL. **A passing gate does not constitute production approval or statistical independence.**

## Real GPU work: SFT and GRPO

The following commands **are not executed in the included CPU validation**. Microsoft’s official Agent Lightning v1 quick start uses a Linux environment with an A100; check pinned versions and VRAM before running. See [GPU runbook](docs/v03/gpu_runbook.md) and official [trainer docs](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/).

```bash
# Run in a CUDA environment with TRL, datasets, transformers, compatible torch
PYTHONPATH=src:. python training/train_sft.py --train-model \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output artifacts/sft_full_checkpoint

# Independently run AGL v1 with the installed verl/vLLM stack:
export AGL_KEY="$(openssl rand -hex 24)"
export SQLRL_REWARD_TOKEN="$(openssl rand -hex 32)"
export SQLRL_VERIFIER_SECRET="$(openssl rand -hex 32)"
export SQLRL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
bash training/run_gpu_local.sh --stage reasoning
```

For **SFT→GRPO**, use the SFT full checkpoint as `SQLRL_MODEL`, then run the GRPO launcher as a separate experiment. Run each arm with three distinct training seeds and record complete lineage, checkpoints, train-time costs and evaluation outputs. Current launcher uses independent stages; it does **not** implement automatic sequential curriculum resume. The upstream v1/verl GPU glue is experimental pending a real GPU smoke test.

## Actual CPU validation snapshot

- 50 automated tests passed, including HTTP authentication, oracle split restriction, hidden-seed scoring and a mocked end-to-end AGL event emitted from the public-only agent.
- All 300 gold SQL queries produced correct results on five secret-derived fixtures; **this verifies the benchmark oracle**, not an LLM.
- The deliberately weak, **non-learning** rule baseline: 0/48 family-holdout test; 0/60 inventory OOD. It attempted unsafe SQL on 24 test tasks and 60 OOD tasks. Not a performance claim about Qwen, SFT or GRPO.
- `training/train_sft.py` and `training/train_agl_v1.py` dry-runs succeeded; **neither made any gradient updates**.

See [VALIDATION.md](VALIDATION.md) for commands and constraints, [experiment protocol](docs/v03/experiment_protocol.md), and [GPU runbook](docs/v03/gpu_runbook.md).

## Sources

- [Microsoft Agent Lightning v1.0 architecture and basics](https://microsoft.github.io/agent-lightning/stable/05-basics/)
- [Agent Lightning v1.0 trainer configuration](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/)
- [Official AGL local A100 example](https://microsoft.github.io/agent-lightning/stable/01-quick-start/)
- [verl PPO/GRPO configuration](https://github.com/verl-project/verl/blob/main/verl/trainer/config/ppo_trainer.yaml)
- [TRL SFTTrainer / completion-only loss](https://huggingface.co/docs/trl/main/en/sft_trainer)

This independently authored educational research lab is not an official Microsoft, verl or Hugging Face release.
