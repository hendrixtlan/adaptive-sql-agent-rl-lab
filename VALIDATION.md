# v0.2 validation record

**Executed** in CPU-only Python 3.13.5 on 2026-10-14 (isolated working filesystem). This is reproducible from the included source; no CUDA, vLLM, Ray, Docker daemon or AGL server was executed here.

| Verification | Outcome |
|---|---|
| `PYTHONPATH=src python -W error::ResourceWarning -m unittest discover -s tests -q` | **38 passed** |
| `python -m sqlrl.cli audit-splits` | **OK**: 26 train / 10 validation / 10 test / 10 OOD; inventory only in OOD |
| `python -m sqlrl.cli robust-eval --data data/train.jsonl` | Weak rule policy **4/26** (15.38%), not a model |
| `python -m sqlrl.cli robust-eval --data data/test.jsonl` | Weak rule policy **0/10** |
| `python -m sqlrl.cli robust-eval --data data/ood.jsonl` | Weak rule policy **0/10** |
| `python -m sqlrl.cli attack-suite --data data/test.jsonl` | 66 predefined probes, **0 detected guard bypasses** |
| `python -m sqlrl.cli attack-suite --data data/ood.jsonl` | 63 predefined probes, **0 detected guard bypasses** |
| `python -m sqlrl.cli signal-sweep` | deterministic Monte Carlo group signal sweep produced |
| `python training/train_agl_v1.py --dry-run --stage foundation` | 13 train / 10 val task rows, no training |
| `python training/train_agl_v1.py --dry-run --stage joins` | 23 train / 10 val task rows, no training |
| `python training/train_agl_v1.py --dry-run --stage reasoning` | 26 train / 10 val task rows, no training |
| `python -m compileall -q src training tests` | no syntax errors |

**Example GRPO signal diagnostic:** synthetic independent Bernoulli correctness probability p=0.1, group size k=4. Empirical flat-group rate = 0.656 (2500 groups); expected formula `p**k+(1-p)**k=0.6562`. This is a signal-distribution test only.

**What is NOT validated:** training with model weights, upstream binary/API compatibility under a particular GPU stack, model evaluation with actual checkpoints, cross-schema transfer from a trained policy, Docker build/runtime, CI hosted execution, benchmark external validity, secure sandboxing, performance at scale and RL lift.

**Scope of attack result:** probes are hand-authored in `src/sqlrl/adversarial.py`; zero bypasses here does not imply no attacker can bypass SQLite controls. Do not use this system on production data or as a hostile-agent execution environment.
