# v0.3 Validation Record — CPU only

## Validated inputs

- Python 3.13 offline standard library environment; no NVIDIA driver, CUDA runtime, Agent Lightning package, verl or TRL installed.
- `PYTHONPATH=src:. python -W error::ResourceWarning -m unittest discover -s tests -q`: **50 tests passed**, 0 failures (including 12 new v0.3 checks).
- `PYTHONPATH=src python -m sqlrl.v03_benchmark`: generated 144 train, 48 val, 48 test (unseen query families), 60 OOD (unseen inventory schema); split audit succeeded and manifest SHA-256 recorded.
- Trusted SQL for all **300 generated tasks** passed equivalence on **five secret-derived fixture seeds** each (1,500 reference-query evaluations; each comparison runs both oracle and candidate reference SQL).
- `PYTHONPATH=src:. python training/train_sft.py`: CPU dry-run, 144 train and 48 validation samples; no optimization.
- `PYTHONPATH=src:. python training/train_agl_v1.py --dry-run --stage [foundation|joins|reasoning]`: 72/108/144 training tasks; proper public-only payload and AGL v1 class config reported; no GPU optimization.
- Authenticated local HTTP server: reward full match +1; wrong answer <1; rejected write -1; unauthorized request 401; heldout/test task 404; only scalar reward returned. End-to-end mocked rollout emitted scalar +1 to an AGL event stub.
- `bash -n training/run_gpu_local.sh`: shell syntax checked; **GPU launcher was not run**.
- Frozen deterministic non-learning rule policy on the new 5-seed hidden verifier: **0/48 test accuracy, 24 rejected/unsafe SQL**, and **0/60 OOD accuracy, 60 unsafe SQL**. Very weak rule baseline, NOT a pretrained LM result.
- TRL, vLLM, AGL Gateway and verl training: **not importable/executed in this environment**; no checkpoint created and no learning claim.

## Reproduction

```bash
PYTHONPATH=src:. python -W error::ResourceWarning -m unittest discover -s tests -q
PYTHONPATH=src python -m sqlrl.v03_benchmark
PYTHONPATH=src:. python training/train_sft.py
PYTHONPATH=src:. python training/train_agl_v1.py --dry-run --stage reasoning
# Use a fresh secret for disposable CPU validation only (not for real model comparisons):
export SQLRL_EVAL_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
PYTHONPATH=src python -m sqlrl.eval_v03 evaluate --split test \
  --checkpoint-id cpu-rule-baseline --baseline --output artifacts/v03_rule_test.json
PYTHONPATH=src python -m sqlrl.eval_v03 evaluate --split ood \
  --checkpoint-id cpu-rule-baseline --baseline --output artifacts/v03_rule_ood.json
```

Test labels and reference queries are never included in public AGL rollout payloads or passed to the model. They are stored in local benchmark files; a hostile process with read access to the repository could still access them. This is a **research prototype, not secure sandbox or proof of RL generalization**.
