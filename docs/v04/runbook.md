# v0.4 runbook

## CPU (Python 3.11+, zero third-party dependencies)

```bash
python -m pip install -e .
python -W error::ResourceWarning -m unittest discover -s tests -q
python training/sweep_v04.py --scenario imbalanced --repetitions 100
python training/sweep_v04.py --scenario uniform_mid --repetitions 100 --output reports/v04_cpu/uniform_mid.json
python training/sweep_v04.py --scenario mostly_wrong --repetitions 100 --output reports/v04_cpu/mostly_wrong.json
python training/environment_manifest.py
python training/train_agl_v1.py --dry-run --native-group-filter --max-gen-batches 4
```

## Recorded reward replay (when real model samples exist)

```bash
python training/replay_v04.py --input /secure-workspace/real-frozen-policy-rewards.jsonl \
  --output reports/v04_replay/actual_model_collection.json
```

The file must contain exactly these keys per JSONL record: `task_id`, `family`, `split`, `attempt`, `reward`, `policy_id`, `decode_id`. `split` must be `train`. For the default max=8, every task must have attempts 0 through 7 (no gaps). All samples must originate from the **same frozen policy** with the same decoding setup and evaluator version. The tool validates policy/decode identifiers, not whether the supplied IDs truly hash the weights. It **will not** collect additional rollouts from a model on its own.

## Experimental GPU (Linux/CUDA only; upstream stack untested here)

```bash
export AGL_KEY="$(openssl rand -hex 24)"
export SQLRL_REWARD_TOKEN="$(openssl rand -hex 32)"
export SQLRL_VERIFIER_SECRET="$(openssl rand -hex 32)"
export SQLRL_MODEL='Qwen/Qwen2.5-1.5B-Instruct'

# Arm A: fixed n=4 (save to own experiment directory)
bash training/run_gpu_local.sh --stage reasoning --group-size 4

# Arm B: from identical initial weights, separately, native verl filtering
bash training/run_gpu_local.sh --stage reasoning --group-size 4 \
  --native-group-filter --max-gen-batches 4
```

**Do not run these sequentially as if B continued training A.** Run each from the same initial checkpoint, with separate logs and outputs and a full reproducible experiment record. This launcher does not itself enforce clean experiment directories, deterministic training seeds, model revision pinning or a total inference-token cap. Validate the end-to-end trainer and generated sample counts before making any scientific claim.

Upstream: [Agent Lightning v1 trainer config](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/) and [verl group filtering](https://github.com/verl-project/verl/blob/main/verl/trainer/config/ppo_trainer.yaml).
