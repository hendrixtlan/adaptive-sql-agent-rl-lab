# Experimental GPU training — AGL v1.0 + verl / GRPO

**Test status:** CPU unit tests and `--dry-run` completed. No Linux/CUDA AGL+verl run was executed in this release; API config may need adjustments for the exact upstream commit and driver/CUDA versions installed in your GPU environment.

Upstream references: [AGL v1 quick start](https://microsoft.github.io/agent-lightning/stable/01-quick-start/), [trainer configuration](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/), [controller configuration](https://microsoft.github.io/agent-lightning/stable/30-controller-configuration/), [official source](https://github.com/microsoft/agent-lightning). AGL v1 was extensively refactored; do not use v0.x `LitAgent` code.

## Upstream environment preparation

Use Linux/WSL2 with a compatible NVIDIA GPU driver, CUDA and working `nvidia-smi`. Microsoft's v1 Calc-X quick start uses 1×A100 as a working reference environment (not an absolute minimum for every model). Install an upstream v1 checkout, its pinned Python environment and the verl GPU stack following its installer; upstream examples include `uv sync` followed by `bash scripts/setup_verl.sh 0.8.0 cu130` on a compatible CUDA 13 machine. Ensure Ray, vLLM, PyTorch and CUDA match the upstream version. Do NOT assume CPU pip installation alone enables verl training.

Install this lab into that same Python environment:

```bash
pip install -e '.[train]'
python training/train_agl_v1.py --dry-run --stage foundation
export AGL_KEY="$(openssl rand -hex 24)"
export SQLRL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
bash training/run_gpu_local.sh --stage foundation
```

This script starts Ray, `agl-server` and `agl-controller` with a localhost Gateway, then invokes the v1 `agentlightning.verl.entrypoint.run_ppo`. The server's `default_proxy.model_name` matches the trainer model path. `AGL_KEY` is the matching authorization secret. On native Windows `runner_type=local` is unsupported; use Linux/WSL2.

## Curriculum usage

Stage 1 = `--stage foundation` (13 basic train tasks); stage 2 = `--stage joins` (23 basic+join); stage 3 = `--stage reasoning` (all 26 train tasks). **The stages are currently independent runs**. The CLI selects tasks but does not implement checkpoint resume or progressive multi-stage weight transfer. Do not conflate three different runs with a completed curriculum learning experiment.

## AGL/verl settings

- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.rollout.n=4`
- `algorithm.enable_rollout_level_advantage=true`
- `actor_rollout_ref.actor.policy_loss.loss_mode=per_rollout_mean`
- `agentlightning.trace_aggregator.level=trajectory`
- `agentlightning.max_ppo_update_times=2`
- the agent posts **multi-seed robust** rewards through `AGL_EVENT_URL`; one scalar per rollout

Observability after running actual GPU training: record gradient updates, reward histograms, group zero-variance fraction, KL, truncation, dropped samples, W&B/console logs, sample IDs, model checkpoint hashes, throughput and total GPU hours. The CPU diagnostics cannot substitute for these logs.

## Troubleshooting

- AGL controller cannot import package: install the repo into the same Python environment as the controller.
- No gradients: verify requests pass through AGL Gateway and reward events actually arrive; inspect group variance.
- Model gateway rejects completions: `SQLRL_MODEL` and `default_proxy.model_name` must match.
- OOM: start with smaller models and limit batch/context; consult upstream supported CUDA/verl setup.
- Gold leakage: `TASK_JSON` currently contains trusted oracle SQL. It is never included in the model's prompt but is visible to the local agent process; keep the agent tool-free until evaluator isolation is implemented.
- Never evaluate checkpoint lift on train alone; compare frozen checkpoints using `robust-model-eval` and `compare` on test and OOD splits.
