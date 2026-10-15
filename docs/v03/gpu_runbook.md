# GPU Runbook — TRL SFT and Agent Lightning v1 + verl GRPO

## Prerequisites and validation boundary

Development container has no NVIDIA driver, no vLLM, no Agent Lightning and no verl GPU installation. Thus all CUDA commands below are **instructions, not verified execution results**. Official AGL v1 local quick start describes a 1×A100 reference system; Windows native local runner is not supported, but Linux/WSL can work if drivers are compatible. Use pinned upstream Agent Lightning and matching verl setup rather than installing arbitrary latest versions into an unknown CUDA stack.

## A. Environment

Install the matching upstream stack using the official [Agent Lightning installation](https://microsoft.github.io/agent-lightning/stable/00-installation/) and [quickstart](https://microsoft.github.io/agent-lightning/stable/01-quick-start/) pages. Validate `nvidia-smi`, Ray, vLLM inference, GPU memory and import of `agentlightning.verl.entrypoint` before training. Install this package editable in the *same* Python environment (`pip install -e .`, plus dependencies for TRL SFT).

Never commit API keys or verifier secrets. Set different keys for AGL gateway, reward API, training verifier and evaluation verifier. **The launcher explicitly unsets `SQLRL_VERIFIER_SECRET` after spawning the isolated scoring process, so it is not inherited by controller/agent subprocesses**; local host filesystem access remains a known weakness.

## B. SFT control

Start with `python training/train_sft.py` in CPU dry-run mode to verify counts; then use `--train-model`. The TRL SFTTrainer path uses prompt-completion records and `completion_only_loss=True`; its source of truth is the 144 train labels, with the 48 validation labels held for monitoring. Full-model SFT is memory intensive, even for a 1.5B model. Save the full checkpoint and record actual weights hash. For SFT→GRPO, set `SQLRL_MODEL` to this model path; the launcher does NOT merge adapters, so the SFT recipe deliberately saves full weights.

## C. GRPO training

```bash
export AGL_KEY="$(openssl rand -hex 24)"
export SQLRL_REWARD_TOKEN="$(openssl rand -hex 32)"
export SQLRL_VERIFIER_SECRET="$(openssl rand -hex 32)"
export SQLRL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
bash training/run_gpu_local.sh --stage reasoning --batch 8 --group-size 4
```

The launcher starts the private oracle, Ray if needed, AGL gateway/controller and `run_ppo`. Gold SQL and private verifier seeds are in the oracle process, NOT mapped into the rollout `PUBLIC_TASK_JSON`. Training samples are public tasks with ID, question and database only. One generation is scored once then emitted as an AGL `reward` event. The trainer collects groups of four rollouts and computes group-relative policy updates if the installed AGL + verl versions support the matching Hydra configuration.

**Known runtime compatibility risk:** the Hydra config override in `training/train_agl_v1.py` follows AGL v1 docs but has NOT been verified against an installed AGL/verl build on GPU. Run a one-batch smoke, inspect traces/events and keep upstream versions pinned. The code does not implement multi-stage checkpoint resume, multi-node training or container-level grader isolation.

## D. Freeze, serve, evaluate

Serve each trained checkpoint separately using vLLM (or another compatible endpoint). Lock inference settings (`temperature=0`, 384 output tokens, identical prompts) and reserve a single `SQLRL_EVAL_SECRET` for evaluation. Evaluate test and inventory OOD only after checkpoint selection. Run compare-suite to verify dataset/protocol consistency and two-axis improvement. Do not interpret a favorable 48-task result as publishable evidence without additional human-written tasks and seed repetitions.
