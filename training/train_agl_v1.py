"""Real GRPO policy optimization via Agent Lightning v1.0 and verl.

Requires a separate CUDA/Linux environment following the upstream installation.
Not executed in the offline CPU-only verification environment.
"""
from __future__ import annotations

import argparse
import importlib.resources
import json
import os
from pathlib import Path


def training_rows(path: str) -> list[dict[str, str]]:
    # Keep gold SQL out of the environment metadata used by the model.
    from sqlrl.tasks import load_tasks
    from sqlrl.public_task import PublicTask
    return [
        {"id": task.id, "public_task_json": json.dumps(PublicTask.from_task(task).to_dict())}
        for task in load_tasks(path)
    ]


def build_config(model: str, gateway: str, key: str, batch: int = 8, group_size: int = 4):
    """Follow upstream Agent Lightning v1.0 run_ppo/Hydra configuration pattern."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = str(importlib.resources.files("agentlightning.verl"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base = compose(config_name="config")
    override = OmegaConf.create({
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "enable_rollout_level_advantage": True,
        },
        "data": {"train_batch_size": batch, "max_prompt_length": 2048,
                 "max_response_length": 512},
        "actor_rollout_ref": {
            "model": {"path": model, "enable_gradient_checkpointing": True,
                      "use_remove_padding": True},
            "rollout": {"n": group_size, "name": "vllm", "tensor_model_parallel_size": 1,
                        "gpu_memory_utilization": 0.6,
                        "log_prob_micro_batch_size_per_gpu": 4},
            "actor": {
                "ppo_mini_batch_size": batch * group_size,
                "ppo_micro_batch_size_per_gpu": 4,
                "optim": {"lr": 1e-6},
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "entropy_coeff": 0,
                "fsdp_config": {"param_offload": True, "optimizer_offload": True},
                "policy_loss": {"loss_mode": "per_rollout_mean"},
            },
            "ref": {"log_prob_micro_batch_size_per_gpu": 4,
                    "fsdp_config": {"param_offload": True}},
        },
        "trainer": {"n_gpus_per_node": 1, "nnodes": 1,
                    "project_name": "adaptive_sql_rl_lab", "experiment_name": "sql_grpo_v03_oracle_separated",
                    "logger": ["console"], "val_before_train": True,
                    "save_freq": 10, "test_freq": 5, "total_epochs": 2},
        "agentlightning": {
            "agl_base_url": gateway, "agl_key": key,
            "rollout_timeout_seconds": 120,
            "reward_fillna_value": -1.0,
            "max_ppo_update_times": 2,
            "trace_aggregator": {"level": "trajectory",
                                 "trajectory_max_prompt_length": 2048,
                                 "trajectory_max_response_length": 2048},
            "local": {"agent_class": "sqlrl.rollout_agent.OracleSeparatedAgent",
                      "env_map": {"PUBLIC_TASK_JSON": "input.public_task_json"}},
        },
    })
    OmegaConf.set_struct(base, False)
    return OmegaConf.merge(base, override)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/v03/train.jsonl")
    parser.add_argument("--val", default="data/v03/val.jsonl")
    parser.add_argument("--model", default=os.environ.get("SQLRL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--gateway", default=os.environ.get("AGL_GATEWAY_URL", "http://127.0.0.1:8181"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--stage", choices=("foundation", "joins", "reasoning"), default="reasoning",
                        help="Curriculum stage (stage 1 basic; stage 2 +joins; stage 3 all)")
    parser.add_argument("--dry-run", action="store_true", help="Print dataset sizes and compatibility checks; no GPU packages needed")
    args = parser.parse_args()
    if args.group_size < 2 or args.batch < 1:
        parser.error("GRPO requires --group-size >= 2 and --batch >= 1")
    from sqlrl.tasks import load_tasks
    from sqlrl.curriculum import LEVELS, curriculum_manifest
    train_tasks = load_tasks(args.train)
    allowed = {"foundation": 0, "joins": 1, "reasoning": 2}[args.stage]
    selected = {t.id for t in train_tasks if LEVELS[t.difficulty] <= allowed}
    train = [r for r in training_rows(args.train) if r["id"] in selected]
    val = training_rows(args.val)
    if not train or not val:
        parser.error("Train and validation sets must be nonempty")
    if args.dry_run:
        print(json.dumps({"model": args.model, "train_rows": len(train), "val_rows": len(val),
                          "group_size": args.group_size, "batch": args.batch,
                          "stage": args.stage, "curriculum_fingerprint": curriculum_manifest(train_tasks)["fingerprint"],
                          "reward": "five_secret_seed_verification",
                          "agent_class": "sqlrl.rollout_agent.OracleSeparatedAgent",
                          "gold_in_rollout": False, "status": "dry_run_ONLY_no_training"}, indent=2))
        return
    key = os.environ.get("AGL_KEY")
    if not key:
        parser.error("Set AGL_KEY to a strong shared key")
    reward_url = os.environ.get("SQLRL_REWARD_URL", "")
    reward_token = os.environ.get("SQLRL_REWARD_TOKEN", "")
    if not reward_url or len(reward_token) < 24:
        parser.error("Start private reward server and set SQLRL_REWARD_URL and SQLRL_REWARD_TOKEN")
    if reward_url not in {"http://127.0.0.1:8765", "http://localhost:8765"}:
        parser.error("v0.3 local experiment requires loopback reward service port 8765")
    config = build_config(args.model, args.gateway, key, args.batch, args.group_size)
    from agentlightning.verl.entrypoint import run_ppo
    run_ppo(config, train_dataset=train, val_dataset=val)


if __name__ == "__main__":
    main()
