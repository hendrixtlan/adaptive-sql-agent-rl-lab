"""Real supervised-fine-tuning control arm (TRL, requires separate GPU environment).

The dataset construction and dry-run are CPU/stdlib; calling --train-model invokes
TRL/transformers to optimize a real LM on TRAIN only. Never use test/OOD gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sqlrl.fixtures import schema_for
from sqlrl.policy import SYSTEM_PROMPT
from sqlrl.tasks import load_tasks


def sft_samples(dataset: str | Path) -> list[dict]:
    if Path(dataset).stem not in {"train", "val"}:
        raise ValueError("SFT labels permitted only for train/val")
    return [{"prompt": [
                {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema_for(task.database))},
                {"role": "user", "content": task.question}],
             "completion": [{"role": "assistant", "content": task.oracle_sql}]}
            for task in load_tasks(dataset)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="artifacts/sft_full_checkpoint")
    parser.add_argument("--train-data", default="data/v03/train.jsonl")
    parser.add_argument("--val-data", default="data/v03/val.jsonl")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--train-model", action="store_true", help="Really train model; default is CPU dry-run")
    args = parser.parse_args()
    train_rows = sft_samples(args.train_data)
    val_rows = sft_samples(args.val_data)
    info = {"arm": "sft_full_weights", "model": args.model, "train_examples": len(train_rows),
            "validation_examples": len(val_rows),
            "train_hash": hashlib.sha256(Path(args.train_data).read_bytes()).hexdigest(),
            "val_hash": hashlib.sha256(Path(args.val_data).read_bytes()).hexdigest(),
            "train_mode": "REAL_GPU" if args.train_model else "CPU_DRY_RUN_NO_GRADIENTS"}
    if not args.train_model:
        print(json.dumps(info, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        parser.error("Real SFT requires CUDA; CPU dry-run omitted --train-model")
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    settings = SFTConfig(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-5, max_length=1024,
        completion_only_loss=True, bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        save_strategy="epoch", logging_steps=10, report_to="none",
        seed=2026,
    )
    trainer = SFTTrainer(model=args.model, args=settings,
                         train_dataset=Dataset.from_list(train_rows),
                         eval_dataset=Dataset.from_list(val_rows))
    trainer.train()
    trainer.save_model(args.output)
    (Path(args.output) / 'SFT_RUN_METADATA.json').write_text(json.dumps(info, indent=2) + "\n")
    print('Saved a full-weights SFT checkpoint to', args.output)


if __name__ == '__main__':
    main()
