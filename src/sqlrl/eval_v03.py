"""Frozen policy evaluation and two-axis promotion comparisons (held-out + OOD).

Strict integrity checks compare exactly the same datasets, prompts, evaluator
seed key, and decoding configuration. No model weights are updated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .experiment import compare_evaluations
from .policy import BaselinePolicy, OpenAICompatiblePolicy, SYSTEM_PROMPT
from .public_task import PublicTask
from .robust import score_sql_with_seeds
from .tasks import load_tasks


class Generator(Protocol):
    def complete(self, task: PublicTask, feedback: str | None = None) -> str: ...


def _canonical_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hidden_eval_seeds(task_id: str, database: str, task_seed: int, secret: str, n: int = 5) -> tuple[int, ...]:
    import hmac
    if len(secret) < 24:
        raise ValueError("Evaluation seed secret must be >=24 characters")
    seeds = []
    for i in range(n):
        payload = f"v03-frozen-eval:{task_id}:{database}:{task_seed}:{i}".encode()
        digest = hmac.new(secret.encode(), payload, hashlib.sha256).digest()
        seed = 1_000_000 + int.from_bytes(digest[:8], "big") % 1_000_000_000
        while seed in seeds:
            seed += 1
        seeds.append(seed)
    return tuple(seeds)


def evaluate_frozen(dataset: str | Path, policy: Generator, *,
                    checkpoint_id: str, secret: str, output: str | Path | None = None,
                    sampling_temperature: float = 0.0) -> dict:
    data_path = Path(dataset)
    if data_path.stem not in {"test", "ood"}:
        raise ValueError("Frozen evaluation restricted to test and OOD splits")
    if not checkpoint_id.strip():
        raise ValueError("Checkpoint identity required")
    if sampling_temperature != 0:
        raise ValueError("Frozen comparison requires temperature=0")
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    # Verifier key fingerprint only (no key or HMAC seeds published).
    verifier_tag = hashlib.sha256(("v03-eval-key:" + secret).encode()).hexdigest()
    contract = {"protocol_version": "v03_frozen_eval_v1", "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "sampling_temperature": sampling_temperature, "max_output_tokens": 384,
                "n_hidden_seeds": 5, "verifier_tag": verifier_tag,
                "split": data_path.stem, "dataset_sha256": data_hash,
                "scoring": "bag_equivalence_five_secret_seed"}
    rows = []
    for task in load_tasks(data_path):
        public = PublicTask.from_task(task)
        sql = policy.complete(public)
        outcome = score_sql_with_seeds(task, sql, hidden_eval_seeds(task.id, task.database, task.seed, secret))
        rows.append({"task_id": task.id, "database": task.database,
                     "candidate_sha256": hashlib.sha256(sql.encode()).hexdigest(),
                     "score": outcome.to_dict()})
    if not rows:
        raise ValueError("Cannot evaluate empty split")
    result = {"run_type": "v03_FROZEN_EVAL_NO_WEIGHT_UPDATES",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model": checkpoint_id, "checkpoint_id": checkpoint_id,
              "split_name": data_path.stem, "dataset_fingerprint": data_hash,
              "evaluation_protocol": "five_secret_seed_strict_v03",
              "protocol_contract": contract, "protocol_hash": _canonical_hash(contract),
              "n": len(rows), "accuracy": sum(r["score"]["correct"] for r in rows)/len(rows),
              "unsafe_queries": sum(not r["score"]["safe"] for r in rows),
              "results": rows}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_suite(baseline_test: str, candidate_test: str,
                  baseline_ood: str, candidate_ood: str, *,
                  resamples: int = 3000) -> dict:
    by_split = {}
    model_pair = None
    for split, baseline, candidate in (("test", baseline_test, candidate_test),
                                      ("ood", baseline_ood, candidate_ood)):
        a, b = [json.loads(Path(p).read_text()) for p in (baseline, candidate)]
        if a.get("split_name") != split or b.get("split_name") != split:
            raise ValueError(f"Expected {split} split reports")
        if not a.get("protocol_hash") or a["protocol_hash"] != b.get("protocol_hash"):
            raise ValueError("Mismatched or missing frozen protocol hashes")
        if a.get("checkpoint_id") == b.get("checkpoint_id"):
            raise ValueError("Candidate checkpoint must differ from baseline")
        pair = (a["checkpoint_id"], b["checkpoint_id"])
        if model_pair is not None and pair != model_pair:
            raise ValueError("Inconsistent checkpoint identity across splits")
        model_pair = pair
        by_split[split] = compare_evaluations(baseline, candidate, resamples=resamples)
    # Research gate: a significant improvement on family holdout and no measured
    # regression on schema OOD, with no increase in rejected/unsafe SQL in either.
    test, ood = by_split["test"], by_split["ood"]
    gate = bool(test["promotion_gate_passed"] and ood["absolute_accuracy_gain"] >= 0 and
                ood["candidate_unsafe"] <= ood["baseline_unsafe"])
    return {"run_type": "v03_two_axis_comparison", "baseline_checkpoint": model_pair[0],
            "candidate_checkpoint": model_pair[1], "test": test, "ood": ood,
            "research_gate_passed": gate,
            "limitations": "Gate is advisory only. Test+OOD task families are correlated within template families; "
                           "paired task bootstrap alone may understate uncertainty. "
                           "Compare independent training seeds and hold out additional families."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--split", choices=("test", "ood"), required=True)
    eval_parser.add_argument("--checkpoint-id", required=True)
    eval_parser.add_argument("--model", default=os.environ.get("SQLRL_MODEL", "auto"))
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--baseline", action="store_true", help="Use non-learning rule baseline, not an LLM")
    cmp_parser = sub.add_parser("compare-suite")
    for p in ("baseline-test", "candidate-test", "baseline-ood", "candidate-ood"):
        cmp_parser.add_argument("--" + p, required=True)
    cmp_parser.add_argument("--output", default="artifacts/v03_suite_comparison.json")
    args = parser.parse_args()
    if args.command == "evaluate":
        secret = os.environ.get("SQLRL_EVAL_SECRET", "")
        if len(secret) < 24:
            parser.error("Set private SQLRL_EVAL_SECRET (24+ chars) for frozen evaluation")
        if args.baseline:
            policy = BaselinePolicy()
        else:
            endpoint = os.environ.get("SQLRL_OPENAI_BASE_URL", "")
            if not endpoint:
                parser.error("Set SQLRL_OPENAI_BASE_URL for a frozen model server")
            policy = OpenAICompatiblePolicy(endpoint, os.environ.get("SQLRL_API_KEY", "unused"),
                                            model=args.model, temperature=0)
        report = evaluate_frozen(f"data/v03/{args.split}.jsonl", policy, checkpoint_id=args.checkpoint_id,
                                 secret=secret, output=args.output)
        print(json.dumps({"checkpoint": report["checkpoint_id"], "split": report["split_name"],
                          "n": report["n"], "accuracy": report["accuracy"],
                          "unsafe_queries": report["unsafe_queries"]}, indent=2))
    else:
        report = compare_suite(args.baseline_test, args.candidate_test, args.baseline_ood,
                               args.candidate_ood)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"research_gate_passed": report["research_gate_passed"],
                          "test_gain": report["test"]["absolute_accuracy_gain"],
                          "ood_gain": report["ood"]["absolute_accuracy_gain"]}, indent=2))


if __name__ == "__main__":
    main()
