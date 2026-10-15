"""Replay a trusted train-only frozen-model rollout reward tape (strict JSONL)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from sqlrl.recorded_replay import compare_recorded


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Real frozen-policy rollout JSONL (train only)')
    p.add_argument('--output', default='reports/v04_replay/replay.json')
    p.add_argument('--budget-per-task', type=int, default=4)
    p.add_argument('--max-per-task', type=int, default=8)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    result = compare_recorded(args.input, budget_per_task=args.budget_per_task,
                              max_per_task=args.max_per_task, seed=args.seed)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'result': str(dst), 'task_count': result['input']['tasks'],
                      'warning': result['warning']}, indent=2))


if __name__ == '__main__':
    main()
