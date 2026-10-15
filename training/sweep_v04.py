"""Run reproducible CPU-only rollout allocation ablation and save JSON evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlrl.adaptive_sampling import paired_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=int, default=144)
    parser.add_argument('--repetitions', type=int, default=80)
    parser.add_argument('--budget-per-task', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--scenario', choices=['imbalanced', 'uniform_mid', 'mostly_wrong'],
                        default='imbalanced')
    parser.add_argument('--output', default='reports/v04_cpu/paired_sampling.json')
    args = parser.parse_args()
    report = paired_sweep(n=args.tasks, budget_per_task=args.budget_per_task,
                          repetitions=args.repetitions, seed=args.seed,
                          scenario=args.scenario)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
