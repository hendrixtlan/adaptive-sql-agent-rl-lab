"""Leakage auditing and deterministic three-stage curriculum manifests."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .tasks import Task, load_tasks

LEVELS = {"basic": 0, "join": 1, "advanced": 2}


def normalized_question(task: Task) -> str:
    return f"{task.database}:" + re.sub(r"\W+", " ", task.question.lower()).strip()


def split_audit(paths: dict[str, str | Path]) -> dict:
    task_map = {name: load_tasks(path) for name, path in paths.items()}
    seen_ids, seen_questions = {}, {}
    collisions = []
    for split, tasks in task_map.items():
        for task in tasks:
            for value, seen, label in ((task.id, seen_ids, 'id'),
                                       (normalized_question(task), seen_questions, 'question')):
                if value in seen:
                    collisions.append({'kind': label, 'value': value, 'splits': [seen[value], split]})
                seen[value] = split
    schemas = {name: dict(Counter(t.database for t in tasks)) for name, tasks in task_map.items()}
    issues = []
    if 'train' in schemas and 'ood' in schemas:
        overlap = set(schemas['train']).intersection(schemas['ood'])
        if overlap:
            issues.append(f'OOD schemas found in train: {sorted(overlap)}')
    for name, tasks in task_map.items():
        if not tasks:
            issues.append(f'Empty split: {name}')
        if any(t.difficulty not in LEVELS for t in tasks):
            issues.append(f'Unknown difficulty in split {name}')
    return {'ok': not collisions and not issues, 'collisions': collisions, 'issues': issues,
            'counts': {name: len(t) for name, t in task_map.items()},
            'schemas': schemas}


def curriculum_manifest(tasks: list[Task], *, seed: int = 2026) -> dict:
    if not tasks:
        raise ValueError('Empty curriculum')
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for level in LEVELS:
        subset = [t.id for t in tasks if t.difficulty == level]
        rng.shuffle(subset)
        groups[level] = subset
    if len(set(t.id for t in tasks)) != len(tasks):
        raise ValueError('Duplicate IDs')
    manifest = {
        'kind': 'training_curriculum_NOT_MODEL_TRAINING',
        'seed': seed,
        'stages': [
            {'name': 'foundation', 'task_ids': groups['basic']},
            {'name': 'joins', 'task_ids': groups['basic'] + groups['join']},
            {'name': 'reasoning', 'task_ids': groups['basic'] + groups['join'] + groups['advanced']},
        ],
    }
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest
