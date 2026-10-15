"""Capture *non-secret* environment provenance for a reproducible GPU run.

No access tokens, URLs with credentials, secret keys, or arbitrary full environment
variables are recorded. Never use the ID string in place of a real weight hash.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


PACKAGES = ('torch', 'transformers', 'datasets', 'trl', 'hydra-core', 'omegaconf',
            'agentlightning', 'verl', 'vllm', 'ray', 'openai')


def environment_manifest(paths: list[str]) -> dict:
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    checksums = {}
    for name in paths:
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(name)
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                      stderr=subprocess.DEVNULL, timeout=2).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        sha = None
    return {
        'purpose': 'environment_provenance_only_NOT_training_proof',
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'git_sha': sha, 'package_versions': packages,
        'file_sha256': checksums,
        'model_weights_sha256': None,
        'gpu': 'not_checked',
        'caution': 'Never publish access tokens, rewards seed keys or personal data in logs.',
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='reports/v04_cpu/environment.json')
    p.add_argument('--files', nargs='+', default=['data/v03/MANIFEST.json',
                                                'experiments/v03_ablation_matrix.json'])
    args = p.parse_args()
    payload = environment_manifest(args.files)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
