.PHONY: test smoke grpo train-check robust audit attacks signal verify

smoke:
	PYTHONPATH=src python -m sqlrl.cli smoke --data data/train.jsonl

grpo:
	PYTHONPATH=src python -m sqlrl.cli simulate-grpo --data data/train.jsonl

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

train-check:
	PYTHONPATH=src python training/train_agl_v1.py --dry-run --stage reasoning

robust:
	PYTHONPATH=src python -m sqlrl.cli robust-eval --data data/test.jsonl
	PYTHONPATH=src python -m sqlrl.cli robust-eval --data data/ood.jsonl

audit:
	PYTHONPATH=src python -m sqlrl.cli audit-splits

attacks:
	PYTHONPATH=src python -m sqlrl.cli attack-suite --data data/ood.jsonl

signal:
	PYTHONPATH=src python -m sqlrl.cli signal-sweep

verify: test audit attacks signal train-check
