.PHONY: test legacy-audit v03-data v03-sft-check v03-grpo-check v03-verify

test:
	PYTHONPATH=src:. python -W error::ResourceWarning -m unittest discover -s tests -v

legacy-audit:
	PYTHONPATH=src python -m sqlrl.cli audit-splits

v03-data:
	PYTHONPATH=src python -m sqlrl.v03_benchmark

v03-sft-check:
	PYTHONPATH=src:. python training/train_sft.py

v03-grpo-check:
	PYTHONPATH=src:. python training/train_agl_v1.py --dry-run --stage reasoning

v03-verify: test v03-data v03-sft-check v03-grpo-check
