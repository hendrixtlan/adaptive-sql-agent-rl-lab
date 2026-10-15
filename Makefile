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

.PHONY: v04-sweep v04-native-filter-check v04-verify

v04-sweep:
	PYTHONPATH=src:. python training/sweep_v04.py --repetitions 100

v04-native-filter-check:
	PYTHONPATH=src:. python training/train_agl_v1.py --dry-run --native-group-filter --max-gen-batches 4

v04-verify: test v04-sweep v04-native-filter-check
