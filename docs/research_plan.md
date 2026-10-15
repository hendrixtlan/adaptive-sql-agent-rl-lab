# Next controlled research increment: v0.3

The most meaningful next experiment is no longer adding more frameworks; it is isolating whether **GRPO changes the policy's generalization**.

1. **Evaluator isolation:** move gold SQL and reward generation into a separate score-serving process; the rollout environment receives only opaque task IDs. Use private per-evaluation seeds, container isolation and request signing.
2. **Scale:** generate hundreds/thousands of nontrivial SQL tasks with parameterized templates, unseen schemas, data types, null edge cases, ranking semantics and varied cardinalities. Validate synthesized oracles by multiple independent checks.
3. **Real training:** run one pinned AGL v1 + verl + vLLM setup on a supported Linux/CUDA machine. Verify reward events and non-empty model traces before full GRPO epochs. Use at least 3 distinct training seeds.
4. **Controls:** frozen model, prompt-only improvement, SFT-only, GRPO-only, SFT+GRPO, and a matched token-budget comparison.
5. **Ablations:** single- vs multi-seed rewards; group size k=2/4/8; partial vs binary rewards; reward variance filtering; KL coefficient; failure feedback vs none.
6. **Measurement:** SQL execution correctness, unsafe statement rejection, truly unseen schema accuracy, oracle/candidate collision risk, tokens per solved question, inference latency and GPU-hours. Report paired uncertainty and long-run holdout.
7. **Stop conditions:** reward hacking, memorization, significant safety regression, policy collapse, cost escalation, or no repeatable held-out improvement.

The current v0.2 does **not** claim the implementation of these future experiments.
