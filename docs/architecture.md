# Architecture v0.2

The model is an OpenAI-compatible SQL policy served through the Agent Lightning v1.0 API Gateway. A local Rollout Controller launches `sqlrl.agent.Agent` on each task and injects `AGL_OPENAI_BASE_URL`, `AGL_EVENT_URL`, and `AGL_KEY`. The agent builds a **schema-specific** message, produces SQL, and calls a trusted Python evaluator. The verifier checks the candidate on **three seeded in-memory SQLite instances** against private reference SQL and posts a scalar to the AGL event endpoint. AGL trace aggregation and the customized verl trainer can then calculate GRPO advantages and optimize the policy.

The CPU-only modules can run independently, with no model, cloud, or API credentials. They test the reward contract, schema isolation, splits and attacker probes. The GRPO signal sweep models zero-variance probability but makes no gradient updates. GPU training remains unvalidated in this artifact.

**Boundaries:** `oracle_sql` must never be interpolated into system/user messages. Currently it is in local AGL environment metadata for scoring. A **subsequent** hardening iteration should move the evaluator into a separate non-agent container and use an opaque `task_id` as the scoring key to avoid leakage through arbitrary agent tools.

**Controlled changes:** The first v0.2 rollout accepts only SQL text (no tool-calling shell agent). Add SQL AST validation, resource quotas, network isolation, and signed validation manifests before expanding this experiment to multi-tool agent workflows.
