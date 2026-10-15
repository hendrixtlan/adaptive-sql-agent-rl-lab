# Security and reward-hacking threat model — v0.2

## Scoped assurance

The **only real execution target** is freshly created **synthetic in-memory SQLite**. There is no external database, no web access, and no user data. The evaluator allows SELECT/WITH SQL with explicit table allowlists per domain and a function whitelist; it denies schema-table inspection (`sqlite_master`), DDL and DML, extension functions and multiple statements. It enforces a query-length threshold, progress checks and a row cap.

This is **not a hardened untrusted-code sandbox**. SQLite's authorizer and VM progress callback are defense-in-depth controls only. In a production agent, use a separate non-root execution microservice, protected egress, seccomp/AppArmor, a CPU/memory/time quota, strong SQL parsing, read-only database credentials and audited access paths. The sample CPU-only Docker Compose improves isolation for tests but is **not** a substitute for a security-reviewed sandbox or a remotely isolated scoring service.

## Concrete reward attack model

1. **Memorization:** returning `SELECT <oracle result>` passes a single fixture but usually fails with three independently seeded datasets. The tests include a successful legacy single-seed cheat and rejection by the new verifier. This defense is probabilistic rather than complete.
2. **Cross-domain information access:** each SQL connection has a distinct domain schema; queries referencing unauthorized tables fail. `sqlite_master` is explicitly denied.
3. **DML/DDL and extension loading:** SQLite authorizer and verb checking reject changes and extension calls. Recursive resource abuse is bounded by progress callback and time.
4. **Reward leakage:** the current experimental AGL controller passes `TASK_JSON` with `oracle_sql` to the agent process. The LLM message never includes the oracle, but a future agent given Python/shell tools might read it. For real adversarial agents, split the verifier into a different trust domain and pass opaque IDs instead.
5. **Small benchmark overfitting:** 26 train and 10 test tasks cannot support broad claims. Inventory is excluded from train/validation and reserved OOD; audit split names/IDs/question normalization before launching training. Do not iterate on OOD errors during model training without retiring and replacing the held-out suite.
6. **Semantics:** bag equality is insensitive to row order; tasks using ORDER BY currently compare the selected ordered subset but not the total ordering of equivalent rows. For production use task-specific assertions, tolerance policies, and null handling.

The offline OOD probe suite includes stacked statements, DML, metadata reads, functions, recursive queries and (where possible) a hardcoded scalar. Zero observed allowed-query guard bypasses in this finite suite is NOT proof of security.

## Audit and evaluation commands

```bash
python -m sqlrl.cli audit-splits
python -m sqlrl.cli attack-suite --data data/ood.jsonl
python -m sqlrl.cli robust-eval --data data/ood.jsonl
```

See [evaluation.md](evaluation.md) for statistical caveats and `VALIDATION.md` for reproducible build observations.
