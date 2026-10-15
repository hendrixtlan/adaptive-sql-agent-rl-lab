# v0.3 Trusted Reward Threat Model

## What is improved

Agent Lightning local-controller rollout payload now has exactly three keys: `id`, `question`, `database`. The oracle process independently opens labeled training/validation records, owns the HMAC verifier seed key, executes SQL on private synthetic SQLite connections, and returns `task_id` + `reward` only. The scorer validates bearer tokens, accepts only POST `/v1/reward`, has body-size, concurrency and per-task request caps, and by default only binds loopback. The service refuses `test.jsonl` and `ood.jsonl` manifests.

The GPU launcher removes the verifier seed secret from its environment *before* launching AGL gateway, controller and training; only the scoring process inherits it. Synthetic fixture row values and gold SQL must never be returned to the model, recorded in public task metadata, or embedded in the agent prompt. The scoring service should run under a different Unix user/container or pod, and the rollout image should not contain benchmark gold files.

## Still open; avoid false guarantees

- **Not a security boundary:** local subprocesses can read files on the same host and may find the labeled train/val datasets in the workspace. Container separation / K8s Jobs with inaccessible grader volume is needed for genuinely adversarial agent code.
- **Reward oracle probing:** agents that know a valid token can send extra candidate queries (up to limits) to infer outputs. Distinct short-lived tokens per rollout, task binding and request-budget enforcement by rollout are future requirements.
- **No network ACL outside localhost:** private loopback-only service makes the local training demo functional but does not implement cross-pod authenticated transport; do not bind port 8765 to the public internet.
- **Not proof of equivalent SQL:** five seeded fixtures reduce trivial constant-answer reward hacking, but cannot exclude all semantic mismatches and do not approximate arbitrary production databases.
- **SQLite authorizer is defense-in-depth only:** test adversarial SQL in OS/container isolation, never against production data; resource limits, network policy and process isolation are separate controls.
- **Benchmark contamination:** training oracle has no test/OOD but offline evaluators, executed by trusted experimenters only, necessarily open private labels. Model server and training processes must not inherit these files or secrets in an adversarial deployment.
