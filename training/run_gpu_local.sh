#!/usr/bin/env bash
# Experimental training entrypoint for Agent Lightning v1.0 on Linux/WSL with CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AGL_KEY:?Set an unpredictable AGL_KEY (e.g. openssl rand -hex 24)}"
export AGL_GATEWAY_URL="${AGL_GATEWAY_URL:-http://127.0.0.1:8181}"
export SQLRL_MODEL="${SQLRL_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${AGL_GATEWAY_URL##*:}"
command -v nvidia-smi >/dev/null || { echo 'NVIDIA/CUDA runtime required' >&2; exit 1; }
command -v agl-server >/dev/null || { echo 'Install upstream Agent Lightning v1.0 and verl first' >&2; exit 1; }
command -v agl-controller >/dev/null || exit 1
nvidia-smi
server_pid=''; controller_pid=''; oracle_pid=''
: "${SQLRL_REWARD_TOKEN:?Set a separate 24+ char reward token}"
: "${SQLRL_VERIFIER_SECRET:?Set a separate 24+ char private verifier key}"
export SQLRL_REWARD_URL="${SQLRL_REWARD_URL:-http://127.0.0.1:8765}"
[[ "$SQLRL_REWARD_TOKEN" != "$SQLRL_VERIFIER_SECRET" ]] || { echo 'Use independent reward/verifier keys' >&2; exit 1; }
cleanup() {
  if [[ -n "$oracle_pid" ]]; then kill "$oracle_pid" 2>/dev/null || true; fi
  if [[ -n "$controller_pid" ]]; then kill "$controller_pid" 2>/dev/null || true; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
# Gold SQL is loaded only by this separate service. Local subprocesses still
# share a host FS, so for adversarial agents use true container/VM isolation.
python -m sqlrl.oracle_service > artifacts/private-oracle.log 2>&1 &
oracle_pid=$!
# Do not pass the verifier seed key into the AGL controller or rollout subprocesses.
unset SQLRL_VERIFIER_SECRET
for i in $(seq 1 30); do
  if curl -fsS "${SQLRL_REWARD_URL}/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "${SQLRL_REWARD_URL}/healthz" >/dev/null || { echo 'Private reward oracle did not start' >&2; exit 1; }
# Ensure Ray is running as configured for the GPU environment.
ray status >/dev/null 2>&1 || ray start --head --dashboard-host=127.0.0.1
agl-server host=127.0.0.1 port="$PORT" key="$AGL_KEY" \
  default_proxy.model_name="$SQLRL_MODEL" \
  > artifacts/agl-server.log 2>&1 &
server_pid=$!
healthy=0
for i in $(seq 1 40); do
  if curl -fsS "${AGL_GATEWAY_URL}/healthz" >/dev/null 2>&1; then healthy=1; break; fi
  sleep 1
done
[[ "$healthy" == 1 ]] || { echo 'Gateway startup failed; see artifacts/agl-server.log' >&2; exit 1; }
agl-controller runner_type=local agl_server.url="$AGL_GATEWAY_URL" \
  agl_server.key="$AGL_KEY" local_runner.maximum_size=8 \
  > artifacts/agl-controller.log 2>&1 &
controller_pid=$!
python training/train_agl_v1.py "$@"
