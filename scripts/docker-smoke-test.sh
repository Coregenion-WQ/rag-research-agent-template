#!/usr/bin/env bash
# Smoke test for the Docker Compose stack. Needs no model API keys.
#
# Checks that the API image builds and becomes healthy, that the API container
# can reach Qdrant over the compose network, that API key auth is enforced, and
# that Qdrant data survives a container restart.
#
# Usage: scripts/docker-smoke-test.sh [--keep]   (--keep leaves the stack running)
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT="${COMPOSE_PROJECT_NAME:-rag-smoke}"
export API_PORT="${API_PORT:-18000}"
export QDRANT_PORT="${QDRANT_PORT:-16333}"
export RAG_API_KEY="smoke-test-key"
API="http://127.0.0.1:${API_PORT}"
QDRANT="http://127.0.0.1:${QDRANT_PORT}"
COLLECTION="smoke_test"
compose() { docker compose -p "$PROJECT" "$@"; }

cleanup() {
  if [[ "${1:-}" != "--keep" ]]; then
    compose down -v --remove-orphans >/dev/null 2>&1 || true
  fi
}
trap 'status=$?; [[ $status -ne 0 ]] && compose logs --tail=50; cleanup "${KEEP:-}"; exit $status' EXIT
[[ "${1:-}" == "--keep" ]] && KEEP="--keep"

wait_for() {  # wait_for <description> <command...>
  local what=$1; shift
  for _ in $(seq 1 60); do
    if "$@" >/dev/null 2>&1; then echo "ok: $what"; return 0; fi
    sleep 2
  done
  echo "timed out waiting for: $what" >&2
  return 1
}

expect_status() {  # expect_status <expected> <curl args...>
  local expected=$1; shift
  local got
  got=$(curl -s -o /dev/null -w '%{http_code}' "$@")
  if [[ "$got" != "$expected" ]]; then
    echo "expected HTTP $expected, got $got for: $*" >&2
    return 1
  fi
  echo "ok: HTTP $expected for ${*: -1}"
}

compose up -d --build --wait
wait_for "api healthy" curl -sf "$API/health"

expect_status 401 -X POST "$API/chat" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
expect_status 422 -X POST "$API/index" -H "X-API-Key: $RAG_API_KEY" \
  -H 'content-type: application/json' -d '{"documents":[]}'

# The API container resolves and reaches Qdrant through the compose network.
compose exec -T api python -c "
import os
from shared.retrieval import get_qdrant_client
client = get_qdrant_client(os.environ['QDRANT_URL'])
client.get_collections()
assert os.environ['RAG_RETRIEVER_PROVIDER'] == 'qdrant'
"
echo "ok: api container reaches qdrant"

# Data written to Qdrant survives a restart (named volume).
curl -sf -X PUT "$QDRANT/collections/$COLLECTION" -H 'content-type: application/json' \
  -d '{"vectors":{"size":4,"distance":"Cosine"}}' >/dev/null
curl -sf -X PUT "$QDRANT/collections/$COLLECTION/points?wait=true" \
  -H 'content-type: application/json' \
  -d '{"points":[{"id":1,"vector":[0.1,0.2,0.3,0.4],"payload":{"page_content":"persisted"}}]}' >/dev/null
compose restart qdrant >/dev/null
wait_for "qdrant back after restart" curl -sf "$QDRANT/readyz"
count=$(curl -sf -X POST "$QDRANT/collections/$COLLECTION/points/count" \
  -H 'content-type: application/json' -d '{"exact":true}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["count"])')
[[ "$count" == "1" ]] || { echo "expected 1 point after restart, got $count" >&2; exit 1; }
echo "ok: qdrant data persisted across restart"

echo "docker smoke test passed"
