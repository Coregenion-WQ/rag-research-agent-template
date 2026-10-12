#!/usr/bin/env bash
# The HTTP API with curl. Start it first: docker compose up -d --build
set -euo pipefail

API="${RAG_API_URL:-http://localhost:8000}"
AUTH=()
if [[ -n "${RAG_API_KEY:-}" ]]; then AUTH=(-H "X-API-Key: ${RAG_API_KEY}"); fi

echo "== health"
curl -s "$API/health"
echo

echo "== index two documents (the second with an explicit id, so it can be replaced later)"
curl -s ${AUTH[@]+"${AUTH[@]}"} -X POST "$API/index" -H 'Content-Type: application/json' -d '{
  "documents": [
    {"page_content": "LangGraph nodes are Python functions that receive the current state and return updates.", "metadata": {"source": "notes"}},
    {"page_content": "Edges decide which node runs next. Conditional edges pick the next node with a function.", "id": "edges-overview"}
  ]
}'
echo

echo "== ask a question"
curl -s ${AUTH[@]+"${AUTH[@]}"} -X POST "$API/chat" -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is a node in LangGraph?"}]
}'
echo

echo "== follow-up: send the whole conversation"
curl -s ${AUTH[@]+"${AUTH[@]}"} -X POST "$API/chat" -H 'Content-Type: application/json' -d '{
  "messages": [
    {"role": "user", "content": "What is a node in LangGraph?"},
    {"role": "assistant", "content": "A node is a Python function that receives the state and returns updates."},
    {"role": "user", "content": "And how does the graph decide which node runs next?"}
  ]
}'
echo
