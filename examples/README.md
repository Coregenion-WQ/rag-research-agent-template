# Examples

| File | What it shows |
| ---- | ------------- |
| [`api_client.py`](api_client.py) | A small Python client for the HTTP API (standard library only): health check, indexing the bundled sample docs, single questions and an interactive chat that keeps the conversation |
| [`curl.sh`](curl.sh) | The same API calls with curl, including document IDs and follow-up questions |
| [`run_graphs_locally.py`](run_graphs_locally.py) | Calling the index and retrieval graphs directly from Python, with an in-process Qdrant store and cross-provider fallbacks |

## Try it end to end

```bash
cp .env.example .env            # set ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up -d --build

python examples/api_client.py index-samples
python examples/api_client.py ask "What is a node in LangGraph?"
python examples/api_client.py chat
```

If you set `RAG_API_KEY` in `.env`, export it in your shell too so the examples can send it.

`run_graphs_locally.py` needs the project installed (`pip install -e .`) but no running services. Run it from the repository root, since it indexes `src/sample_docs.json`:

```bash
python examples/run_graphs_locally.py "How do I stream tokens from a chat model?"
```
