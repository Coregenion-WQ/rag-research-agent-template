# Contributing

Thanks for helping improve this project. Bug reports, documentation fixes and new features are all welcome.

## Setup

```bash
git clone https://github.com/Coregenion-WQ/rag-research-agent-template.git
cd rag-research-agent-template
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Python 3.11 or newer is required. Docker is only needed for the Compose smoke test and for running Qdrant locally.

## Making a change

1. Create a branch from `main`, e.g. `feature/rerank-results` or `fix/qdrant-timeout`.
2. Make the change, with tests. Unit tests should not need network access or API keys: use the fixtures in `tests/conftest.py` (`make_fake_llm`, `fake_retriever`) and the fakes in `tests/fakes.py`.
3. Run the checks:

   ```bash
   make lint          # ruff, formatting, mypy --strict
   make coverage      # unit tests; CI fails below 95% coverage
   ```

   If you touched the vector store code, the Dockerfile or `docker-compose.yml`, also run:

   ```bash
   docker compose up -d qdrant
   QDRANT_URL=http://localhost:6333 make integration_tests
   make docker_smoke
   ```

4. Open a pull request against `main` describing what changed and how you tested it. CI runs lint, the unit tests on Python 3.11-3.13, the Qdrant integration tests and the Docker smoke test.

## Guidelines

- Keep configuration in `BaseConfiguration` / `AgentConfiguration` (graph behaviour) or `RAG_*` environment variables (API server). Connection details for vector stores come from environment variables.
- New vector store providers go in `src/shared/retrieval.py` with a `make_<name>_retriever` context manager, a `retriever_provider` value and a wiring test in `tests/unit_tests/test_vector_store_providers.py`.
- Model names use the `provider/model` form. Structured output goes through `shared.utils.make_structured_model` so provider-specific handling and fallbacks stay in one place.
- Integration tests that need a service or API key must skip cleanly when it isn't available.
- Update the README (and `.env.example` for new environment variables) when behaviour or setup changes.
