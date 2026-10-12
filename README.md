# LangGraph RAG Research Agent Template

[![CI](https://github.com/Coregenion-WQ/rag-research-agent-template/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/Coregenion-WQ/rag-research-agent-template/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/Coregenion-WQ/rag-research-agent-template/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/Coregenion-WQ/rag-research-agent-template/actions/workflows/integration-tests.yml)

[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/langchain-ai/rag-research-agent-template)

This is a starter project to help you get started with developing a RAG research agent using [LangGraph](https://github.com/langchain-ai/langgraph) in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio).

![Graph view in LangGraph studio UI](./static/studio_ui.png)

On top of the LangGraph template, this repository adds what you need to run it as a service:

- **Qdrant** as a persistent, self-hostable vector store (alongside Elasticsearch, MongoDB Atlas and Pinecone)
- a **FastAPI** service (`/index`, `/chat`, `/health`) with optional API key auth
- **Claude and OpenAI** as first-class model providers, with cross-provider fallbacks
- **Docker Compose** for running the API and Qdrant together
- a test suite (100% line coverage, no API keys needed) and CI with Qdrant and Docker smoke tests

## Quickstart

Run the API with Qdrant in Docker, index the sample docs and ask a question:

```bash
git clone https://github.com/Coregenion-WQ/rag-research-agent-template.git
cd rag-research-agent-template
cp .env.example .env    # set ANTHROPIC_API_KEY and OPENAI_API_KEY (used for embeddings)
docker compose up -d --build

python examples/api_client.py index-samples
python examples/api_client.py ask "What is a node in LangGraph?"
```

More in [`examples/`](examples/). To use LangGraph Studio instead, see [Getting Started](#getting-started).

## Architecture

```mermaid
flowchart LR
    client["Your app / curl / examples"] -->|"POST /index, /chat"| api["FastAPI (rag_api)"]
    studio["LangGraph Studio"] --> graphs
    api --> graphs
    subgraph graphs["LangGraph graphs"]
        index["index graph"]
        retrieval["retrieval graph"] --> researcher["researcher subgraph"]
    end
    index -->|"embed + upsert"| vs[("Vector store<br/>Qdrant / Elastic / Mongo / Pinecone")]
    researcher -->|"similarity search"| vs
    retrieval -->|"route, plan, answer"| llm["Chat models<br/>Claude / OpenAI (+ fallbacks)"]
    researcher -->|"search queries"| llm
```

| Path | Contents |
| ---- | -------- |
| `src/index_graph/` | Graph that embeds documents and writes them to the vector store |
| `src/retrieval_graph/` | Routing, research planning and answer generation; `researcher_graph/` runs the searches |
| `src/shared/` | Configuration, retriever factory (all vector stores), model loading, document helpers |
| `src/rag_api/` | FastAPI service |
| `examples/` | API client, curl and in-process usage examples |
| `tests/unit_tests/` | Fast tests with a fake model and in-memory stores |
| `tests/integration_tests/` | Tests against real Qdrant, Elasticsearch and model APIs |

## What it does

This project has three graphs:

* an "index" graph (`src/index_graph/graph.py`)
* a "retrieval" graph (`src/retrieval_graph/graph.py`)
* a "researcher" subgraph (part of the retrieval graph) (`src/retrieval_graph/researcher_graph/graph.py`)

The index graph takes in document objects indexes them.

```json
[{ "page_content": "LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows." }]
```

If an empty list is provided (default), a list of sample documents from `src/sample_docs.json` is indexed instead. Those sample documents are based on the conceptual guides for LangChain and LangGraph.

The retrieval graph manages a chat history and responds based on the fetched documents. Specifically, it:

1. Takes a user **query** as input
2. Analyzes the query and determines how to route it:
- if the query is about "LangChain", it creates a research plan based on the user's query and passes the plan to the researcher subgraph
- if the query is ambiguous, it asks for more information
- if the query is general (unrelated to LangChain), it lets the user know
3. If the query is about "LangChain", the researcher subgraph runs for each step in the research plan, until no more steps are left:
- it first generates a list of queries based on the step
- it then retrieves the relevant documents in parallel for all queries and return the documents to the retrieval graph
4. Finally, the retrieval graph generates a response based on the retrieved documents and the conversation context

## Getting Started

Assuming you have already [installed LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download), to set up:

1. Create a `.env` file.

```bash
cp .env.example .env
```

2. Select your retriever & index, and save the access instructions to your `.env` file.

<!--
Setup instruction auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
-->

### Setup Retriever

The defaults values for `retriever_provider` are shown below:

```yaml
retriever_provider: elastic-local
```

Follow the instructions below to get set up, or pick one of the additional options.

#### Elasticsearch

Elasticsearch (as provided by Elastic) is an open source distributed search and analytics engine, scalable data store and vector database optimized for speed and relevance on production-scale workloads.

##### Setup Elasticsearch
Elasticsearch can be configured as the knowledge base provider for a retrieval agent by being deployed on Elastic Cloud (either as a hosted deployment or serverless project) or on your local environment.

**Elasticsearch Serverless**

1. Signup for a free 14 day trial with [Elasticsearch Serverless](https://cloud.elastic.co/registration?onboarding_token=search&cta=cloud-registration&tech=trial&plcmt=article%20content&pg=langchain).
2. Get the Elasticsearch URL, found on home under "Copy your connection details".
3. Create an API key found on home under "API Key".
4. Copy the URL and API key to your `.env` file created above:

```
ELASTICSEARCH_URL=<ES_URL>
ELASTICSEARCH_API_KEY=<API_KEY>
```

**Elastic Cloud**

1. Signup for a free 14 day trial with [Elastic Cloud](https://cloud.elastic.co/registration?onboarding_token=search&cta=cloud-registration&tech=trial&plcmt=article%20content&pg=langchain).
2. Get the Elasticsearch URL, found under Applications of your deployment.
3. Create an API key. See the [official elastic documentation](https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-an-api-key) for more information.
4. Copy the URL and API key to your `.env` file created above:

```
ELASTICSEARCH_URL=<ES_URL>
ELASTICSEARCH_API_KEY=<API_KEY>
```
**Local Elasticsearch (Docker)**

```
docker run \
  -p 127.0.0.1:9200:9200 \
  -d \
  --name elasticsearch \
  -e ELASTIC_PASSWORD=changeme \
  -e "discovery.type=single-node" \
  -e "xpack.security.http.ssl.enabled=false" \
  -e "xpack.license.self_generated.type=trial" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.1
```

See the [official Elastic documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html) for more information on running it locally.

Then populate the following in your `.env` file:

```
# As both Elasticsearch and LangGraph Studio runs in Docker, we need to use host.docker.internal to access.

ELASTICSEARCH_URL=http://host.docker.internal:9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=changeme
```
#### MongoDB Atlas

MongoDB Atlas is a fully-managed cloud database that includes vector search capabilities for AI-powered applications.

1. Create a free Atlas cluster:
- Go to the [MongoDB Atlas website](https://www.mongodb.com/cloud/atlas/register) and sign up for a free account.
- After logging in, create a free cluster by following the on-screen instructions.

2. Create a vector search index
- Follow the instructions at [the Mongo docs](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/)
- By default, we use the collection `langgraph_retrieval_agent.default` - create the index there
- Add an indexed filter for path `user_id`
- **IMPORTANT**: select Atlas Vector Search NOT Atlas Search when creating the index
Your final JSON editor configuration should look something like the following:

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

The exact numDimensions may differ if you select a different embedding model.

2. Set up your environment:
- In the Atlas dashboard, click on "Connect" for your cluster.
- Choose "Connect your application" and copy the provided connection string.
- Create a `.env` file in your project root if you haven't already.
- Add your MongoDB Atlas connection string to the `.env` file:

```
MONGODB_URI="mongodb+srv://username:password@your-cluster-url.mongodb.net/?retryWrites=true&w=majority&appName=your-cluster-name"
```

Replace `username`, `password`, `your-cluster-url`, and `your-cluster-name` with your actual credentials and cluster information.
#### Pinecone Serverless

Pinecone is a managed, cloud-native vector database that provides long-term memory for high-performance AI applications.

1. Sign up for a Pinecone account at [https://login.pinecone.io/login](https://login.pinecone.io/login) if you haven't already.

2. After logging in, generate an API key from the Pinecone console.

3. Create a serverless index:
   - Choose a name for your index (e.g., "example-index")
   - Set the dimension based on your embedding model (e.g., 1536 for OpenAI embeddings)
   - Select "cosine" as the metric
   - Choose "Serverless" as the index type
   - Select your preferred cloud provider and region (e.g., AWS us-east-1)

4. Once you have created your index and obtained your API key, add them to your `.env` file:

```
PINECONE_API_KEY=your-api-key
PINECONE_INDEX_NAME=your-index-name
```

#### Qdrant

[Qdrant](https://qdrant.tech/) is an open source vector database. Data is persisted to disk, so the index survives restarts.

1. Start a local instance (or create a free cluster on [Qdrant Cloud](https://cloud.qdrant.io/)):

```bash
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.19.2
```

2. Add the connection details to your `.env` file and set `retriever_provider: qdrant`:

```
QDRANT_URL=http://localhost:6333   # use http://host.docker.internal:6333 from LangGraph Studio
QDRANT_API_KEY=                    # only needed for Qdrant Cloud
QDRANT_COLLECTION_NAME=langchain_index
```

The collection is created automatically on first use, sized for the configured embedding model. Re-indexing the same document (same content, or same `id`) overwrites it instead of adding a duplicate.


### Setup Model

The defaults values for `response_model`, `query_model` are shown below:

```yaml
response_model: anthropic/claude-opus-5-5
query_model: anthropic/claude-haiku-5-5
```

The query model handles routing, research planning and search-query generation, so a fast model works well there; the response model writes the final answer. Any `provider/model` supported by LangChain works, e.g. `openai/gpt-5` or `anthropic/claude-sonnet-5-5`.

Claude and OpenAI models use their native JSON-schema mode for structured output (routing, plans and queries), so they work with Claude models that don't support forced tool calls.

Follow the instructions below to get set up, or pick one of the additional options.

#### Anthropic

To use Anthropic's chat models:

1. Sign up for an [Anthropic API key](https://console.anthropic.com/) if you haven't already.
2. Once you have your API key, add it to your `.env` file:

```
ANTHROPIC_API_KEY=your-api-key
```
#### OpenAI

To use OpenAI's chat models:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup).
2. Once you have your API key, add it to your `.env` file:
```
OPENAI_API_KEY=your-api-key
```


#### Fallback models

Both models can fall back to other models, including from another provider, when a call fails (outage, rate limit, invalid output):

```yaml
query_model: anthropic/claude-haiku-5-5
query_model_fallbacks: [openai/gpt-5-mini]
response_model: anthropic/claude-opus-5-5
response_model_fallbacks: [openai/gpt-5]
```

Fallbacks are tried in order. Set the API key for every provider you list.


### Setup Embedding Model

The defaults values for `embedding_model` are shown below:

```yaml
embedding_model: openai/text-embedding-3-small
```

Follow the instructions below to get set up, or pick one of the additional options.

#### OpenAI

To use OpenAI's embeddings:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup).
2. Once you have your API key, add it to your `.env` file:
```
OPENAI_API_KEY=your-api-key
```

#### Cohere

To use Cohere's embeddings:

1. Sign up for a [Cohere API key](https://dashboard.cohere.com/welcome/register).
2. Once you have your API key, add it to your `.env` file:

```bash
COHERE_API_KEY=your-api-key
```





<!--
End setup instructions
-->

## Using

Once you've set up your retriever and saved your model secrets, it's time to try it out! First, let's add some information to the index. Open studio, select the "indexer" graph from the dropdown in the top-left, and then add some content to chat over. You can just invoke it with an empty list (default) to index sample documents from LangChain and LangGraph documentation.

You'll know that the indexing is complete when the indexer "delete"'s the content from its graph memory (since it's been persisted in your configured storage provider).

Next, open the "retrieval_graph" using the dropdown in the top-left. Ask it questions about LangChain to confirm it can fetch the required information!

## HTTP API

Besides LangGraph Studio, the graphs can be served as a small REST API so other applications can index documents and ask questions.

```bash
pip install -e .
uvicorn --factory rag_api.app:create_app --port 8000
```

Interactive docs are served at `http://localhost:8000/docs`.

| Method | Path      | Description |
| ------ | --------- | ----------- |
| GET    | `/health` | Liveness check (no API key needed) |
| POST   | `/index`  | Add documents: `{"documents": [{"page_content": "...", "metadata": {}, "id": "optional"}]}` |
| POST   | `/chat`   | Ask a question: `{"messages": [{"role": "user", "content": "..."}]}`. Returns `answer`, `route` and the `sources` used |

The API is stateless: send the whole conversation in `messages` on every call. It is configured with environment variables:

| Variable | Description |
| -------- | ----------- |
| `RAG_API_KEY` | If set, every request except `/health` must send it in the `X-API-Key` header |
| `RAG_RETRIEVER_PROVIDER` | Vector store to use, e.g. `qdrant` |
| `RAG_EMBEDDING_MODEL` | Embedding model, e.g. `openai/text-embedding-3-small` |
| `RAG_QUERY_MODEL` / `RAG_RESPONSE_MODEL` | Chat models in `provider/model` form |
| `RAG_SEARCH_K` | Number of documents to retrieve per search query |

Unset values fall back to the graph defaults. Clients can't change these per request.

## Docker Compose

`docker-compose.yml` runs the HTTP API together with a Qdrant vector store whose data is kept in a named volume:

```bash
cp .env.example .env     # add ANTHROPIC_API_KEY / OPENAI_API_KEY, optionally RAG_API_KEY
docker compose up -d --build
curl http://localhost:8000/health
```

The API waits for Qdrant to be healthy before starting and uses it as the retriever (`RAG_RETRIEVER_PROVIDER` defaults to `qdrant` here). Both ports are bound to `127.0.0.1`; change `API_PORT` / `QDRANT_PORT` to use other host ports. Set `RAG_API_KEY` before exposing the API beyond your machine.

`docker compose down` keeps the indexed data; `docker compose down -v` deletes it.

`scripts/docker-smoke-test.sh` builds and starts the stack on separate ports, checks health, auth and Qdrant connectivity, and verifies that data survives a Qdrant restart. It needs no model API keys.

## How to customize

You can customize this retrieval agent template in several ways:

1. **Change the retriever**: You can switch between different vector stores (Elasticsearch, MongoDB, Pinecone, Qdrant) by modifying the `retriever_provider` in the configuration. Each provider has its own setup instructions in the "Getting Started" section above.

2. **Modify the embedding model**: You can change the embedding model used for document indexing and query embedding by updating the `embedding_model` in the configuration. Options include various OpenAI and Cohere models.

3. **Adjust search parameters**: Fine-tune the retrieval process by modifying the `search_kwargs` in the configuration. This allows you to control aspects like the number of documents retrieved or similarity thresholds.

4. **Customize the response generation**: You can modify the `response_system_prompt` to change how the agent formulates its responses. This allows you to adjust the agent's personality or add specific instructions for answer generation.

5. **Modify prompts**: Update the prompts used for user query routing, research planning, query generation and more in `src/retrieval_graph/prompts.py` to better suit your specific use case or to improve the agent's performance. You can also modify these directly in LangGraph Studio. For example, you can:

  * Modify system prompt for creating research plan (`research_plan_system_prompt`)
  * Modify system prompt for generating search queries based on the research plan (`generate_queries_system_prompt`)

6. **Change the language model**: Update the `response_model` in the configuration to use different language models for response generation. Options include various Claude models from Anthropic, as well as models from other providers like Fireworks AI.

7. **Extend the graph**: You can add new nodes or modify existing ones in the `src/retrieval_graph/graph.py` file to introduce additional processing steps or decision points in the agent's workflow.

8. **Add tools**: Implement tools to expand the researcher agent's capabilities beyond simple retrieval generation.

Remember to test your changes thoroughly to ensure they improve the agent's performance for your specific use case.

## Development

### Running the tests

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

make test                # unit tests, no API keys or services needed
make coverage            # unit tests with a coverage report
make lint                # ruff + mypy
make docker_smoke        # build and check the Docker Compose stack

# integration tests against real services; tests whose service or key is missing are skipped
docker compose up -d qdrant
QDRANT_URL=http://localhost:6333 make integration_tests
```

Unit tests replace the chat models with a scripted fake and use in-memory vector stores (including in-process Qdrant), so the graphs, API and examples are tested end to end without network access. CI runs lint, the unit tests on Python 3.11-3.13 (with a 95% coverage floor), the Qdrant integration tests and the Docker smoke test on every pull request. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Working in LangGraph Studio

While iterating on your graph, you can edit past state and rerun your app from past states to debug specific nodes. Local changes will be automatically applied via hot reload. Try adding an interrupt before the agent calls the researcher subgraph, updating the default system message in `src/retrieval_graph/prompts.py` to take on a persona, or adding additional nodes and edges!

Follow up requests will be appended to the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

You can find the latest (under construction) docs on [LangGraph](https://github.com/langchain-ai/langgraph) here, including examples and other references. Using those guides can help you pick the right patterns to adapt here for your use case.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates.

<!--
Configuration auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
{
  "config_schemas": {
    "indexer": {
      "type": "object",
      "properties": {
        "embedding_model": {
          "type": "string",
          "default": "openai/text-embedding-3-small",
          "description": "Name of the embedding model to use. Must be a valid embedding model name.",
          "environment": [
            {
              "value": "cohere/embed-english-light-v2.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-english-light-v3.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-english-v2.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-english-v3.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-multilingual-light-v3.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-multilingual-v2.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-multilingual-v3.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "openai/text-embedding-3-large",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/text-embedding-3-small",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/text-embedding-ada-002",
              "variables": "OPENAI_API_KEY"
            }
          ]
        },
        "retriever_provider": {
          "enum": [
            "elastic-local",
            "elastic",
            "mongodb",
            "pinecone",
            "qdrant"
          ],
          "default": "elastic-local",
          "description": "The vector store provider to use for retrieval. Options are 'elastic', 'elastic-local', 'pinecone', 'mongodb', or 'qdrant'.",
          "environment": [
            {
              "value": "elastic",
              "variables": [
                "ELASTICSEARCH_URL",
                "ELASTICSEARCH_API_KEY"
              ]
            },
            {
              "value": "elastic-local",
              "variables": [
                "ELASTICSEARCH_URL",
                "ELASTICSEARCH_USER",
                "ELASTICSEARCH_PASSWORD"
              ]
            },
            {
              "value": "mongodb",
              "variables": [
                "MONGODB_URI"
              ]
            },
            {
              "value": "pinecone",
              "variables": [
                "PINECONE_API_KEY",
                "PINECONE_INDEX_NAME"
              ]
            },
            {
              "value": "qdrant",
              "variables": [
                "QDRANT_URL"
              ]
            }
          ],
          "type": "string"
        }
      }
    },
    "retrieval_graph": {
      "type": "object",
      "properties": {
        "embedding_model": {
          "type": "string",
          "default": "openai/text-embedding-3-small",
          "description": "Name of the embedding model to use. Must be a valid embedding model name.",
          "environment": [
            {
              "value": "cohere/embed-english-light-v2.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-english-light-v3.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-english-v2.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-english-v3.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-multilingual-light-v3.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-multilingual-v2.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "cohere/embed-multilingual-v3.0",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "openai/text-embedding-3-large",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/text-embedding-3-small",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/text-embedding-ada-002",
              "variables": "OPENAI_API_KEY"
            }
          ]
        },
        "retriever_provider": {
          "enum": [
            "elastic-local",
            "elastic",
            "mongodb",
            "pinecone",
            "qdrant"
          ],
          "default": "elastic-local",
          "description": "The vector store provider to use for retrieval. Options are 'elastic', 'elastic-local', 'pinecone', 'mongodb', or 'qdrant'.",
          "environment": [
            {
              "value": "elastic",
              "variables": [
                "ELASTICSEARCH_URL",
                "ELASTICSEARCH_API_KEY"
              ]
            },
            {
              "value": "elastic-local",
              "variables": [
                "ELASTICSEARCH_URL",
                "ELASTICSEARCH_USER",
                "ELASTICSEARCH_PASSWORD"
              ]
            },
            {
              "value": "mongodb",
              "variables": [
                "MONGODB_URI"
              ]
            },
            {
              "value": "pinecone",
              "variables": [
                "PINECONE_API_KEY",
                "PINECONE_INDEX_NAME"
              ]
            },
            {
              "value": "qdrant",
              "variables": [
                "QDRANT_URL"
              ]
            }
          ],
          "type": "string"
        },
        "response_model": {
          "type": "string",
          "default": "anthropic/claude-opus-5-5",
          "description": "The language model used for generating responses. Should be in the form: provider/model-name.",
          "environment": [
            {
              "value": "anthropic/claude-haiku-5-5",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-opus-5-5",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-sonnet-5-5",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0125",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0301",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-1106",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0125-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-1106-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-vision-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o-mini",
              "variables": "OPENAI_API_KEY"
            }
          ]
        },
        "query_model": {
          "type": "string",
          "default": "anthropic/claude-haiku-5-5",
          "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name.",
          "environment": [
            {
              "value": "anthropic/claude-haiku-5-5",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-opus-5-5",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-sonnet-5-5",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0125",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0301",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-1106",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0125-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-1106-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-vision-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o-mini",
              "variables": "OPENAI_API_KEY"
            }
          ]
        }
      }
    }
  }
}
-->
