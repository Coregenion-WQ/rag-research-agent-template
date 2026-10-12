"""Run the examples against a real HTTP server backed by the fake model."""

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator

import pytest
import uvicorn

from rag_api import create_app
from rag_api.settings import Settings
from shared import retrieval
from shared.retrieval import get_qdrant_client
from tests.fakes import FakeLLMFactory, InMemoryRetrieverFactory, fake_embeddings

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
API_KEY = "example-key"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api_client = _load("api_client")
run_graphs_locally = _load("run_graphs_locally")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _serve(api_key: str | None) -> Iterator[str]:
    port = _free_port()
    config = uvicorn.Config(
        create_app(Settings(api_key=api_key)),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def llm(make_fake_llm: Callable[..., FakeLLMFactory]) -> FakeLLMFactory:
    return make_fake_llm(text="Nodes are functions.")


@pytest.fixture
def server(
    llm: FakeLLMFactory, fake_retriever: InMemoryRetrieverFactory
) -> Iterator[str]:
    yield from _serve(API_KEY)


@pytest.fixture
def open_server(
    llm: FakeLLMFactory, fake_retriever: InMemoryRetrieverFactory
) -> Iterator[str]:
    yield from _serve(None)


# --- api_client.py ----------------------------------------------------------


def test_client_indexes_samples_and_answers(
    server: str, fake_retriever: InMemoryRetrieverFactory
) -> None:
    client = api_client.RagClient(server, API_KEY)
    assert client.health() == {"status": "ok"}

    docs = api_client.load_sample_docs()
    unique = {d["page_content"] for d in docs}
    assert api_client.index_in_batches(client, docs, batch_size=40) == len(unique)
    assert len(fake_retriever.store.store) == len(unique)

    result = client.ask("What is a node?")
    assert result["answer"] == "Nodes are functions."
    assert result["route"] == "langchain"
    assert result["sources"]

    rendered = api_client.format_answer(result)
    assert rendered.startswith("Nodes are functions.")
    assert f"Sources ({len(result['sources'])}):" in rendered


def test_client_reports_http_errors(server: str) -> None:
    client = api_client.RagClient(server, "wrong-key")
    with pytest.raises(RuntimeError, match=r"POST /chat failed \(401\)"):
        client.ask("hi")


def test_cli_commands(server: str, capsys: pytest.CaptureFixture[str]) -> None:
    common = ["--url", server, "--api-key", API_KEY]
    assert api_client.main([*common, "health"]) == 0
    assert api_client.main([*common, "index-samples"]) == 0
    assert api_client.main([*common, "ask", "What is a node?"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ok\nIndexed ")
    assert "Nodes are functions." in out


def test_interactive_chat_keeps_history(
    server: str,
    llm: FakeLLMFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["What is LangGraph?", "", "And a node?"])

    def fake_input(prompt: str) -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    assert api_client.main(["--url", server, "--api-key", API_KEY, "chat"]) == 0
    router_calls = [c for c in llm.calls if c.get("schema") == "Router"]
    second_turn = router_calls[-1]["messages"]
    assert [m.content for m in second_turn[1:]] == [
        "What is LangGraph?",
        "Nodes are functions.",
        "And a node?",
    ]
    assert capsys.readouterr().out.count("Nodes are functions.") == 2


# --- curl.sh ----------------------------------------------------------------


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl not installed")
@pytest.mark.parametrize("with_key", [True, False])
def test_curl_script(
    with_key: bool,
    llm: FakeLLMFactory,
    fake_retriever: InMemoryRetrieverFactory,
) -> None:
    for url in _serve(API_KEY if with_key else None):
        env = {**os.environ, "RAG_API_URL": url}
        env.pop("RAG_API_KEY", None)
        if with_key:
            env["RAG_API_KEY"] = API_KEY
        result = subprocess.run(
            ["bash", str(EXAMPLES / "curl.sh")],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        responses = [
            json.loads(line) for line in result.stdout.splitlines() if line[:1] == "{"
        ]
        assert responses[0] == {"status": "ok"}
        assert responses[1]["indexed"] == 2
        assert "edges-overview" in responses[1]["ids"]
        assert [r["answer"] for r in responses[2:]] == ["Nodes are functions."] * 2


# --- run_graphs_locally.py --------------------------------------------------


async def test_run_graphs_locally(
    llm: FakeLLMFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "examples_test")
    monkeypatch.setattr(retrieval, "make_text_encoder", lambda _: fake_embeddings())
    get_qdrant_client.cache_clear()
    try:
        config = run_graphs_locally.make_config()
        result = await run_graphs_locally.run("What is a node?", config)
    finally:
        get_qdrant_client.cache_clear()
    assert result["messages"][-1].content == "Nodes are functions."
    assert result["documents"]
    assert set(llm.requested) == {
        "anthropic/claude-haiku-5-5",
        "anthropic/claude-opus-5-5",
        "openai/gpt-5",
    }


def test_examples_have_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(EXAMPLES / "curl.sh")], check=True)
    assert sys.executable  # examples run with the project interpreter
