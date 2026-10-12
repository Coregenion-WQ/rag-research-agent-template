"""Minimal client for the RAG HTTP API, using only the standard library.

Start the API first (`docker compose up -d --build`), then:

    python examples/api_client.py index-samples
    python examples/api_client.py ask "What is a node in LangGraph?"
    python examples/api_client.py chat   # interactive, keeps the conversation

Set RAG_API_URL (default http://localhost:8000) and RAG_API_KEY if the server
requires one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "src" / "sample_docs.json"


class RagClient:
    """Thin wrapper around the API's JSON endpoints."""

    def __init__(
        self, base_url: str, api_key: str | None = None, timeout: float = 300
    ) -> None:
        """Create a client for the API at `base_url`."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"{method} {path} failed ({error.code}): {detail}")

    def health(self) -> dict[str, str]:
        """Return the server status."""
        result: dict[str, str] = self._request("GET", "/health")
        return result

    def index(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """Index documents shaped like {"page_content": ..., "metadata": {...}}."""
        result: dict[str, Any] = self._request(
            "POST", "/index", {"documents": documents}
        )
        return result

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send the conversation and return the answer, route and sources."""
        result: dict[str, Any] = self._request("POST", "/chat", {"messages": messages})
        return result

    def ask(self, question: str) -> dict[str, Any]:
        """Ask a single question without prior conversation."""
        return self.chat([{"role": "user", "content": question}])


def load_sample_docs(path: Path = SAMPLE_DOCS) -> list[dict[str, Any]]:
    """Load the LangChain/LangGraph sample documents shipped with the repo."""
    docs: list[dict[str, Any]] = json.loads(path.read_text())
    return [
        {"page_content": d["page_content"], "metadata": d.get("metadata", {})}
        for d in docs
    ]


def index_in_batches(
    client: RagClient, documents: list[dict[str, Any]], batch_size: int = 50
) -> int:
    """Index documents in batches and return how many were stored."""
    total = 0
    for start in range(0, len(documents), batch_size):
        total += client.index(documents[start : start + batch_size])["indexed"]
    return total


def format_answer(result: dict[str, Any]) -> str:
    """Render an answer and its sources for the terminal."""
    lines = [result["answer"]]
    if result["sources"]:
        lines.append("")
        lines.append(f"Sources ({len(result['sources'])}):")
        for source in result["sources"]:
            snippet = " ".join(source["page_content"].split())[:100]
            lines.append(f"  - {snippet}...")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url", default=os.environ.get("RAG_API_URL", "http://localhost:8000")
    )
    parser.add_argument("--api-key", default=os.environ.get("RAG_API_KEY"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("health", help="check that the API is up")
    commands.add_parser("index-samples", help="index the bundled sample documents")
    ask = commands.add_parser("ask", help="ask a single question")
    ask.add_argument("question")
    commands.add_parser("chat", help="interactive conversation")
    args = parser.parse_args(argv)

    client = RagClient(args.url, args.api_key)
    if args.command == "health":
        print(client.health()["status"])
    elif args.command == "index-samples":
        count = index_in_batches(client, load_sample_docs())
        print(f"Indexed {count} documents")
    elif args.command == "ask":
        print(format_answer(client.ask(args.question)))
    else:
        messages: list[dict[str, str]] = []
        while True:
            try:
                question = input("you> ").strip()
            except EOFError:
                break
            if not question:
                continue
            messages.append({"role": "user", "content": question})
            result = client.chat(messages)
            messages.append({"role": "assistant", "content": result["answer"]})
            print(format_answer(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
