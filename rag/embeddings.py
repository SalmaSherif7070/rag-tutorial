"""Turn text into vectors using the Jina Cloud embeddings API.

This replaces the notebook's local `SentenceTransformer("LaBSE")`. Instead of
running a model on your machine, we send the text to Jina over HTTPS and get
vectors back. No GPU, no model download.

Key idea for RAG: we embed *passages* and *queries* with DIFFERENT tasks.
Jina v3 produces better-matching vectors when it knows whether a piece of text
is a document to be searched ("retrieval.passage") or a search query
("retrieval.query").
"""

from __future__ import annotations

from typing import List

import requests

JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"


class JinaEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "jina-embeddings-v3",
        dimensions: int = 1024,
        batch_size: int = 64,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout

    def _embed(self, texts: List[str], task: str) -> List[List[float]]:
        """Call the Jina API in batches and return one vector per input text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            body = {
                "model": self.model,
                "task": task,
                "dimensions": self.dimensions,
                "normalized": True,          # unit vectors -> cosine similarity is clean
                "input": batch,
            }
            resp = requests.post(JINA_ENDPOINT, headers=headers, json=body, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()["data"]
            # The API may return items out of order; sort by index to be safe.
            data.sort(key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in data)
        return vectors

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embed documents we want to store and search over."""
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user question."""
        return self._embed([text], task="retrieval.query")[0]
