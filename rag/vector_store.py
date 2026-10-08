"""A thin wrapper around the Qdrant client.

Qdrant is our vector database (running in Docker). It stores each passage as a
"point": an id, a vector, and a payload (the original text + which domain it came
from). We give it a query vector and it returns the most similar points.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class VectorStore:
    def __init__(self, url: str, api_key: str | None, dim: int) -> None:
        # api_key is None for a local Docker container (no auth).
        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.dim = dim

    def ensure_collection(self, name: str, recreate: bool = False) -> None:
        """Create the collection if needed. If recreate=True, wipe it first."""
        exists = self.client.collection_exists(name)
        if exists and recreate:
            self.client.delete_collection(name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=name,
                # Cosine distance pairs with the normalized Jina vectors.
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(
        self,
        name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """Insert passages (vector + payload) into a collection."""
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            for vector, payload in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=name, points=points)

    def search(self, name: str, query_vector: List[float], top_k: int):
        """Return the top_k most similar points for a query vector."""
        result = self.client.query_points(
            collection_name=name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return result.points

    def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True if it existed, False if it didn't."""
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
            return True
        return False

    def count(self, name: str) -> int:
        """How many points a collection holds (0 if it doesn't exist)."""
        if not self.client.collection_exists(name):
            return 0
        return self.client.count(collection_name=name, exact=True).count
