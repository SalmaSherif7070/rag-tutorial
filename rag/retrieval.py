"""The RETRIEVAL step: given a question, find the most relevant passages.

This is the heart of RAG. We embed the question with the SAME model used for the
passages, then ask Qdrant for the nearest vectors in a specific domain's
collection. The matching passages become the "context" we feed to the LLM.
"""

from __future__ import annotations

from typing import List, Optional

from rag.dependencies import Dependencies
from rag.domains import DOMAIN_CONFIGS, Domain
from rag.schemas import RetrievedPassage


def retrieve(
    deps: Dependencies,
    domain: Domain,
    query: str,
    top_k: Optional[int] = None,
) -> List[RetrievedPassage]:
    """Return the top_k passages from one domain that best match the query."""
    cfg = DOMAIN_CONFIGS[domain]
    top_k = top_k or deps.settings.retrieval_top_k

    query_vector = deps.embedder.embed_query(query)
    points = deps.store.search(cfg.collection, query_vector, top_k)

    return [
        RetrievedPassage(
            domain=domain.value,
            doc_id=str(point.id),
            text=(point.payload or {}).get("text", ""),
            score=float(point.score),
        )
        for point in points
    ]
