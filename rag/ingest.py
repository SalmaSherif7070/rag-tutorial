"""The INDEXING pipeline (the "R" in RAG is only possible after this runs).

For each domain we:
  1. download its passages from Hugging Face,
  2. turn them into vectors with Jina,
  3. store the vectors + original text in that domain's Qdrant collection.

Run it from the command line:  python -m rag.cli ingest
"""

from __future__ import annotations

from typing import Dict, Optional

from rag.data_loader import load_domain_passages
from rag.dependencies import Dependencies
from rag.domains import DOMAIN_CONFIGS, Domain


def ingest_domain(
    deps: Dependencies,
    domain: Domain,
    split: str = "test",
    max_rows: Optional[int] = None,
    recreate: bool = True,
) -> int:
    """Index one domain. Returns how many passages were stored."""
    cfg = DOMAIN_CONFIGS[domain]

    passages = load_domain_passages(cfg.subset, split=split, max_rows=max_rows)
    if not passages:
        return 0

    # Fresh collection sized to our embedding dimension.
    deps.store.ensure_collection(cfg.collection, recreate=recreate)

    vectors = deps.embedder.embed_passages(passages)
    payloads = [{"text": text, "domain": domain.value} for text in passages]
    deps.store.upsert(cfg.collection, vectors, payloads)

    return len(passages)


def ingest_all(
    deps: Dependencies,
    split: str = "test",
    max_rows: Optional[int] = None,
    recreate: bool = True,
) -> Dict[str, int]:
    """Index every domain. Returns {domain: passage_count}."""
    counts: Dict[str, int] = {}
    for domain in DOMAIN_CONFIGS:
        counts[domain.value] = ingest_domain(
            deps, domain, split=split, max_rows=max_rows, recreate=recreate
        )
    return counts
