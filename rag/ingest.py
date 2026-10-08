"""Load the data from Hugging Face and build the knowledge base (the "index").

For each domain we:
  1. download its passages from the RAGBench dataset,
  2. turn them into vectors with Jina,
  3. store them in that domain's Qdrant collection.

Run it:  python -m rag.cli ingest
"""

from __future__ import annotations

from typing import Dict, List, Optional

from datasets import load_dataset

from rag.dependencies import Dependencies
from rag.domains import DOMAIN_CONFIGS, Domain

RAGBENCH = "galileo-ai/ragbench"


# --- Loading data from Hugging Face ------------------------------------------

def load_domain_passages(subset: str, split: str = "test", max_rows: Optional[int] = None) -> List[str]:
    """Download and de-duplicate the passages for one domain."""
    dataset = load_dataset(RAGBENCH, subset, split=split)
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    passages: List[str] = []
    seen: set[str] = set()
    for row in dataset:
        for doc in row.get("documents") or []:
            text = (doc or "").strip()
            if text and text not in seen:
                seen.add(text)
                passages.append(text)
    return passages


def load_demo_questions(subset: str, split: str = "test", max_rows: int = 5) -> List[Dict[str, str]]:
    """A few ready-made {question, answer} pairs you can use to test the system."""
    dataset = load_dataset(RAGBENCH, subset, split=split)
    dataset = dataset.select(range(min(max_rows, len(dataset))))
    return [{"question": row["question"], "answer": row["response"]} for row in dataset]


# --- Indexing into Qdrant ----------------------------------------------------

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
    """Index all domains. Returns {domain: passage_count}."""
    return {
        domain.value: ingest_domain(deps, domain, split=split, max_rows=max_rows, recreate=recreate)
        for domain in DOMAIN_CONFIGS
    }
