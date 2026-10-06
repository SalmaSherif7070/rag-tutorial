"""Download the knowledge base from Hugging Face.

We use the RAGBench dataset (galileo-ai/ragbench). Each subset (covidqa, finqa,
cuad, techqa) is a list of examples shaped like:

    {
      "question":  "What is ...?",
      "documents": ["passage 1 ...", "passage 2 ...", ...],  # <- what we index
      "response":  "the ground-truth answer",
      ...
    }

For RAG we only need the `documents` passages to build our searchable knowledge
base. The `question` / `response` pairs are handy as ready-made demo queries.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from datasets import load_dataset

RAGBENCH = "galileo-ai/ragbench"


def load_domain_passages(
    subset: str,
    split: str = "test",
    max_rows: Optional[int] = None,
) -> List[str]:
    """Return a de-duplicated list of passage strings for one domain."""
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


def load_demo_questions(
    subset: str,
    split: str = "test",
    max_rows: int = 5,
) -> List[Dict[str, str]]:
    """Return a few {question, answer} pairs you can use to test the system."""
    dataset = load_dataset(RAGBENCH, subset, split=split)
    dataset = dataset.select(range(min(max_rows, len(dataset))))
    return [{"question": row["question"], "answer": row["response"]} for row in dataset]
