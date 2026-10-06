"""Wire the building blocks together.

Rather than creating the embedder, vector store and LLM in many places (and
passing API keys around), we build them once here and carry them in a single
`Dependencies` object. Every part of the app receives this object, which keeps
the code easy to test and reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from rag.config import Settings, get_settings
from rag.embeddings import JinaEmbedder
from rag.llm import build_llm
from rag.vector_store import VectorStore


@dataclass
class Dependencies:
    settings: Settings
    embedder: JinaEmbedder
    store: VectorStore
    llm: ChatGoogleGenerativeAI


def build_dependencies(settings: Optional[Settings] = None) -> Dependencies:
    settings = settings or get_settings()

    embedder = JinaEmbedder(
        api_key=settings.jina_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )
    store = VectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        dim=settings.embedding_dim,
    )
    llm = build_llm(settings)

    return Dependencies(settings=settings, embedder=embedder, store=store, llm=llm)
