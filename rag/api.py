"""A small REST API around the RAG app, with Swagger UI built in.

FastAPI automatically generates interactive API docs:
  - Swagger UI:  http://localhost:8000/docs
  - ReDoc:       http://localhost:8000/redoc

Run locally:   uvicorn rag.api:app --reload
Run in Docker: see docker-compose.yml (service "api").

The heavy objects (embedder, Qdrant client, Gemini, the compiled graph) are
built once, lazily, on the first request that needs them. That way the docs page
and /health load even before you've added your API keys.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from rag.data_loader import load_demo_questions
from rag.dependencies import Dependencies, build_dependencies
from rag.domains import DOMAIN_CONFIGS, Domain
from rag.graph.builder import build_rag_graph
from rag.ingest import ingest_all, ingest_domain

app = FastAPI(
    title="Beginner RAG Tutorial API",
    description=(
        "A tiny Retrieval-Augmented Generation service.\n\n"
        "**Order of use:** first `POST /ingest` to build the knowledge base, "
        "then `POST /ask` to query it. Try the endpoints right here in Swagger UI."
    ),
    version="0.1.0",
)


# --- Lazily-built singletons -------------------------------------------------

@lru_cache
def get_deps() -> Dependencies:
    return build_dependencies()


@lru_cache
def get_graph():
    return build_rag_graph(get_deps())


# --- Request / response models (these shape the Swagger UI forms) ------------

class AskRequest(BaseModel):
    question: str = Field(..., examples=["What are the symptoms of COVID-19?"])


class SourceOut(BaseModel):
    domain: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    domains: List[str]
    sources: List[SourceOut]


class IngestRequest(BaseModel):
    domain: Optional[Domain] = Field(
        default=None, description="Index only this domain. Omit to index all 4."
    )
    split: str = Field(default="test", description="Dataset split to load.")
    max_rows: Optional[int] = Field(
        default=50, description="Rows per domain. Keep small while learning; null = all."
    )


class IngestResponse(BaseModel):
    counts: Dict[str, int] = Field(..., description="Passages stored per domain.")


# --- Endpoints ---------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    """Send visitors straight to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health():
    """Liveness check (does not touch the models)."""
    return {"status": "ok"}


@app.get("/domains", tags=["meta"])
def domains():
    """List the available knowledge domains."""
    return {
        d.value: {"subset": cfg.subset, "collection": cfg.collection, "about": cfg.description}
        for d, cfg in DOMAIN_CONFIGS.items()
    }


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
def ingest(req: IngestRequest):
    """Build the knowledge base: download -> embed (Jina) -> store (Qdrant)."""
    deps = get_deps()
    try:
        if req.domain is not None:
            count = ingest_domain(deps, req.domain, split=req.split, max_rows=req.max_rows)
            return IngestResponse(counts={req.domain.value: count})
        counts = ingest_all(deps, split=req.split, max_rows=req.max_rows)
        return IngestResponse(counts=counts)
    except Exception as exc:  # surface errors as clean HTTP responses
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest):
    """Answer a question with RAG (router -> workers -> synthesizer)."""
    graph = get_graph()
    try:
        result = graph.invoke({"query": req.question})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    final = result["final_answer"]
    return AskResponse(
        question=req.question,
        answer=final.answer,
        domains=[d.value for d in result.get("domains", [])],
        sources=[
            SourceOut(domain=s.domain, score=s.score, text=s.text) for s in final.sources
        ],
    )


@app.get("/demo/{domain}", tags=["rag"])
def demo(domain: Domain, n: int = 5):
    """Return a few real {question, answer} pairs from a domain's dataset."""
    cfg = DOMAIN_CONFIGS[domain]
    return {
        "domain": domain.value,
        "subset": cfg.subset,
        "questions": load_demo_questions(cfg.subset, max_rows=n),
    }
