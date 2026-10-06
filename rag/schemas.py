"""The data shapes used across the app.

Two kinds of models live here:
  1. Plain data models (RetrievedPassage, WorkerResult, FinalAnswer) that we build
     ourselves in code.
  2. LLM-output models (Route, WorkerAnswer, Synthesis) that we hand to
     `llm.with_structured_output(...)` so Gemini returns clean, typed JSON.

`RagState` is the shared memory that flows through the LangGraph graph.
"""

from __future__ import annotations

import operator
from typing import List

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from rag.domains import Domain


# --- Plain data models -------------------------------------------------------

class RetrievedPassage(BaseModel):
    """One passage returned by Qdrant, kept so we can cite our sources."""

    domain: str
    doc_id: str
    text: str
    score: float


class WorkerResult(BaseModel):
    """What one domain worker produces: an answer plus the passages it read."""

    domain: str
    answer: str
    passages: List[RetrievedPassage]


class FinalAnswer(BaseModel):
    """The synthesizer's output: one combined answer + all sources used."""

    answer: str
    sources: List[RetrievedPassage]


# --- LLM-output models (structured output) -----------------------------------

class Route(BaseModel):
    """The router's decision: which domains are relevant to the question."""

    domains: List[Domain] = Field(
        ...,
        description="The knowledge domains relevant to the user's question. "
        "Return several if the question spans multiple domains.",
    )


class WorkerAnswer(BaseModel):
    """A single worker's answer, grounded in the context passages it was given."""

    answer: str = Field(
        ...,
        description="Answer using ONLY the provided context passages. "
        "If they do not contain the answer, say you do not have enough information.",
    )


class Synthesis(BaseModel):
    """The synthesizer merges per-domain answers into one."""

    answer: str = Field(
        ...,
        description="A single clear, non-repetitive answer combining the per-domain findings.",
    )


# --- Graph state -------------------------------------------------------------

class RagState(TypedDict, total=False):
    """Shared state passed between graph nodes.

    `worker_results` uses `operator.add` as a reducer: because several workers run
    in parallel (one per domain), each returns a 1-item list and LangGraph
    concatenates them instead of overwriting.
    """

    query: str
    domains: List[Domain]
    worker_results: Annotated[List[WorkerResult], operator.add]
    final_answer: FinalAnswer
