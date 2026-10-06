"""The three node types of the orchestrator, plus the fan-out helper.

This mirrors the travel notebook's design:

    orchestrator  (router)   -> decides which domains are relevant
    worker        (per domain) -> retrieves passages + drafts an answer
    synthesizer              -> merges the workers' answers into one

Each `make_*` function takes the shared `Dependencies` and returns the actual
node function (a closure). This way nodes can use the embedder / store / llm
without relying on global variables.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from rag.dependencies import Dependencies
from rag.domains import Domain, router_catalog
from rag.retrieval import retrieve
from rag.schemas import (
    FinalAnswer,
    RagState,
    Route,
    Synthesis,
    WorkerAnswer,
    WorkerResult,
)


def make_orchestrator(deps: Dependencies):
    """Node 1: the router. Chooses which domains can answer the question."""
    router = deps.llm.with_structured_output(Route)

    def orchestrator(state: RagState) -> dict:
        decision = router.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a router for a retrieval-augmented question answering system.\n"
                        "Choose ONLY the knowledge domains that can help answer the question.\n\n"
                        "Available domains:\n"
                        f"{router_catalog()}\n\n"
                        "Return several domains if the question clearly spans more than one. "
                        "If none fit well, return the single closest domain."
                    )
                ),
                HumanMessage(content=f"Question: {state['query']}"),
            ]
        )
        # Fall back to all domains if the router returns nothing.
        domains = decision.domains or list(Domain)
        return {"domains": domains}

    return orchestrator


def assign_workers(state: RagState):
    """Conditional edge: fan out to one `worker` per chosen domain, in parallel.

    `Send` lets us launch the same node many times with different inputs. Each
    worker receives just the domain it owns plus the shared query.
    """
    return [
        Send("worker", {"domain": domain.value, "query": state["query"]})
        for domain in state["domains"]
    ]


def make_worker(deps: Dependencies):
    """Node 2: one worker per domain. Retrieves passages, then answers from them."""
    answerer = deps.llm.with_structured_output(WorkerAnswer)

    def worker(state: dict) -> dict:
        domain = Domain(state["domain"])
        query = state["query"]

        passages = retrieve(deps, domain, query)
        context = "\n\n".join(
            f"[{i}] {p.text}" for i, p in enumerate(passages)
        ) or "(no passages found)"

        result = answerer.invoke(
            [
                SystemMessage(
                    content=(
                        f"You are a {domain.value} expert. Answer the question using ONLY the "
                        "context passages below. If they do not contain the answer, say you do "
                        "not have enough information.\n\n"
                        f"Context:\n{context}"
                    )
                ),
                HumanMessage(content=f"Question: {query}"),
            ]
        )

        # Return a 1-item list; the `operator.add` reducer merges all workers.
        return {
            "worker_results": [
                WorkerResult(domain=domain.value, answer=result.answer, passages=passages)
            ]
        }

    return worker


def make_synthesizer(deps: Dependencies):
    """Node 3: merge the workers' answers into one final, sourced answer."""
    synthesizer_llm = deps.llm.with_structured_output(Synthesis)

    def synthesizer(state: RagState) -> dict:
        results = state.get("worker_results", [])
        if not results:
            return {"final_answer": FinalAnswer(answer="I couldn't find relevant information.", sources=[])}

        per_domain = "\n\n".join(f"## {r.domain}\n{r.answer}" for r in results)
        merged = synthesizer_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Combine the per-domain answers below into ONE clear, non-repetitive "
                        "answer for the user. Stay faithful to the findings; do not invent facts."
                    )
                ),
                HumanMessage(
                    content=f"Question: {state['query']}\n\nPer-domain findings:\n{per_domain}"
                ),
            ]
        )

        # Attach every passage the workers used so the user can verify the answer.
        sources = [passage for r in results for passage in r.passages]
        return {"final_answer": FinalAnswer(answer=merged.answer, sources=sources)}

    return synthesizer
