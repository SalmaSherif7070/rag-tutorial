"""The orchestrator: router -> workers -> synthesizer.

A question flows through three steps:
  1. orchestrator (router) : which domain(s) can answer this?
  2. worker (one per domain): find passages + draft an answer from them.
  3. synthesizer           : merge the workers' answers into one, with sources.

Each node is a plain function that also needs `deps` (our embedder/store/llm).
When we build the graph we bind `deps` with functools.partial, so LangGraph can
call each node with just the state.
"""

from __future__ import annotations

from functools import partial

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from rag.dependencies import Dependencies
from rag.domains import Domain, router_catalog
from rag.retrieval import retrieve
from rag.schemas import FinalAnswer, RagState, Route, Synthesis, WorkerAnswer, WorkerResult


# 1. ROUTER -------------------------------------------------------------------

def orchestrator(state: RagState, deps: Dependencies) -> dict:
    """Ask the LLM which domains are relevant to the question."""
    router = deps.llm.with_structured_output(Route)
    decision = router.invoke(
        [
            SystemMessage(
                content=(
                    "You are a router. Pick ONLY the domains that can answer the question.\n"
                    f"Domains:\n{router_catalog()}\n\n"
                    "Return several if it spans more than one; if unsure, pick the closest."
                )
            ),
            HumanMessage(content=f"Question: {state['query']}"),
        ]
    )
    # Fall back to all domains if the router picked none.
    return {"domains": decision.domains or list(Domain)}


def assign_workers(state: RagState):
    """Fan out: start one worker per chosen domain (they run in parallel)."""
    return [
        Send("worker", {"domain": d.value, "query": state["query"]})
        for d in state["domains"]
    ]


# 2. WORKER -------------------------------------------------------------------

def worker(state: dict, deps: Dependencies) -> dict:
    """Retrieve passages for one domain, then answer using only those passages."""
    domain = Domain(state["domain"])
    query = state["query"]

    passages = retrieve(deps, domain, query)
    context = "\n\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages)) or "(nothing found)"

    result = deps.llm.with_structured_output(WorkerAnswer).invoke(
        [
            SystemMessage(
                content=(
                    f"You are a {domain.value} expert. Answer using ONLY the context below. "
                    "If it lacks the answer, say you don't have enough information.\n\n"
                    f"Context:\n{context}"
                )
            ),
            HumanMessage(content=f"Question: {query}"),
        ]
    )

    # Return a 1-item list; the reducer in RagState concatenates all workers.
    return {"worker_results": [WorkerResult(domain=domain.value, answer=result.answer, passages=passages)]}


# 3. SYNTHESIZER --------------------------------------------------------------

def synthesizer(state: RagState, deps: Dependencies) -> dict:
    """Merge the workers' answers into one, and gather all the sources."""
    results = state.get("worker_results", [])
    if not results:
        return {"final_answer": FinalAnswer(answer="I couldn't find relevant information.", sources=[])}

    per_domain = "\n\n".join(f"## {r.domain}\n{r.answer}" for r in results)
    merged = deps.llm.with_structured_output(Synthesis).invoke(
        [
            SystemMessage(
                content="Combine the per-domain answers into one clear, non-repetitive answer. Do not invent facts."
            ),
            HumanMessage(content=f"Question: {state['query']}\n\nFindings:\n{per_domain}"),
        ]
    )

    sources = [passage for r in results for passage in r.passages]
    return {"final_answer": FinalAnswer(answer=merged.answer, sources=sources)}


# BUILD -----------------------------------------------------------------------

def build_rag_graph(deps: Dependencies):
    """Wire the three nodes into a runnable graph."""
    builder = StateGraph(RagState)

    builder.add_node("orchestrator", partial(orchestrator, deps=deps))
    builder.add_node("worker", partial(worker, deps=deps))
    builder.add_node("synthesizer", partial(synthesizer, deps=deps))

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
    builder.add_edge("worker", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()
