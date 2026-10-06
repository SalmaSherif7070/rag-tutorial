"""Assemble the nodes into a runnable LangGraph graph.

The shape is identical to the travel notebook:

        START
          |
     orchestrator        (router picks domains)
          |  (fan-out: assign_workers -> Send)
        worker  worker  worker ...   (one per chosen domain, in parallel)
          |  (results merged by the operator.add reducer)
      synthesizer         (combine into one answer)
          |
         END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag.dependencies import Dependencies
from rag.graph.nodes import (
    assign_workers,
    make_orchestrator,
    make_synthesizer,
    make_worker,
)
from rag.schemas import RagState


def build_rag_graph(deps: Dependencies):
    """Build and compile the orchestrator graph."""
    builder = StateGraph(RagState)

    builder.add_node("orchestrator", make_orchestrator(deps))
    builder.add_node("worker", make_worker(deps))
    builder.add_node("synthesizer", make_synthesizer(deps))

    builder.add_edge(START, "orchestrator")
    # orchestrator -> (fan out to) worker(s)
    builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
    builder.add_edge("worker", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()
