"""The knowledge domains.

In the original travel notebook there were 8 collections (activities, visa, ...).
Here we have 4 domains, each backed by one RAGBench subset and stored in its own
Qdrant collection. The router (see graph/nodes.py) picks which domain(s) a question
belongs to, exactly like the travel orchestrator picked travel collections.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Domain(str, Enum):
    """The 4 domains. `str` mixin so the value serializes nicely for the LLM router."""

    health = "health"
    finance = "finance"
    legal = "legal"
    tech = "tech"


@dataclass(frozen=True)
class DomainConfig:
    domain: Domain
    subset: str        # the RAGBench config name on Hugging Face
    collection: str    # the Qdrant collection name
    description: str    # shown to the router LLM so it can decide relevance


# The single source of truth. Add a domain here and it flows through the whole app.
DOMAIN_CONFIGS: dict[Domain, DomainConfig] = {
    Domain.health: DomainConfig(
        domain=Domain.health,
        subset="covidqa",
        collection="rag_health",
        description="Health and medical information, including COVID-19 and biomedical questions.",
    ),
    Domain.finance: DomainConfig(
        domain=Domain.finance,
        subset="finqa",
        collection="rag_finance",
        description="Finance, accounting, company earnings, and numerical reasoning over financial reports.",
    ),
    Domain.legal: DomainConfig(
        domain=Domain.legal,
        subset="cuad",
        collection="rag_legal",
        description="Legal contracts and clauses, obligations, and terms in commercial agreements.",
    ),
    Domain.tech: DomainConfig(
        domain=Domain.tech,
        subset="techqa",
        collection="rag_tech",
        description="Technical support, IT/software troubleshooting, and product documentation.",
    ),
}


def router_catalog() -> str:
    """A human-readable list of domains for the router's system prompt."""
    return "\n".join(
        f"- {cfg.domain.value}: {cfg.description}" for cfg in DOMAIN_CONFIGS.values()
    )
