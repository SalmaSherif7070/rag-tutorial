"""Command-line entry point.

    # 1. Index the knowledge base (downloads data, embeds, stores in Qdrant)
    python -m rag.cli ingest                 # all domains
    python -m rag.cli ingest --domain health # just one
    python -m rag.cli ingest --max-rows 50   # faster: fewer rows per domain

    # 2. Ask a question
    python -m rag.cli ask "What are the symptoms of COVID-19?"

    # 3. See ready-made demo questions from the dataset
    python -m rag.cli demo --domain finance
"""

from __future__ import annotations

import argparse

from rag.data_loader import load_demo_questions
from rag.dependencies import build_dependencies
from rag.domains import DOMAIN_CONFIGS, Domain
from rag.graph.builder import build_rag_graph
from rag.ingest import ingest_all, ingest_domain


def cmd_ingest(args: argparse.Namespace) -> None:
    deps = build_dependencies()
    print("Indexing knowledge base into Qdrant...\n")
    if args.domain:
        count = ingest_domain(
            deps, Domain(args.domain), split=args.split, max_rows=args.max_rows
        )
        print(f"  {args.domain}: {count} passages")
    else:
        counts = ingest_all(deps, split=args.split, max_rows=args.max_rows)
        for domain, count in counts.items():
            print(f"  {domain}: {count} passages")
    print("\nDone. Now ask a question:  python -m rag.cli ask \"...\"")


def cmd_ask(args: argparse.Namespace) -> None:
    deps = build_dependencies()
    graph = build_rag_graph(deps)

    result = graph.invoke({"query": args.question})
    final = result["final_answer"]

    print("\n=== ANSWER ===\n")
    print(final.answer)

    print("\n=== SOURCES ===")
    for src in final.sources[:10]:
        snippet = src.text[:160].replace("\n", " ")
        print(f"- [{src.domain}] (score {src.score:.3f}) {snippet}...")


def cmd_demo(args: argparse.Namespace) -> None:
    cfg = DOMAIN_CONFIGS[Domain(args.domain)]
    print(f"Demo questions from '{args.domain}' ({cfg.subset}):\n")
    for item in load_demo_questions(cfg.subset, max_rows=args.n):
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="A tiny RAG tutorial app.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    domain_choices = [d.value for d in Domain]

    p_ingest = subparsers.add_parser("ingest", help="Download, embed, and store the knowledge base.")
    p_ingest.add_argument("--domain", choices=domain_choices, help="Only index this domain.")
    p_ingest.add_argument("--split", default="test", help="Dataset split (default: test).")
    p_ingest.add_argument("--max-rows", type=int, default=None, help="Limit rows per domain (faster).")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = subparsers.add_parser("ask", help="Ask a question against the knowledge base.")
    p_ask.add_argument("question", help="Your question, in quotes.")
    p_ask.set_defaults(func=cmd_ask)

    p_demo = subparsers.add_parser("demo", help="Print sample questions from a domain.")
    p_demo.add_argument("--domain", choices=domain_choices, default="health")
    p_demo.add_argument("-n", type=int, default=5, help="How many questions to show.")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
