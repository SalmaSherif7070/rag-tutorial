"""A small, readable Retrieval-Augmented Generation (RAG) tutorial project.

Read the files in this order to understand the whole system:

    config.py        -> settings loaded from .env
    domains.py       -> the 4 knowledge domains (each is one Qdrant collection)
    schemas.py       -> the data shapes (Pydantic models + the graph State)
    embeddings.py    -> turn text into vectors with the Jina Cloud API
    vector_store.py  -> store & search those vectors in Qdrant
    data_loader.py   -> download the RAGBench dataset from Hugging Face
    ingest.py        -> the "index" pipeline: load -> embed -> store
    retrieval.py     -> the "search" step: embed a question -> find passages
    llm.py           -> the Gemini model that writes the answers
    dependencies.py  -> wires the pieces above into one object
    graph/           -> the orchestrator: router -> workers -> synthesizer
    cli.py           -> run it:  python -m rag.cli ingest / ask
"""

__version__ = "0.1.0"
