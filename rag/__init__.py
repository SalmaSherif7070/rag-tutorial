"""A small, readable Retrieval-Augmented Generation (RAG) tutorial project.

Read the files in this order to understand the whole system:

    config.py        -> settings loaded from .env
    domains.py       -> the 4 knowledge domains (each is one Qdrant collection)
    schemas.py       -> the data shapes (Pydantic models + the graph State)
    embeddings.py    -> turn text into vectors with the Jina Cloud API
    vector_store.py  -> store & search those vectors in Qdrant
    ingest.py        -> load the data from Hugging Face + index it (load -> embed -> store)
    retrieval.py     -> the "search" step: embed a question -> find passages
    dependencies.py  -> builds the embedder, vector store and Gemini LLM in one place
    graph.py         -> the orchestrator: router -> workers -> synthesizer
    api.py           -> FastAPI web service (Swagger UI at /docs)
    cli.py           -> run it:  python -m rag.cli ingest / ask / delete
"""

__version__ = "0.1.0"
