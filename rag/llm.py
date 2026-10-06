"""The LLM (the "G" in RAG: Generation).

We keep Gemini, same as the notebook, but read the key and model name from
settings instead of hard-coding them.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from rag.config import Settings


def build_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.1,  # low = focused, factual answers
        google_api_key=settings.google_api_key,
    )
