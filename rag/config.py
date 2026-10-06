"""All configuration lives here, loaded from the .env file.

We never hard-code API keys in the source (unlike the original notebook!).
Instead we read them from environment variables / the .env file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings. Field names map to .env keys (case-insensitive)."""

    # --- Secrets (no defaults: the app refuses to start without them) ---
    jina_api_key: str
    google_api_key: str

    # --- Qdrant (the Docker container) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # --- Models ---
    embedding_model: str = "jina-embeddings-v3"
    embedding_dim: int = 1024          # Jina v3 can shrink to this via Matryoshka
    gemini_model: str = "gemini-2.5-flash"

    # --- Retrieval ---
    retrieval_top_k: int = 5           # how many passages each worker pulls

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them (cached)."""
    return Settings()
