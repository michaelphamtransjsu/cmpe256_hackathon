"""Process configuration, with no credentials written to disk."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AgentName = Literal["clinical", "corporate_responsibility", "technology", "sec"]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    embeddings_provider: Literal["hash", "openai"] = "hash"
    llm_provider: Literal["extractive", "openai"] = "extractive"
    chat_model: str = Field(default="gpt-4.1-mini", min_length=1)
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.08, ge=0, le=1)

    @classmethod
    def from_env(cls) -> "Settings":
        names = {
            "data_dir": "SPECIALIST_DATA_DIR",
            "embeddings_provider": "SPECIALIST_EMBEDDINGS_PROVIDER",
            "llm_provider": "SPECIALIST_LLM_PROVIDER",
            "chat_model": "SPECIALIST_CHAT_MODEL",
            "embedding_model": "SPECIALIST_EMBEDDING_MODEL",
            "top_k": "SPECIALIST_TOP_K",
            "min_score": "SPECIALIST_MIN_SCORE",
        }
        return cls(**{key: os.environ[name] for key, name in names.items() if name in os.environ})



def make_chat_model(settings: Settings):
    """Construct the optional hosted model without changing process configuration."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.chat_model, temperature=0, timeout=60, max_retries=2)
