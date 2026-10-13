from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]  # repo root (configs/ is one level down)



class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=ROOT_DIR/".env",
        extra="ignore",
    )

    chroma_dir: Path = ROOT_DIR / "data" / "processed" / "chroma"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = Field(default=4, ge=1)
    min_relevance: float = Field(default=0.72, ge=0, le=1)

class GroqSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GROQ_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )

    api_key: SecretStr                       # required: app fails fast if missing
    model: str = "qwen/qwen3.8-27b"
    fallback_model: str = "openai/gpt-oss-safeguard-20b"


class AppSettings(BaseSettings):
    groq: GroqSettings = Field(default_factory=GroqSettings)
    rag: RagSettings = Field(default_factory=RagSettings)


@lru_cache
def load_settings() -> AppSettings:
    """Load settings once; later calls return the same object."""
    return AppSettings()