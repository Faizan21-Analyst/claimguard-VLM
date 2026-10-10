from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]  # repo root (configs/ is one level down)


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


@lru_cache
def load_settings() -> AppSettings:
    """Load settings once; later calls return the same object."""
    return AppSettings()