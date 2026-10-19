"""UI settings. Deliberately separate from configs/settings.py.

The UI only talks to the API over HTTP and must never hold the Groq key, so it must not
use load_settings() (AppSettings requires GROQ_API_KEY).
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]  # repo root (src/claimguard/ui/ is three levels down)


class UiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UI_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",  # other keys in .env (like the Groq key) are ignored, never exposed
    )

    api_url: str = "http://localhost:8000"
    timeout_s: float = Field(default=120.0, gt=0)  # a real analysis takes tens of seconds


@lru_cache
def load_ui_settings() -> UiSettings:
    return UiSettings()