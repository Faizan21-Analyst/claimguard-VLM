from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]  # repo root (configs/ is one level down)

class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )

    db_path: Path = ROOT_DIR / "data" / "processed" / "claimguard.db"
    claims_dir: Path = ROOT_DIR / "data" / "synthetic" / "claims"

    # "Try your own claim": stored apart from the demo claims. Unset = next to the audit database.
    user_db_path: Path | None = None        # default: <db_path folder>/user_claims.db
    user_claims_dir: Path | None = None     # default: <db_path folder>/user_claims/
    policies_dir: Path = ROOT_DIR / "data" / "synthetic" / "policies"
    max_photos: int = Field(default=6, ge=1, le=12)
    max_photo_mb: int = Field(default=10, ge=1, le=50)


class WeatherSettings(BaseSettings):
    """Live weather check (Open-Meteo, free, no key) used for user-submitted claims."""

    model_config = SettingsConfigDict(
        env_prefix="WEATHER_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )

    wind_gust_kmh: float = Field(default=60.0, gt=0)   # daily max gust at or above this = wind event
    hail_codes: str = "96,99"                           # WMO codes that count as hail (95 = thunderstorm only)
    timeout_s: float = Field(default=10.0, gt=0)

    def hail_code_list(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self.hail_codes.replace(" ", "").split(",") if c)

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

    # "Try your own claim": one library of policy documents searched all at once (no policy to pick).
    # The PDFs are built here on first use if the folder is empty; replace them with your own any time.
    library_dir: Path = ROOT_DIR / "data" / "policy_library"
    library_chroma_dir: Path = ROOT_DIR / "data" / "processed" / "chroma_library"
    library_top_k: int = Field(default=12, ge=1)

class VlmSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VLM_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )

    backend: str = "groq"
    vision_model: str = "qwen/qwen3.8-27b"
    max_tokens: int = Field(default=1500, ge=1)
    max_image_side: int = Field(default=1024, ge=64)

    # Local fine-tuned model (backend "local_ft") / plain base model (backend "local_base")
    local_base_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    local_adapter_dir: Path = ROOT_DIR / "models" / "qwen3vl_ft_A"
    local_max_side: int = Field(default=512, ge=64)      # must equal MAX_SIDE used in training
    local_max_new_tokens: int = Field(default=80, ge=16)
    local_dtype: str = "auto"   # auto = float16 on GPU, bfloat16 on CPU; or float16 / bfloat16 / float32

    # Remote GPU service (backend "remote_ft"): the fine-tuned model on a Hugging Face ZeroGPU Space.
    remote_space: str = ""                  # "username/space-name" or a full URL
    remote_token: SecretStr | None = None   # Hugging Face token; its owner's free GPU time is what gets used
    remote_api_name: str = "/analyze"
    remote_timeout_s: float = Field(default=90.0, gt=0)   # queue + wake-up + generation
    # What to use when the GPU service is unavailable. "none" = show an error instead.
    # A fallback is always written into the saved assessment's model_version.
    remote_fallback: str = "groq"
    remote_cooldown_s: float = Field(default=300.0, ge=0)  # after a failure, skip the GPU service for this long

    # Optional second model that only WRITES NOTES (visible evidence) for models that do not write them.
    # "none" = off (the notes column shows that the model does not produce notes). "groq" = use Groq.
    notes_backend: str = "none"
    notes_timeout_s: float = Field(default=30.0, gt=0)

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
    vlm: VlmSettings = Field(default_factory=VlmSettings)   # NEW
    api: ApiSettings = Field(default_factory=ApiSettings)
    weather: WeatherSettings = Field(default_factory=WeatherSettings)
    
@lru_cache
def load_settings() -> AppSettings:
    """Load settings once; later calls return the same object."""
    return AppSettings()