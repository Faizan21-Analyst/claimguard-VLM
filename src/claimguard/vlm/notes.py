"""An optional second model that only writes NOTES for a photo (and one damage type as a weak second opinion).

Why it exists: the fine-tuned model is trained on four labels only, so it cannot say what it sees in words and has no
real confidence. This class asks a general model (Groq) to describe the visible evidence. Rules:

- It never raises. Any failure (quota, timeout, bad JSON) returns "no notes", because notes are cosmetic and a
  missing note must never block or change a claim decision.
- Its output is shown as coming from this model (see VisualFinding.notes_source), never as the labelling model's.
- Nothing here reaches the decision rules in agent/fusion.py.
"""
import logging
from pathlib import Path
from typing import get_args

from groq import Groq

from claimguard.schemas.visual import DamageType
from claimguard.vlm.images import image_to_data_url
from claimguard.vlm.parsing import extract_json
from claimguard.vlm.prompt import NOTES_PROMPT
from configs.settings import load_settings

logger = logging.getLogger(__name__)

MAX_NOTES = 6
MAX_NOTE_CHARS = 120

# Groq's free tier allows about 1000 output tokens per minute for this model, and a request that asks for more is
# refused outright. Six short phrases need ~150 tokens, so keep the cap small and switch thinking off (below).
NOTES_MAX_TOKENS = 300
_DAMAGE_TYPES = set(get_args(DamageType))


def clean_notes(data: dict) -> tuple[list[str], str | None]:
    """(notes, damage_type) from the model's JSON. Drops anything that is not a short plain string."""
    raw = data.get("findings")
    notes = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                notes.append(" ".join(item.split())[:MAX_NOTE_CHARS])
    damage = data.get("damage_type")
    return notes[:MAX_NOTES], damage if damage in _DAMAGE_TYPES else None


class GroqNotes:
    """Pass `client` to inject a fake in tests."""

    def __init__(self, client: Groq | None = None):
        settings = load_settings()
        self.model = settings.vlm.vision_model
        self.max_tokens = NOTES_MAX_TOKENS
        self.max_image_side = settings.vlm.max_image_side
        # Qwen3.8 thinks before it answers unless told not to; thinking would eat the whole token budget.
        # Only sent to Qwen3.8: other models reject or ignore this field.
        self.extra_body = {"reasoning_effort": "none"} if "qwen3.8" in self.model else None
        self.client = client or Groq(
            api_key=settings.groq.api_key.get_secret_value(),
            timeout=settings.vlm.notes_timeout_s,
            max_retries=0,   # a rate limit should fail fast here, not delay the claim
        )
        self.name = f"groq:{self.model}"

    def describe(self, image_path: Path | str) -> tuple[list[str], str | None]:
        """(notes, second-opinion damage type). ([], None) on any problem."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=self.max_tokens,
                extra_body=self.extra_body,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": NOTES_PROMPT},
                            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path, self.max_image_side)}},
                        ],
                    }
                ],
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("notes output was cut off")
            return clean_notes(extract_json(choice.message.content or ""))
        except Exception as exc:  # noqa: BLE001 - notes are optional; nothing may escape from here
            logger.warning("%s: notes unavailable: %s: %s", self.name, type(exc).__name__, exc)
            return [], None


def get_notes_writer():
    """The configured notes writer, or None when VLM_NOTES_BACKEND is "none"."""
    backend = load_settings().vlm.notes_backend.strip().lower()
    if backend in ("", "none"):
        return None
    if backend == "groq":
        return GroqNotes()
    raise ValueError(f"Unknown VLM_NOTES_BACKEND {backend!r}. Available: none, groq")
