from pathlib import Path

from groq import Groq

from configs.settings import load_settings
from claimguard.vlm.base import VLMBackend
from claimguard.vlm.images import image_to_data_url
from claimguard.vlm.prompt import VLM_PROMPT


class GroqVLM(VLMBackend):
    """Zero-shot VLM through Groq. Pass `client` to inject a fake in tests."""

    def __init__(self, client: Groq | None = None):
        settings = load_settings()
        self.model = settings.vlm.vision_model
        self.max_tokens = settings.vlm.max_tokens
        self.max_image_side = settings.vlm.max_image_side
        self.client = client or Groq(
            api_key=settings.groq.api_key.get_secret_value(),
            timeout=60.0,
        )
        self.name = f"groq:{self.model}"

    def _raw(self, image_path: Path) -> str:
        data_url = image_to_data_url(image_path, self.max_image_side)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        # Reasoning models can spend the whole budget thinking and return nothing useful.
        if choice.finish_reason == "length":
            raise ValueError("model output was cut off; raise VLM_MAX_TOKENS")
        if not text.strip():
            raise ValueError("model returned an empty response")
        return text