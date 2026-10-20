import logging
from abc import ABC, abstractmethod
from pathlib import Path

from claimguard.schemas.visual import VisualFinding
from claimguard.vlm.parsing import extract_json

logger = logging.getLogger(__name__)

SAFE_FALLBACK = VisualFinding(
    damage_type="unclear",
    severity="unclear",
    component="unclear",
    findings=[],
    image_quality="poor",
    confidence=0.0,
)


class ModelUnavailable(RuntimeError):
    """The model could not be reached, or has no capacity right now (daily GPU quota, queue, timeout, service down).

    Deliberately NOT a ValueError: analyze() turns a ValueError (bad model output) into an "unclear" photo, and a
    photo must never be called unclear just because a server was down. This error travels up unchanged, so the
    caller can show an honest message or switch to a fallback that is recorded as such.
    """


class VLMBackend(ABC):
    """Anything that can turn one photo into a VisualFinding."""

    name: str = "base"

    @abstractmethod
    def _raw(self, image_path: Path) -> str:
        """One model call. Return the model's raw text for the image."""

    def analyze_traced(self, image_path: Path | str, retries: int = 1) -> tuple[VisualFinding, bool, str]:
        """Like analyze(), plus the name of the backend that really produced the answer.

        A plain backend always answers as itself. FallbackVLM overrides this to say when a fallback answered,
        so the saved assessment never claims a model that did not produce it.
        """
        finding, ok = self.analyze(image_path, retries)
        return finding, ok, self.name

    def analyze(self, image_path: Path | str, retries: int = 1) -> tuple[VisualFinding, bool]:
        """Return (finding, json_valid).

        Never guesses: if every attempt fails, returns an all-"unclear" finding and False.
        Only ValueError is caught. That covers bad JSON, pydantic ValidationError (a ValueError
        subclass) and our own truncation errors. Real bugs and network errors still raise.
        """
        if retries < 0:
            raise ValueError("retries must be >= 0")

        path = Path(image_path)
        for attempt in range(retries + 1):
            try:
                finding = VisualFinding.model_validate(extract_json(self._raw(path)))
                return finding, True
            except ValueError as exc:
                logger.warning("%s: attempt %d/%d failed: %s", self.name, attempt + 1, retries + 1, exc)

        return SAFE_FALLBACK.model_copy(deep=True), False