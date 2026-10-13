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


class VLMBackend(ABC):
    """Anything that can turn one photo into a VisualFinding."""

    name: str = "base"

    @abstractmethod
    def _raw(self, image_path: Path) -> str:
        """One model call. Return the model's raw text for the image."""

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