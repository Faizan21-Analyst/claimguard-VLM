"""Use the primary model; if it is unavailable, use a secondary one and SAY SO.

The fallback is never silent: the backend name that comes back from analyze_traced() names the model that really
answered and the one that was unavailable, and the graph writes that into the assessment's model_version. A reviewer
therefore always sees which model produced a result, and the fine-tuned model is never credited for Groq's answer.

Only ModelUnavailable triggers the switch. Bad model output (ValueError) is still handled inside each backend, and
real bugs still raise.
"""
import logging
import time
from pathlib import Path

from claimguard.schemas.visual import VisualFinding
from claimguard.vlm.base import ModelUnavailable, VLMBackend

logger = logging.getLogger(__name__)


class FallbackVLM(VLMBackend):
    def __init__(
        self,
        primary: VLMBackend,
        secondary: VLMBackend,
        cooldown_s: float = 300.0,
        clock=time.monotonic,
    ):
        self.primary = primary
        self.secondary = secondary
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._skip_primary_until = 0.0
        self.name = primary.name   # what is shown before any photo has been analyzed

    def _raw(self, image_path: Path) -> str:
        return self.primary._raw(image_path)

    def analyze_traced(self, image_path: Path | str, retries: int = 1) -> tuple[VisualFinding, bool, str]:
        # After a failure, skip the primary for a while: otherwise every photo of a claim would wait for its
        # own timeout while the service is down or out of quota.
        if self._clock() >= self._skip_primary_until:
            try:
                return self.primary.analyze_traced(image_path, retries)
            except ModelUnavailable as exc:
                logger.warning("%s unavailable (%s); using %s", self.primary.name, exc, self.secondary.name)
                self._skip_primary_until = self._clock() + self.cooldown_s

        finding, ok, name = self.secondary.analyze_traced(image_path, retries)
        return finding, ok, f"{name} (fallback: {self.primary.name} unavailable)"

    def analyze(self, image_path: Path | str, retries: int = 1) -> tuple[VisualFinding, bool]:
        finding, ok, _ = self.analyze_traced(image_path, retries)
        return finding, ok
