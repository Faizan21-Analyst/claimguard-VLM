from typing import Literal

from pydantic import BaseModel, Field

DamageType = Literal["hail", "wind", "water", "fire", "wear_and_tear", "none", "unclear"]
Severity = Literal["none", "minor", "moderate", "severe", "unclear"]
Component = Literal["roof_shingles", "gutter", "siding", "window", "ceiling", "other", "unclear"]
ImageQuality = Literal["good", "poor"]


class VisualFinding(BaseModel):
    """What the VLM may say: visible evidence only. Never coverage or payout."""

    damage_type: DamageType
    severity: Severity
    component: Component
    findings: list[str] = Field(default_factory=list, max_length=6)
    image_quality: ImageQuality = "good"
    # The fine-tuned model is trained on 4 labels only and never reports a confidence, so this default (0.50) is a
    # placeholder, not a measurement. Decision rules never read it.
    confidence: float | None = Field(default=0.50, ge=0, le=1)
    # Set only when `findings` were written by a different model than the one that chose the labels.
    notes_source: str | None = None
    # What that second model thought the damage type was; a weak hint for the reviewer, never used in decisions.
    second_opinion_damage: DamageType | None = None