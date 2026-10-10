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
    confidence: float = Field(default=0.5, ge=0, le=1)