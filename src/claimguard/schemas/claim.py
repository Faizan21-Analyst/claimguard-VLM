from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from claimguard.schemas.evidence import PolicyEvidence, WeatherEvidence
from claimguard.schemas.visual import Component, DamageType, VisualFinding

Decision = Literal["supported", "needs_review", "insufficient_evidence", "possible_exclusion"]


class Claim(BaseModel):
    claim_id: str
    policy_id: str
    claimed_damage_type: DamageType
    component: Component = "unclear"
    incident_date: date | None = None
    description: str = ""


class ClaimAssessment(BaseModel):
    claim_id: str
    decision: Decision
    reasons: list[str]
    visual: list[VisualFinding] = Field(default_factory=list)
    policy_citations: list[PolicyEvidence] = Field(default_factory=list)
    weather: WeatherEvidence | None = None
    missing_evidence: list[str] = Field(default_factory=list)
    model_version: str = ""
    summary: str = ""