"""Decision rules. Plain code, no AI, so every decision can be explained and tested.

The system never denies a claim. It only tells a human reviewer what the evidence supports.
"""
from pydantic import BaseModel, Field

from claimguard.agent.perils import (
    WEATHER_PERILS,
    clause_mentions_peril,
    says_covered,
    says_excluded,
)
from claimguard.schemas.claim import Claim, Decision
from claimguard.schemas.evidence import PolicyEvidence, WeatherEvidence
from claimguard.schemas.visual import VisualFinding


class FusionResult(BaseModel):
    decision: Decision
    reasons: list[str] = Field(min_length=1)  # every decision must explain itself
    missing_evidence: list[str] = Field(default_factory=list)


def _label(damage_type: str) -> str:
    return "no visible damage" if damage_type == "none" else damage_type


def _ids(clauses: list[PolicyEvidence]) -> str:
    return ", ".join(c.clause_id for c in clauses)


def _where(clause: PolicyEvidence) -> str:
    """Where a clause is. Demo clauses only have a number; library clauses also know their document."""
    if not clause.policy_name:
        return f"Clause {clause.clause_id}"
    heading = f" ({clause.section_title})" if clause.section_title else ""
    return f"{clause.policy_name}, Section {clause.clause_id}{heading}"


def _refs(clauses: list[PolicyEvidence]) -> str:
    if not any(c.policy_name for c in clauses):
        return f"Clause {_ids(clauses)}"  # the original wording, unchanged
    return "; ".join(_where(c) for c in clauses)


def _quote_line(clause: PolicyEvidence) -> list[str]:
    """The exact words of the policy, when we have them, so the record can be checked against the document."""
    return [f'Policy text: "{clause.quote}"'] if clause.quote else []


def _weather_note(weather: WeatherEvidence | None) -> list[str]:
    """The numbers behind a live weather answer, for the official record. Demo (mock) weather has none."""
    return [f"Weather data: {weather.details}"] if weather is not None and weather.details else []


def fuse(
    claim: Claim,
    findings: list[VisualFinding],
    policy: list[PolicyEvidence],
    weather: WeatherEvidence | None,
) -> FusionResult:
    """Combine photos, policy clauses and weather. The first rule that fires decides.

    Note: the VLM's `confidence` is deliberately ignored. A model can be confidently wrong.
    """
    peril = claim.claimed_damage_type

    # Rule 1: at least one usable photo.
    usable = [f for f in findings if f.image_quality == "good" and f.damage_type != "unclear"]
    if not usable:
        why = (
            "No photos were provided."
            if not findings
            else "No photo is usable: every photo is poor quality or unclear."
        )
        return FusionResult(
            decision="insufficient_evidence",
            reasons=[why],
            missing_evidence=["clear damage photos"],
        )

    # Rule 2: at least one policy clause that actually talks about this peril.
    relevant = [c for c in policy if clause_mentions_peril(c.text, peril)]
    if not relevant:
        return FusionResult(
            decision="insufficient_evidence",
            reasons=[f"No policy clause mentions {peril} damage."],
            missing_evidence=["relevant policy clause"],
        )

    # Rule 3: the policy excludes the peril. Checked before the photos on purpose:
    # if the policy excludes hail, a matching hail photo does not change that.
    excluding = [c for c in relevant if says_excluded(c.text)]
    if excluding:
        reasons = []
        for c in excluding:
            reasons.append(f"{_where(c)} appears to exclude {peril} damage.")
            reasons.extend(_quote_line(c))
        return FusionResult(decision="possible_exclusion", reasons=reasons)

    # Rule 4: the photos must show the claimed damage (one matching usable photo is enough).
    if not any(f.damage_type == peril for f in usable):
        shown = ", ".join(sorted({_label(f.damage_type) for f in usable}))
        return FusionResult(
            decision="needs_review",
            reasons=[f"Usable photos show {shown}, but the claim is for {peril} damage."],
        )

    # Rule 5: the clauses must clearly grant coverage, not just mention the peril.
    covering = [c for c in relevant if says_covered(c.text)]
    if not covering:
        return FusionResult(
            decision="needs_review",
            reasons=[
                f"{_refs(relevant)} mentions {peril} damage but does not clearly grant coverage.",
                *(line for c in relevant[:3] for line in _quote_line(c)),
            ],
        )

    # Rule 6: weather confirmation, only for perils a weather record can confirm.
    if peril in WEATHER_PERILS:
        if claim.incident_date is None:
            return FusionResult(
                decision="needs_review",
                reasons=["The claim has no incident date, so the weather cannot be checked."],
                missing_evidence=["incident date"],
            )
        if weather is None or not weather.found:
            return FusionResult(
                decision="needs_review",
                reasons=[f"No weather record confirms a {peril} event on {claim.incident_date}.", *_weather_note(weather)],
                missing_evidence=["weather confirmation"],
            )

    # Rule 7: everything lines up.
    reasons = [f"Photos show {peril} damage."]
    if any(c.policy_name for c in covering):
        for c in covering[:3]:  # the three best matches are enough to check; the rest are in the citations tab
            reasons.append(f"{_where(c)} covers {peril} damage.")
            reasons.extend(_quote_line(c))
    else:
        reasons.append(f"Clause {_ids(covering)} covers {peril} damage.")
    if peril in WEATHER_PERILS:
        reasons.append(f"A weather record supports a {peril} event on {claim.incident_date}.")
        reasons.extend(_weather_note(weather))
    return FusionResult(decision="supported", reasons=reasons)