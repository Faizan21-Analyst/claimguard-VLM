"""Pure display helpers: no Streamlit, no HTTP. Easy to test."""
from datetime import datetime, timezone

from claimguard.agent.perils import says_covered, says_excluded
from claimguard.api.models import AssessmentRecord
from claimguard.schemas.claim import ClaimAssessment
from claimguard.schemas.evidence import PolicyEvidence, WeatherEvidence
from claimguard.schemas.visual import VisualFinding

# Icon AND words, so meaning never depends on colour alone.
DECISION_LABELS: dict[str, tuple[str, str]] = {
    "supported": ("✅", "Supported by the evidence"),
    "needs_review": ("🔎", "Needs human review"),
    "insufficient_evidence": ("❓", "Not enough evidence"),
    "possible_exclusion": ("⚠️", "Possible policy exclusion"),
}

# What the reviewer picks, and how a saved review is described afterwards.
REVIEW_ACTION_LABELS: dict[str, str] = {
    "accept": "Accept the assessment",
    "request_more": "Request more evidence",
    "override": "Override (a note is required)",
}
REVIEW_DONE_LABELS: dict[str, str] = {
    "accept": "Accepted",
    "request_more": "More evidence requested",
    "override": "Overridden",
}

WEATHER_SOURCE_NOTES: dict[str, str] = {
    "missing": "no weather file for this claim",
    "invalid": "the weather file could not be read",
    "no_match": "no event of the right kind on that date",
    # live weather ("Try your own claim")
    "open-meteo": "the weather data for that day does not show the claimed event",
    "no_location": "no location was given",
    "location_not_found": "the location could not be found",
    "weather_service_error": "the weather service could not be reached",
    "future_date": "the incident date is in the future",
    "no_data": "the weather service has no data for that day",
}

WEATHER_ATTRIBUTION = (
    "Live weather from Open-Meteo.com (reanalysis data on a 9-25 km grid, not a weather-station report). "
    "It can miss small local storms, so 'not confirmed' is not proof that nothing happened."
)


def decision_label(decision: str) -> str:
    icon, text = DECISION_LABELS.get(decision, ("❔", decision))
    return f"{icon} {text}"


# ---------- policy citations: which document, which section, which words ----------

# Words, not just colour: what role a clause plays for the claim. Same word lists the decision rules use.
CITATION_ROLES = ("Exclusion", "Coverage", "Related term")


def citation_role(clause: PolicyEvidence) -> str:
    if says_excluded(clause.text):
        return "Exclusion"
    if says_covered(clause.text):
        return "Coverage"
    return "Related term"


def sort_citations(clauses: list[PolicyEvidence]) -> list[PolicyEvidence]:
    """Clauses that can change the decision first (exclusions, then coverage), then the best match first."""
    return sorted(clauses, key=lambda c: (CITATION_ROLES.index(citation_role(c)), -c.relevance))


def citation_title(clause: PolicyEvidence, fallback_policy: str = "") -> str:
    """'HomeShield Homeowners Policy (Form HS-100) · Section 3.4 · Hail · page 3'.

    Demo clauses know no document title, so the claim's policy name is used instead."""
    parts = [clause.policy_name or fallback_policy, f"Section {clause.clause_id}", clause.section_title or ""]
    if clause.page:
        parts.append(f"page {clause.page}")
    return " · ".join(part for part in parts if part)


def policy_basis(clauses: list[PolicyEvidence], limit: int = 3) -> list[PolicyEvidence]:
    """The few clauses a reviewer should read first: exclusions, then coverage. Neutral clauses are left out."""
    return [c for c in sort_citations(clauses) if citation_role(c) != "Related term"][:limit]


def verdict_recap(assessment: ClaimAssessment, fallback_policy: str = "") -> list[tuple[str, str]]:
    """One plain line per kind of evidence, then the verdict. Built from the saved record, no AI involved."""
    shown = sorted(
        {
            "no visible damage" if f.damage_type == "none" else f"{f.damage_type.replace('_', ' ')} damage"
            for f in assessment.visual
            if f.image_quality == "good" and f.damage_type != "unclear"
        }
    )
    photos = f"Photos show {', '.join(shown)}." if shown else "No photo could be judged clearly."

    basis = policy_basis(assessment.policy_citations, limit=1)
    if basis:
        verb = "Excluded by" if citation_role(basis[0]) == "Exclusion" else "Covered by"
        policy = f"{verb} {citation_title(basis[0], fallback_policy)}."
    else:
        policy = "No clause that grants or excludes cover for this damage was found."

    return [
        ("Photos", photos),
        ("Policy", policy),
        ("Weather", describe_weather(assessment.weather)),
        ("Verdict", decision_label(assessment.decision)),
    ]


def describe_weather(weather: WeatherEvidence | None) -> str:
    if weather is None:
        return "Not applicable for this claim type."
    if weather.found:
        where = f" in {weather.location}" if weather.location else ""
        text = f"Confirmed: {weather.event} on {weather.date}{where} (source: {weather.source})."
    else:
        reason = WEATHER_SOURCE_NOTES.get(weather.source, weather.source)
        text = f"No matching weather record found ({reason})."
    return f"{text} {weather.details}" if weather.details else text


def format_confidence(value: float) -> str:
    return f"{value:.0%}"


def format_time(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def last_review_action(record: AssessmentRecord) -> str:
    """Reviews arrive oldest-first, so the newest is the last one."""
    if not record.reviews:
        return "Not reviewed yet"
    action = record.reviews[-1].action
    return REVIEW_DONE_LABELS.get(action, action)


def finding_rows(findings: list[VisualFinding]) -> list[dict[str, str]]:
    return [
        {
            "Photo": str(number),
            "Damage": f.damage_type.replace("_", " "),
            "Severity": f.severity,
            "Part": f.component.replace("_", " "),
            "Photo quality": f.image_quality,
            "Model confidence": format_confidence(f.confidence),
            "What the model noted": "; ".join(f.findings) or "-",
        }
        for number, f in enumerate(findings, start=1)
    ]


def history_rows(records: list[AssessmentRecord]) -> list[dict[str, str]]:
    return [
        {
            "ID": str(r.id),
            "Time": format_time(r.created_at),
            "Decision": decision_label(r.assessment.decision),
            "Model": r.assessment.model_version or "-",
            "Review": last_review_action(r),
        }
        for r in records
    ]