import json
from datetime import date
from pathlib import Path

from claimguard.agent.perils import WEATHER_PERILS
from claimguard.rag.search import search_policy
from claimguard.schemas.claim import Claim
from claimguard.schemas.evidence import PolicyEvidence, WeatherEvidence

# Which weather-record event names confirm which peril.
# Must have exactly the same keys as WEATHER_PERILS (a test guards this).
EVENT_NAMES = {
    "hail": {"hail", "hailstorm", "hailstones", "hailstone"},
    "wind": {"wind", "windstorm"},
}

WEATHER_EVENTS_FILE = Path("external") / "weather.json"


def build_policy_query(claim: Claim) -> str:
    """Build a search query from the claimed peril and component."""
    peril = claim.claimed_damage_type.replace("_", " ")

    if claim.component == "unclear":
        return f"{peril} damage"

    component = claim.component.replace("_", " ")
    return f"{peril} damage to {component}"


def policy_search_tool(
    claim: Claim,
    chroma_dir: Path | str | None = None,
) -> list[PolicyEvidence]:
    """Retrieve policy evidence for the claim's own policy only.

    No must_mention filter here: fuse() already ignores clauses about other perils.
    """
    return search_policy(
        query=build_policy_query(claim),
        policy_id=claim.policy_id,
        chroma_dir=chroma_dir,
    )


def weather_check_tool(
    claim: Claim,
    claim_dir: Path | str,
) -> WeatherEvidence | None:
    """Look for a weather event of the right kind on the incident date.

    Returns None when a weather check does not apply (peril not weather-related,
    or no incident date). Otherwise returns evidence with found=True/False.
    """
    peril = claim.claimed_damage_type

    if peril not in WEATHER_PERILS or claim.incident_date is None:
        return None

    weather_path = Path(claim_dir) / WEATHER_EVENTS_FILE

    if not weather_path.is_file():
        return WeatherEvidence(found=False, source="missing")

    try:
        records = json.loads(weather_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return WeatherEvidence(found=False, source="invalid")

    if not isinstance(records, list):
        return WeatherEvidence(found=False, source="invalid")

    accepted_events = EVENT_NAMES.get(peril, set())
    incident_date = claim.incident_date.isoformat()

    for record in records:
        if not isinstance(record, dict):
            continue

        event = record.get("event")
        event_date = record.get("date")

        if not isinstance(event, str) or not isinstance(event_date, str):
            continue

        normalized_event = " ".join(event.lower().split())

        try:
            normalized_date = date.fromisoformat(event_date).isoformat()
        except ValueError:
            continue

        if normalized_date == incident_date and normalized_event in accepted_events:
            return WeatherEvidence(
                event=event,
                date=normalized_date,
                location=record.get("location"),
                found=True,
                source=str(record.get("source", "mock")),
            )

    # The file was read fine, but nothing in it matched.
    return WeatherEvidence(found=False, source="no_match")