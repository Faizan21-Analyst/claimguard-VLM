from datetime import datetime, timezone
from typing import get_args

from claimguard.api.models import AssessmentRecord, ReviewRecord
from claimguard.schemas.claim import ClaimAssessment, Decision
from claimguard.schemas.evidence import WeatherEvidence
from claimguard.schemas.visual import VisualFinding
from claimguard.ui.formatting import (
    DECISION_LABELS,
    REVIEW_ACTION_LABELS,
    REVIEW_DONE_LABELS,
    decision_label,
    describe_weather,
    finding_rows,
    format_confidence,
    format_time,
    history_rows,
    last_review_action,
)

T0 = datetime(2026, 10, 13, 10, 0, tzinfo=timezone.utc)


def make_record(reviews=(), decision="needs_review") -> AssessmentRecord:
    return AssessmentRecord(
        id=3,
        created_at=T0,
        assessment=ClaimAssessment(
            claim_id="claim_001", decision=decision, reasons=["r"], model_version="fake-vlm"
        ),
        reviews=list(reviews),
    )


def review(action, note="") -> ReviewRecord:
    return ReviewRecord(id=1, action=action, note=note, created_at=T0)


def test_every_decision_has_a_label():
    # Fails the moment a fifth decision is added without updating the UI.
    assert set(DECISION_LABELS) == set(get_args(Decision))


def test_every_review_action_has_labels():
    assert set(REVIEW_ACTION_LABELS) == set(REVIEW_DONE_LABELS) == {"accept", "request_more", "override"}


def test_decision_label_has_icon_and_words():
    assert decision_label("needs_review") == "🔎 Needs human review"


def test_unknown_decision_does_not_crash():
    assert "mystery" in decision_label("mystery")


def test_weather_not_applicable():
    assert describe_weather(None) == "Not applicable for this claim type."


def test_weather_found():
    text = describe_weather(
        WeatherEvidence(found=True, event="hailstorm", date="2026-06-10", location="Demo City", source="synthetic")
    )
    assert "Confirmed" in text and "hailstorm" in text and "Demo City" in text


def test_weather_found_without_location():
    assert " in None" not in describe_weather(WeatherEvidence(found=True, event="hailstorm", date="2026-06-10"))


def test_weather_not_found_explains_why():
    assert "no weather file" in describe_weather(WeatherEvidence(found=False, source="missing"))
    assert "right kind" in describe_weather(WeatherEvidence(found=False, source="no_match"))


def test_format_confidence():
    assert format_confidence(0.954) == "95%"


def test_format_time_is_utc_and_handles_naive_times():
    assert format_time(T0) == "2026-10-13 10:00 UTC"
    assert format_time(datetime(2026, 10, 13, 10, 0)) == "2026-10-13 10:00 UTC"


def test_last_review_action():
    assert last_review_action(make_record()) == "Not reviewed yet"
    record = make_record([review("request_more"), review("accept")])
    assert last_review_action(record) == "Accepted"


def test_finding_rows():
    rows = finding_rows([
        VisualFinding(damage_type="wear_and_tear", severity="minor", component="roof_shingles",
                      findings=["moss", "fading"], confidence=0.5),
        VisualFinding(damage_type="hail", severity="severe", component="gutter"),
    ])
    assert rows[0]["Photo"] == "1" and rows[0]["Damage"] == "wear and tear"
    assert rows[0]["What the model noted"] == "moss; fading" and rows[0]["Model confidence"] == "50%"
    assert rows[1]["What the model noted"] == "notes unavailable"


def test_history_rows():
    rows = history_rows([make_record([review("override")], decision="supported")])
    assert rows == [{
        "ID": "3", "Time": "2026-10-13 10:00 UTC", "Decision": "✅ Supported by the evidence",
        "Model": "fake-vlm", "Review": "Overridden",
    }]