import pytest
from pydantic import ValidationError

from claimguard.schemas.claim import Claim, ClaimAssessment
from claimguard.schemas.visual import VisualFinding

GOOD = dict(damage_type="hail", severity="moderate", component="roof_shingles")


def failed_fields(exc_info) -> set[str]:
    """Names of the fields that caused the validation error."""
    return {e["loc"][0] for e in exc_info.value.errors()}


def test_valid_visual_finding_passes():
    f = VisualFinding(**GOOD, findings=["circular impact marks"], confidence=0.85)
    assert f.damage_type == "hail" and f.image_quality == "good"


def test_defaults_are_applied():
    f = VisualFinding(**GOOD)
    assert f.findings == [] and f.confidence == 0.5


def test_bad_damage_type_is_rejected():
    with pytest.raises(ValidationError) as e:
        VisualFinding(**{**GOOD, "damage_type": "earthquake"})
    assert failed_fields(e) == {"damage_type"}


def test_confidence_above_one_is_rejected():
    with pytest.raises(ValidationError) as e:
        VisualFinding(**GOOD, confidence=1.5)
    assert failed_fields(e) == {"confidence"}


def test_too_many_findings_is_rejected():
    with pytest.raises(ValidationError) as e:
        VisualFinding(**GOOD, findings=["x"] * 7)
    assert failed_fields(e) == {"findings"}


def test_bad_decision_is_rejected():
    with pytest.raises(ValidationError) as e:
        ClaimAssessment(claim_id="CG-001", decision="approved", reasons=["x"])
    assert failed_fields(e) == {"decision"}


def test_bad_incident_date_is_rejected():
    with pytest.raises(ValidationError) as e:
        Claim(claim_id="CG-001", policy_id="policy_A",
              claimed_damage_type="hail", incident_date="2026-13-45")
    assert failed_fields(e) == {"incident_date"}