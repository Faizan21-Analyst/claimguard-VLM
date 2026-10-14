import json
from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from claimguard.agent.fusion import FusionResult, fuse
from claimguard.agent.perils import clause_mentions_peril, says_covered, says_excluded
from claimguard.rag.clauses import split_clauses
from claimguard.schemas.claim import Claim
from claimguard.schemas.evidence import PolicyEvidence, WeatherEvidence
from claimguard.schemas.visual import VisualFinding

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
ALLOWED = {"supported", "needs_review", "insufficient_evidence", "possible_exclusion"}


# ---------- helpers ----------

def photo(damage="hail", quality="good") -> VisualFinding:
    return VisualFinding(damage_type=damage, severity="moderate",
                         component="roof_shingles", image_quality=quality)


def clause(cid: str, text: str) -> PolicyEvidence:
    return PolicyEvidence(clause_id=cid, text=text, relevance=0.9)


def make_claim(peril="hail", incident_date="2026-06-10") -> Claim:
    return Claim(claim_id="claim_x", policy_id="policy_X",
                 claimed_damage_type=peril, component="roof_shingles",
                 incident_date=incident_date)


FOUND = WeatherEvidence(found=True, event="hailstorm", date="2026-06-10", source="synthetic")
NOT_FOUND = WeatherEvidence(found=False)

COVER_HAIL = clause("4.2", "Hail damage to roof shingles is covered.")
EXCLUDE_HAIL = clause("4.1", "Hail damage to roofs is excluded under this policy.")
UNRELATED_EXCLUSION = clause("6.1", "Gradual wear and tear is excluded.")
TRAP_WATER = clause(
    "3.1",
    "Sudden and accidental water damage to ceilings is covered "
    "when the event is not otherwise excluded.",
)


# ---------- the six demo claims (linked to data/synthetic/expected_decisions.json) ----------

def load_claim(claim_id: str) -> Claim:
    path = DATA / "claims" / claim_id / "claim.json"
    return Claim.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_weather(claim_id: str) -> WeatherEvidence:
    # Matching event names ("hail" vs "hailstorm") is the weather tool's job in Module 6.
    # Here a record simply means "found".
    path = DATA / "claims" / claim_id / "external" / "weather.json"
    events = json.loads(path.read_text(encoding="utf-8"))
    if not events:
        return WeatherEvidence(found=False)
    first = events[0]
    return WeatherEvidence(found=True, event=first.get("event"), date=first.get("date"),
                           location=first.get("location"), source=first.get("source", "synthetic"))


def all_clauses(policy_id: str) -> list[PolicyEvidence]:
    """EVERY clause of the real policy PDF, as if retrieval returned everything.
    This makes fuse() prove it ignores clauses about other perils."""
    with pymupdf.open(DATA / "policies" / f"{policy_id}.pdf") as pdf:
        text = "\n".join(page.get_text("text") for page in pdf)
    return [PolicyEvidence(clause_id=cid, text=body, relevance=1.0)
            for cid, body in split_clauses(text)]


EXPECTED = json.loads((DATA / "expected_decisions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("claim_id", sorted(EXPECTED))
def test_demo_claim_gets_expected_decision(claim_id):
    claim = load_claim(claim_id)
    result = fuse(claim, [photo(claim.claimed_damage_type)],
                  all_clauses(claim.policy_id), load_weather(claim_id))
    assert result.decision == EXPECTED[claim_id], result.reasons


def test_every_demo_claim_folder_has_an_expected_decision():
    folders = {p.name for p in (DATA / "claims").glob("claim_*")}
    assert folders == set(EXPECTED)


# ---------- rule 1: photos ----------

def test_no_photos_is_insufficient():
    r = fuse(make_claim(), [], [COVER_HAIL], FOUND)
    assert r.decision == "insufficient_evidence"
    assert "clear damage photos" in r.missing_evidence


def test_poor_photo_is_insufficient():
    assert fuse(make_claim(), [photo(quality="poor")], [COVER_HAIL], FOUND).decision == "insufficient_evidence"


def test_unclear_photo_is_insufficient():
    assert fuse(make_claim(), [photo("unclear")], [COVER_HAIL], FOUND).decision == "insufficient_evidence"


def test_poor_photo_beats_exclusion():
    r = fuse(make_claim(), [photo(quality="poor")], [EXCLUDE_HAIL], FOUND)
    assert r.decision == "insufficient_evidence"


# ---------- rule 2: relevant clause ----------

def test_no_relevant_clause_is_insufficient():
    fire_only = clause("3.1", "Accidental fire damage is covered.")
    r = fuse(make_claim(), [photo()], [fire_only], FOUND)
    assert r.decision == "insufficient_evidence"
    assert "relevant policy clause" in r.missing_evidence


def test_empty_policy_is_insufficient():
    assert fuse(make_claim(), [photo()], [], FOUND).decision == "insufficient_evidence"


def test_unspecific_claimed_damage_is_insufficient():
    r = fuse(make_claim("unclear"), [photo("hail")], [COVER_HAIL], FOUND)
    assert r.decision == "insufficient_evidence"


# ---------- rule 3: exclusion ----------

def test_exclusion_beats_matching_photo():
    assert fuse(make_claim(), [photo()], [EXCLUDE_HAIL], FOUND).decision == "possible_exclusion"


def test_exclusion_wins_over_coverage_when_both_present():
    assert fuse(make_claim(), [photo()], [COVER_HAIL, EXCLUDE_HAIL], FOUND).decision == "possible_exclusion"


def test_exclusion_about_another_peril_is_ignored():
    r = fuse(make_claim(), [photo()], [COVER_HAIL, UNRELATED_EXCLUSION], FOUND)
    assert r.decision == "supported"


# ---------- rule 4: photo must match ----------

def test_photo_of_other_damage_needs_review():
    r = fuse(make_claim("hail"), [photo("fire")], [COVER_HAIL], FOUND)
    assert r.decision == "needs_review"
    assert "fire" in " ".join(r.reasons)


def test_photo_showing_no_damage_needs_review():
    r = fuse(make_claim(), [photo("none")], [COVER_HAIL], FOUND)
    assert r.decision == "needs_review"
    assert "no visible damage" in " ".join(r.reasons)


# ---------- rule 5: coverage must be explicit ----------

def test_clause_without_coverage_language_needs_review():
    only_a_condition = clause("4.9", "Hail damage must be reported within 30 days.")
    assert fuse(make_claim(), [photo()], [only_a_condition], FOUND).decision == "needs_review"


# ---------- rule 6: weather ----------

def test_weather_not_found_needs_review():
    r = fuse(make_claim(), [photo()], [COVER_HAIL], NOT_FOUND)
    assert r.decision == "needs_review"
    assert "weather confirmation" in r.missing_evidence


def test_weather_not_checked_needs_review():
    assert fuse(make_claim(), [photo()], [COVER_HAIL], None).decision == "needs_review"


def test_missing_incident_date_needs_review():
    r = fuse(make_claim(incident_date=None), [photo()], [COVER_HAIL], FOUND)
    assert r.decision == "needs_review"
    assert "incident date" in r.missing_evidence


def test_water_claim_does_not_need_weather():
    water = make_claim("water", incident_date=None)
    assert fuse(water, [photo("water")], [TRAP_WATER], None).decision == "supported"


# ---------- rule 7: supported ----------

def test_supported_explains_itself():
    r = fuse(make_claim(), [photo()], [COVER_HAIL], FOUND)
    assert r.decision == "supported" and r.missing_evidence == []
    text = " ".join(r.reasons)
    assert "4.2" in text and "weather" in text


def test_one_good_photo_is_enough_even_if_another_is_poor():
    assert fuse(make_claim(), [photo(quality="poor"), photo()], [COVER_HAIL], FOUND).decision == "supported"


def test_one_matching_photo_is_enough_even_if_another_shows_something_else():
    assert fuse(make_claim(), [photo("wind"), photo("hail")], [COVER_HAIL], FOUND).decision == "supported"


def test_vlm_confidence_is_ignored():
    low = VisualFinding(damage_type="hail", severity="moderate", component="roof_shingles", confidence=0.0)
    assert fuse(make_claim(), [low], [COVER_HAIL], FOUND).decision == "supported"


# ---------- helpers in perils.py ----------

def test_trap_clause_is_not_an_exclusion():
    assert says_covered(TRAP_WATER.text)
    assert not says_excluded(TRAP_WATER.text)


def test_phrase_split_across_lines_still_matches():
    assert says_excluded("Hail damage is\nexcluded under this policy.")


def test_peril_matching_uses_whole_words():
    assert not clause_mentions_peril("Broken windows are covered.", "wind")
    assert clause_mentions_peril("Damage from windstorms is covered.", "wind")
    assert clause_mentions_peril("A confirmed hailstorm is covered.", "hail")
    assert clause_mentions_peril("FIRE damage is covered.", "fire")
    assert not clause_mentions_peril("anything", "unclear")


# ---------- output contract ----------

def test_result_cannot_have_empty_reasons():
    with pytest.raises(ValidationError):
        FusionResult(decision="supported", reasons=[])


CASES = [
    ([], [COVER_HAIL], FOUND),
    ([photo()], [], FOUND),
    ([photo()], [EXCLUDE_HAIL], FOUND),
    ([photo("fire")], [COVER_HAIL], FOUND),
    ([photo()], [COVER_HAIL], NOT_FOUND),
    ([photo()], [COVER_HAIL], FOUND),
]


@pytest.mark.parametrize("findings,policy,weather", CASES)
def test_every_path_gives_a_reason_and_an_allowed_decision(findings, policy, weather):
    r = fuse(make_claim(), findings, policy, weather)
    assert r.decision in ALLOWED and r.reasons