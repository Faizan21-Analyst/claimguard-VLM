import json
import re
from pathlib import Path

import pymupdf

from claimguard.schemas.claim import Claim
from claimguard.schemas.evidence import WeatherEvidence

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
CLAIM_DIRS = sorted(DATA.glob("claims/claim_*"))
CLAUSE_LINE = re.compile(r"^\d+(\.\d+)*\s+[A-Z]", re.M)
ALL_DECISIONS = {"supported", "needs_review", "possible_exclusion", "insufficient_evidence"}


def load_expected() -> dict[str, str]:
    return json.loads((DATA / "expected_decisions.json").read_text(encoding="utf-8"))


def test_six_claims_exist_and_validate():
    assert len(CLAIM_DIRS) == 6
    for d in CLAIM_DIRS:
        claim = Claim.model_validate(json.loads((d / "claim.json").read_text(encoding="utf-8")))
        assert claim.claim_id == d.name


def test_every_claim_points_to_an_existing_policy():
    for d in CLAIM_DIRS:
        claim = Claim.model_validate(json.loads((d / "claim.json").read_text(encoding="utf-8")))
        assert (DATA / "policies" / f"{claim.policy_id}.pdf").exists(), d.name


def test_expected_decisions_match_claim_folders():
    expected = load_expected()
    assert set(expected) == {d.name for d in CLAIM_DIRS}
    assert set(expected.values()) == ALL_DECISIONS


def test_weather_files_are_valid():
    for d in CLAIM_DIRS:
        events = json.loads((d / "external" / "weather.json").read_text(encoding="utf-8"))
        assert isinstance(events, list)
        for e in events:
            WeatherEvidence.model_validate(e)


def test_only_needs_review_claim_has_no_weather_record():
    expected = load_expected()
    for d in CLAIM_DIRS:
        events = json.loads((d / "external" / "weather.json").read_text(encoding="utf-8"))
        assert (events == []) == (expected[d.name] == "needs_review"), d.name


def test_policy_pdfs_have_numbered_clauses():
    pdfs = sorted((DATA / "policies").glob("*.pdf"))
    assert len(pdfs) == 3
    for pdf in pdfs:
        text = "".join(page.get_text() for page in pymupdf.open(pdf))
        assert len(CLAUSE_LINE.findall(text)) >= 2, pdf.name