import sqlite3

import pytest

from claimguard.api import db
from claimguard.schemas.claim import ClaimAssessment


def make_assessment(claim_id="claim_001"):
    return ClaimAssessment(
        claim_id=claim_id,
        decision="supported",
        reasons=["The policy supports the claimed peril."],
        model_version="test-vlm",
        summary="The available evidence supports this claim.",
    )


def test_init_db_is_safe_to_call_twice(tmp_path):
    path = tmp_path / "nested" / "audit.db"
    db.init_db(path)
    db.init_db(path)

    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {"assessments", "reviews"}.issubset(tables)


def test_assessment_round_trip(tmp_path):
    path = tmp_path / "audit.db"
    db.init_db(path)
    assessment = make_assessment()

    assessment_id = db.save_assessment(path, assessment)
    row = db.get_assessment(path, assessment_id)

    assert row is not None
    restored = ClaimAssessment.model_validate_json(row["assessment_json"])
    assert restored == assessment


def test_history_is_newest_first(tmp_path):
    path = tmp_path / "audit.db"
    db.init_db(path)

    first_id = db.save_assessment(path, make_assessment())
    second_id = db.save_assessment(path, make_assessment())

    history = db.list_assessments(path, "claim_001")

    assert [row["id"] for row in history] == [second_id, first_id]


def test_review_for_missing_assessment_violates_foreign_key(tmp_path):
    path = tmp_path / "audit.db"
    db.init_db(path)

    with pytest.raises(sqlite3.IntegrityError):
        db.save_review(path, 99999, "accept", "")


def test_review_does_not_modify_assessment_json(tmp_path):
    path = tmp_path / "audit.db"
    db.init_db(path)
    assessment_id = db.save_assessment(path, make_assessment())

    before = db.get_assessment(path, assessment_id)["assessment_json"]
    db.save_review(path, assessment_id, "accept", "Reviewed and accepted.")
    after = db.get_assessment(path, assessment_id)["assessment_json"]

    assert after == before
    assert len(db.get_reviews(path, assessment_id)) == 1
