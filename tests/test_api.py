import json
import shutil
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIConnectionError

from claimguard.api.main import create_app
from claimguard.schemas.claim import ClaimAssessment


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CLAIMS = REPO_ROOT / "data" / "synthetic" / "claims"


def make_assessment(claim_id="claim_001"):
    return ClaimAssessment(
        claim_id=claim_id,
        decision="supported",
        reasons=["The evidence supports the claimed damage."],
        model_version="fake-vlm",
        summary="The available evidence supports the claim.",
    )


class FakeGraph:
    def __init__(self, assessment=None, error=None):
        self.assessment = assessment or make_assessment()
        self.error = error

    def invoke(self, state):
        if self.error:
            raise self.error
        return self.assessment


@pytest.fixture
def claims_dir(tmp_path):
    destination = tmp_path / "claims"
    shutil.copytree(DEMO_CLAIMS, destination)
    return destination


@pytest.fixture
def client(tmp_path, claims_dir):
    app = create_app(
        db_path=tmp_path / "audit.db",
        claims_dir=claims_dir,
        graph=FakeGraph(),
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_list_claims(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    response = client.get("/claims")
    assert response.status_code == 200
    assert response.json() == sorted(
        folder.name
        for folder in DEMO_CLAIMS.iterdir()
        if folder.is_dir() and (folder / "claim.json").is_file()
    )


def test_analyze_and_read_assessment(client):
    response = client.post("/claims/claim_001/analyze")
    assert response.status_code == 200

    created = response.json()
    assessment_id = created["id"]
    assert created["assessment"]["decision"] == "supported"

    fetched = client.get(f"/assessments/{assessment_id}")
    assert fetched.status_code == 200
    assert fetched.json()["assessment"]["decision"] == "supported"


def test_analyzing_twice_creates_two_audit_records(client):
    first = client.post("/claims/claim_001/analyze")
    second = client.post("/claims/claim_001/analyze")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] != second.json()["id"]

    history = client.get("/claims/claim_001/assessments")
    assert history.status_code == 200
    assert len(history.json()) == 2
    assert history.json()[0]["id"] == second.json()["id"]


def test_unknown_and_unsafe_claim_ids_return_404(client):
    assert client.post("/claims/not_a_claim/analyze").status_code == 404
    assert client.post("/claims/../analyze").status_code in (404, 405)
    assert client.get("/claims/../assessments").status_code in (404, 405)


def test_broken_claim_json_returns_422(tmp_path, claims_dir):
    broken = claims_dir / "broken_claim"
    broken.mkdir()
    (broken / "claim.json").write_text("{invalid json", encoding="utf-8")

    app = create_app(
        db_path=tmp_path / "audit.db",
        claims_dir=claims_dir,
        graph=FakeGraph(),
    )
    with TestClient(app) as test_client:
        response = test_client.post("/claims/broken_claim/analyze")

    assert response.status_code == 422


def test_groq_api_error_returns_generic_502(tmp_path, claims_dir):
    error = APIConnectionError(
        request=httpx.Request("POST", "https://example.com")
    )
    app = create_app(
        db_path=tmp_path / "audit.db",
        claims_dir=claims_dir,
        graph=FakeGraph(error=error),
    )

    with TestClient(app) as test_client:
        response = test_client.post("/claims/claim_001/analyze")

    assert response.status_code == 502
    assert "traceback" not in response.text.lower()
    assert "sk-" not in response.text


def test_unexpected_error_returns_500(tmp_path, claims_dir):
    app = create_app(
        db_path=tmp_path / "audit.db",
        claims_dir=claims_dir,
        graph=FakeGraph(error=RuntimeError("unexpected test failure")),
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/claims/claim_001/analyze")

    assert response.status_code == 500
    assert "unexpected test failure" not in response.text


def test_review_is_saved_and_returned(client):
    created = client.post("/claims/claim_001/analyze").json()
    assessment_id = created["id"]

    response = client.post(
        f"/assessments/{assessment_id}/review",
        json={"action": "accept", "note": "Reviewed."},
    )
    assert response.status_code == 200
    assert response.json()["reviews"][0]["action"] == "accept"

    fetched = client.get(f"/assessments/{assessment_id}")
    assert fetched.status_code == 200
    assert fetched.json()["reviews"][0]["note"] == "Reviewed."


@pytest.mark.parametrize(
    "body",
    [
        {"action": "override", "note": ""},
        {"action": "override", "note": "   "},
        {"action": "unknown_action", "note": "test"},
    ],
)
def test_invalid_review_body_returns_422(client, body):
    created = client.post("/claims/claim_001/analyze").json()
    response = client.post(
        f"/assessments/{created['id']}/review",
        json=body,
    )
    assert response.status_code == 422


def test_review_for_unknown_assessment_returns_404(client):
    response = client.post(
        "/assessments/999999/review",
        json={"action": "accept"},
    )
    assert response.status_code == 404


def test_unknown_assessment_read_returns_404(client):
    assert client.get("/assessments/999999").status_code == 404
