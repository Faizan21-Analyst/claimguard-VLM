import httpx
import pytest

from claimguard.schemas.claim import ClaimAssessment
from claimguard.ui.api_client import (
    ApiClient,
    ApiError,
    ApiUnreachable,
    ClaimNotFound,
    ServiceUnavailable,
)

RECORD = {
    "id": 7,
    "created_at": "2026-10-13T10:00:00+00:00",
    "assessment": ClaimAssessment(
        claim_id="claim_001", decision="needs_review", reasons=["Photos show wind."],
        model_version="fake",
    ).model_dump(mode="json"),
    "reviews": [],
}
CLAIM = {
    "claim_id": "claim_001", "policy_id": "policy_A", "claimed_damage_type": "hail",
    "component": "roof_shingles", "incident_date": "2026-06-10", "description": "x",
}


def client_for(handler) -> ApiClient:
    return ApiClient("http://api.test", transport=httpx.MockTransport(handler))


def respond(status=200, json=None, content=None):
    def handler(request):
        if content is not None:
            return httpx.Response(status, content=content)
        return httpx.Response(status, json=json)
    return handler


def test_list_claims():
    assert client_for(respond(json=["claim_001", "claim_002"])).list_claims() == ["claim_001", "claim_002"]


def test_get_claim_is_parsed():
    claim = client_for(respond(json=CLAIM)).get_claim("claim_001")
    assert claim.claimed_damage_type == "hail" and str(claim.incident_date) == "2026-06-10"


def test_analyze_posts_and_parses_the_record():
    seen = {}

    def handler(request):
        seen["method"], seen["path"] = request.method, request.url.path
        return httpx.Response(200, json=RECORD)

    record = client_for(handler).analyze("claim_001")

    assert (seen["method"], seen["path"]) == ("POST", "/claims/claim_001/analyze")
    assert record.id == 7 and record.assessment.decision == "needs_review"


def test_history_is_parsed():
    records = client_for(respond(json=[RECORD, RECORD])).history("claim_001")
    assert [r.id for r in records] == [7, 7]


def test_review_sends_action_and_note():
    seen = {}

    def handler(request):
        seen["path"], seen["body"] = request.url.path, request.read()
        return httpx.Response(200, json=RECORD)

    client_for(handler).review(7, "override", "Photos clearly show fire.")

    assert seen["path"] == "/assessments/7/review"
    assert b'"action":"override"' in seen["body"].replace(b" ", b"")
    assert b"Photos clearly show fire." in seen["body"]


def test_image_bytes_and_path_encoding():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, content=b"\x89PNGdata")

    data = client_for(handler).get_image_bytes("claim_001", "my photo.png")

    assert data == b"\x89PNGdata"
    assert seen["path"] == "/claims/claim_001/images/my photo.png"


def test_slashes_in_names_cannot_change_the_route():
    seen = {}

    def handler(request):
        seen["raw"] = request.url.raw_path.decode()
        return httpx.Response(200, content=b"x")

    client_for(handler).get_image_bytes("claim_001", "../claim.json")

    assert seen["raw"] == "/claims/claim_001/images/..%2Fclaim.json"


def test_404_becomes_claim_not_found():
    with pytest.raises(ClaimNotFound) as error:
        client_for(respond(404, json={"detail": "x"})).get_claim("nope")
    assert error.value.status_code == 404


def test_502_becomes_service_unavailable():
    with pytest.raises(ServiceUnavailable) as error:
        client_for(respond(502, json={"detail": "x"})).analyze("claim_001")
    assert "temporarily unavailable" in str(error.value)


def test_other_errors_keep_status_but_never_leak_server_text():
    secret = "Traceback ... api_key=sk-SECRET"
    with pytest.raises(ApiError) as error:
        client_for(respond(500, content=secret)).list_claims()

    assert error.value.status_code == 500
    assert "SECRET" not in str(error.value) and "Traceback" not in str(error.value)
    assert not isinstance(error.value, (ClaimNotFound, ServiceUnavailable, ApiUnreachable))


def test_connection_error_becomes_api_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiUnreachable) as error:
        client_for(handler).list_claims()
    assert "http://api.test" in str(error.value) and "uvicorn" in str(error.value)


def test_timeout_has_its_own_message():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ApiUnreachable) as error:
        client_for(handler).analyze("claim_001")
    assert "timed out" in str(error.value)


def test_non_json_body_is_an_unexpected_response():
    with pytest.raises(ApiError, match="unexpected response"):
        client_for(respond(200, content=b"<html>not json</html>")).list_claims()


def test_wrong_shape_is_an_unexpected_response():
    with pytest.raises(ApiError, match="unexpected response"):
        client_for(respond(200, json={"claim_id": "x"})).get_claim("claim_001")
    with pytest.raises(ApiError, match="unexpected response"):
        client_for(respond(200, json={"not": "a list"})).list_claims()