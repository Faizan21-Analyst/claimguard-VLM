"""API tests for "Try your own claim": uploads, validation, separate storage, analysis, delete."""
import io
from datetime import date, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIConnectionError
from PIL import Image

from claimguard.api.main import create_app
from claimguard.schemas.claim import ClaimAssessment
from configs.settings import load_settings


def jpeg(size=(60, 40), color="gray", fmt="JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


class FakeLiveGraph:
    def __init__(self, error=None):
        self.error = error
        self.states = []

    def invoke(self, state):
        if self.error:
            raise self.error
        self.states.append(state)
        return ClaimAssessment(
            claim_id=state["claim"].claim_id,
            decision="needs_review",
            reasons=["No weather record confirms a wind event on 2026-06-10."],
            model_version="fake-vlm",
        )


class FailIfCalled:
    def invoke(self, state):
        raise AssertionError("the demo graph must not be used for user claims")


@pytest.fixture
def env(tmp_path):
    claims = tmp_path / "claims" / "claim_001"
    claims.mkdir(parents=True)
    (claims / "claim.json").write_text(
        '{"claim_id": "claim_001", "policy_id": "policy_A", "claimed_damage_type": "hail"}', encoding="utf-8"
    )
    policies = tmp_path / "policies"
    policies.mkdir()
    for name in ("policy_A", "policy_B"):
        (policies / f"{name}.pdf").write_bytes(b"%PDF-1.4")

    live = FakeLiveGraph()
    app = create_app(
        db_path=tmp_path / "audit.db",
        claims_dir=tmp_path / "claims",
        graph=FailIfCalled(),
        user_db_path=tmp_path / "user.db",
        user_claims_dir=tmp_path / "user_claims",
        policies_dir=policies,
        live_graph=live,
    )
    with TestClient(app) as client:
        yield {"client": client, "live": live, "root": tmp_path}


def form(**overrides):
    data = {
        "policy_id": "policy_A",
        "claimed_damage_type": "wind",
        "component": "roof_shingles",
        "incident_date": "2026-06-10",
        "location": "Mumbai, India",
        "description": "Shingles blew off.",
    }
    data.update(overrides)
    return data


def photos(*blobs):
    blobs = blobs or (jpeg(),)
    return [("photos", (f"p{i}.jpg", blob, "image/jpeg")) for i, blob in enumerate(blobs, start=1)]


def submit(env, files=None, **overrides):
    return env["client"].post("/user-claims", data=form(**overrides), files=files if files is not None else photos())


def test_policies_are_listed(env):
    assert env["client"].get("/policies").json() == ["policy_A", "policy_B"]


def test_a_valid_claim_is_stored_with_clean_photos(env):
    response = submit(env, files=photos(jpeg(), jpeg(color="red", fmt="PNG")))

    assert response.status_code == 201
    claim = response.json()
    assert claim["claim_id"].startswith("user_")
    assert (claim["claimed_damage_type"], claim["location"], claim["incident_date"]) == ("wind", "Mumbai, India", "2026-06-10")

    client = env["client"]
    assert client.get(f"/user-claims/{claim['claim_id']}/images").json() == ["photo_1.jpg", "photo_2.jpg"]
    image = client.get(f"/user-claims/{claim['claim_id']}/images/photo_2.jpg")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(image.content)).format == "JPEG"   # the PNG was converted


def test_user_data_is_kept_apart_from_the_demo_claims(env):
    claim_id = submit(env).json()["claim_id"]
    client = env["client"]

    assert client.get("/claims").json() == ["claim_001"]
    assert client.get("/user-claims").json() == [claim_id]
    assert (env["root"] / "user.db").is_file()
    assert (env["root"] / "user_claims" / claim_id / "images" / "photo_1.jpg").is_file()
    assert not (env["root"] / "claims" / claim_id).exists()
    assert client.get(f"/claims/{claim_id}/assessments").status_code == 404   # not a demo claim


def test_newest_claim_is_listed_first(env):
    first = submit(env).json()["claim_id"]
    second = submit(env).json()["claim_id"]
    assert env["client"].get("/user-claims").json() == [second, first]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"policy_id": "policy_Z"}, "Unknown policy"),
        ({"claimed_damage_type": "none"}, "must be one of"),
        ({"claimed_damage_type": "tornado"}, "must be one of"),
        ({"component": "garage"}, "Unknown part"),
        ({"incident_date": ""}, "incident date and the location"),
        ({"location": "   "}, "incident date and the location"),
        ({"incident_date": "10/06/2026"}, "not valid"),
        ({"incident_date": (date.today() + timedelta(days=30)).isoformat()}, "in the future"),
        ({"location": "x" * 121}, "too long"),
        ({"description": "x" * 1001}, "too long"),
    ],
)
def test_invalid_forms_are_refused_with_a_readable_reason(env, overrides, message):
    response = submit(env, **overrides)

    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert env["client"].get("/user-claims").json() == []   # nothing was saved


def test_water_and_fire_claims_do_not_need_a_date_or_location(env):
    response = submit(env, claimed_damage_type="water", incident_date="", location="")
    assert response.status_code == 201
    assert response.json()["incident_date"] is None and response.json()["location"] is None


def test_at_least_one_photo_is_required(env):
    response = submit(env, files=[])
    assert response.status_code == 422 and "at least one photo" in response.json()["detail"]


def test_too_many_photos_are_refused(env):
    limit = load_settings().api.max_photos
    response = submit(env, files=photos(*[jpeg() for _ in range(limit + 1)]))
    assert response.status_code == 422 and f"at most {limit}" in response.json()["detail"]


@pytest.mark.parametrize("blob", [b"not an image at all", b"", jpeg(fmt="GIF")])
def test_files_that_are_not_jpeg_or_png_photos_are_refused(env, blob):
    response = submit(env, files=photos(jpeg(), blob))

    assert response.status_code == 422 and response.json()["detail"].startswith("Photo 2:")
    assert env["client"].get("/user-claims").json() == []                    # even the good photo was not kept
    assert not any((env["root"] / "user_claims").glob("*")) if (env["root"] / "user_claims").exists() else True


def test_oversized_photos_are_refused(env, monkeypatch):
    monkeypatch.setenv("API_MAX_PHOTO_MB", "1")
    load_settings.cache_clear()
    try:
        app = create_app(
            db_path=env["root"] / "audit2.db", claims_dir=env["root"] / "claims", graph=FailIfCalled(),
            user_db_path=env["root"] / "user2.db", user_claims_dir=env["root"] / "uc2",
            policies_dir=env["root"] / "policies", live_graph=FakeLiveGraph(),
        )
        with TestClient(app) as client:
            response = client.post("/user-claims", data=form(), files=photos(b"x" * (1024 * 1024 + 1)))
    finally:
        load_settings.cache_clear()

    assert response.status_code == 413 and "larger than 1 MB" in response.json()["detail"]


def test_analyze_uses_the_live_graph_and_saves_an_audit_record(env):
    claim_id = submit(env).json()["claim_id"]
    client = env["client"]

    response = client.post(f"/user-claims/{claim_id}/analyze")

    assert response.status_code == 200
    assert response.json()["assessment"]["claim_id"] == claim_id
    state = env["live"].states[0]
    assert state["claim"].location == "Mumbai, India"            # the live weather tool will receive this
    assert (state["claim_dir"] / "images" / "photo_1.jpg").is_file()

    history = client.get(f"/user-claims/{claim_id}/assessments").json()
    assert [h["id"] for h in history] == [response.json()["id"]]


def test_reviews_work_for_user_claim_assessments(env):
    claim_id = submit(env).json()["claim_id"]
    record = env["client"].post(f"/user-claims/{claim_id}/analyze").json()

    reviewed = env["client"].post(f"/assessments/{record['id']}/review", json={"action": "accept", "note": "ok"})

    assert reviewed.status_code == 200 and reviewed.json()["reviews"][0]["action"] == "accept"


@pytest.mark.parametrize("claim_id", ["user_doesnotexist", "claim_001", "..%2Fuser", "a.b"])
def test_unknown_or_unsafe_ids_return_404(env, claim_id):
    client = env["client"]
    assert client.get(f"/user-claims/{claim_id}").status_code == 404
    assert client.post(f"/user-claims/{claim_id}/analyze").status_code == 404
    assert client.get(f"/user-claims/{claim_id}/images").status_code == 404
    assert client.delete(f"/user-claims/{claim_id}").status_code == 404


def test_unknown_image_name_returns_404(env):
    claim_id = submit(env).json()["claim_id"]
    client = env["client"]

    assert client.get(f"/user-claims/{claim_id}/images/nope.jpg").status_code == 404
    assert client.get(f"/user-claims/{claim_id}/images/..%2F..%2Fuser.db").status_code == 404


def test_groq_failure_returns_a_generic_502_and_saves_nothing(env):
    env["live"].error = APIConnectionError(request=httpx.Request("POST", "https://example.com"))
    claim_id = submit(env).json()["claim_id"]

    response = env["client"].post(f"/user-claims/{claim_id}/analyze")

    assert response.status_code == 502
    assert response.json() == {"detail": "The assessment service is temporarily unavailable."}
    assert env["client"].get(f"/user-claims/{claim_id}/assessments").json() == []


def test_delete_removes_photos_and_form_but_keeps_the_audit_record(env):
    claim_id = submit(env).json()["claim_id"]
    client = env["client"]
    record = client.post(f"/user-claims/{claim_id}/analyze").json()

    assert client.delete(f"/user-claims/{claim_id}").status_code == 204

    assert client.get(f"/user-claims/{claim_id}").status_code == 404
    assert client.get("/user-claims").json() == []
    assert not (env["root"] / "user_claims" / claim_id).exists()
    assert client.get(f"/assessments/{record['id']}").status_code == 200   # audit rows are append-only
