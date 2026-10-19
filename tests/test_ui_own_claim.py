"""UI side of "Try your own claim": the HTTP client, the display text, and the Streamlit screen."""
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from PIL import Image

from claimguard.api.models import AssessmentRecord
from claimguard.schemas.claim import Claim, ClaimAssessment
from claimguard.schemas.evidence import WeatherEvidence
from claimguard.schemas.visual import VisualFinding
from claimguard.ui.api_client import ApiClient, ApiError, ApiUnreachable, ClaimNotFound, InvalidInput
from claimguard.ui.formatting import WEATHER_SOURCE_NOTES, describe_weather

CLAIM_JSON = {"claim_id": "user_abc", "policy_id": "policy_A", "claimed_damage_type": "wind",
              "component": "roof_shingles", "incident_date": "2026-06-10", "location": "Mumbai, India",
              "description": "", }


def client_with(handler) -> ApiClient:
    return ApiClient("http://api", transport=httpx.MockTransport(handler))


def create_args(**overrides):
    args = dict(policy_id="policy_A", claimed_damage_type="wind", component="roof_shingles",
                incident_date="2026-06-10", location="Mumbai, India", description="",
                photos=[("a.jpg", b"JPEGDATA", "image/jpeg")])
    args.update(overrides)
    return args


# ---------- API client ----------

def test_create_sends_a_multipart_form_and_parses_the_claim():
    seen = {}

    def handler(request):
        seen["method"], seen["path"] = request.method, request.url.path
        seen["type"], seen["body"] = request.headers["content-type"], request.read()
        return httpx.Response(201, json=CLAIM_JSON)

    claim = client_with(handler).create_user_claim(**create_args())

    assert (seen["method"], seen["path"]) == ("POST", "/user-claims")
    assert seen["type"].startswith("multipart/form-data")
    assert b'name="policy_id"' in seen["body"] and b'name="location"' in seen["body"]
    assert b'name="photos"; filename="a.jpg"' in seen["body"] and b"JPEGDATA" in seen["body"]
    assert isinstance(claim, Claim) and claim.claim_id == "user_abc" and claim.location == "Mumbai, India"


@pytest.mark.parametrize("status", [413, 422])
def test_form_refusals_show_our_own_message(status):
    handler = lambda request: httpx.Response(status, json={"detail": "Photo 1: That file is not a readable photo."})

    with pytest.raises(InvalidInput, match="not a readable photo") as excinfo:
        client_with(handler).create_user_claim(**create_args())
    assert excinfo.value.status_code == status


def test_framework_validation_lists_are_never_shown():
    handler = lambda request: httpx.Response(422, json={"detail": [{"loc": ["body", "x"], "msg": "secret internals"}]})

    with pytest.raises(InvalidInput) as excinfo:
        client_with(handler).create_user_claim(**create_args())
    assert "secret internals" not in str(excinfo.value)


def test_other_422_errors_stay_generic():
    handler = lambda request: httpx.Response(422, json={"detail": "internal wording"})

    with pytest.raises(ApiError) as excinfo:
        client_with(handler).analyze_user_claim("user_abc")
    assert not isinstance(excinfo.value, InvalidInput) and "internal wording" not in str(excinfo.value)


def test_create_reports_an_unreachable_api():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiUnreachable):
        client_with(handler).create_user_claim(**create_args())


def test_read_methods_use_the_user_claim_routes():
    paths = []

    def handler(request):
        paths.append((request.method, request.url.path))
        route = request.url.path
        if route == "/policies":
            return httpx.Response(200, json=["policy_A"])
        if route == "/user-claims":
            return httpx.Response(200, json=["user_abc"])
        if route.endswith("/images"):
            return httpx.Response(200, json=["photo_1.jpg"])
        if "/images/" in route:
            return httpx.Response(200, content=b"bytes")
        if route.endswith("/assessments"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=CLAIM_JSON)

    client = client_with(handler)

    assert client.list_policies() == ["policy_A"]
    assert client.list_user_claims() == ["user_abc"]
    assert client.get_user_claim("user_abc").claim_id == "user_abc"
    assert client.list_user_images("user_abc") == ["photo_1.jpg"]
    assert client.get_user_image_bytes("user_abc", "photo_1.jpg") == b"bytes"
    assert client.user_history("user_abc") == []
    assert paths == [
        ("GET", "/policies"), ("GET", "/user-claims"), ("GET", "/user-claims/user_abc"),
        ("GET", "/user-claims/user_abc/images"), ("GET", "/user-claims/user_abc/images/photo_1.jpg"),
        ("GET", "/user-claims/user_abc/assessments"),
    ]


def test_delete_works_and_missing_claims_are_reported():
    assert client_with(lambda request: httpx.Response(204)).delete_user_claim("user_abc") is None
    with pytest.raises(ClaimNotFound):
        client_with(lambda request: httpx.Response(404)).delete_user_claim("user_abc")


def test_ids_cannot_change_the_route():
    seen = []
    client_with(lambda request: seen.append(request.url.raw_path) or httpx.Response(204)).delete_user_claim("../x")
    assert seen == [b"/user-claims/..%2Fx"]


# ---------- display text ----------

def test_demo_weather_text_is_unchanged():
    found = WeatherEvidence(found=True, event="hailstorm", date="2026-06-10", location="Demo City", source="synthetic")
    assert describe_weather(found) == "Confirmed: hailstorm on 2026-06-10 in Demo City (source: synthetic)."
    assert describe_weather(WeatherEvidence(found=False, source="no_match")) == (
        "No matching weather record found (no event of the right kind on that date)."
    )


def test_live_weather_text_includes_the_numbers():
    weather = WeatherEvidence(found=False, source="open-meteo", details="Mumbai on 2026-06-10: max gust 35 km/h")
    text = describe_weather(weather)

    assert "does not show the claimed event" in text and text.endswith("max gust 35 km/h")


@pytest.mark.parametrize("source", ["no_location", "location_not_found", "weather_service_error", "future_date", "no_data"])
def test_every_live_failure_has_a_readable_reason(source):
    assert source in WEATHER_SOURCE_NOTES
    assert source not in describe_weather(WeatherEvidence(found=False, source=source))


# ---------- the Streamlit screen ----------

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeClient:
    def __init__(self, user_ids=("user_abc",)):
        self.user_ids = list(user_ids)
        self.records = []
        self.demo_analyze_calls = 0
        self.user_analyze_calls = 0
        self.deleted = []

    # demo side
    def list_claims(self):
        return ["claim_001"]

    def get_claim(self, claim_id):
        return Claim(claim_id=claim_id, policy_id="policy_A", claimed_damage_type="hail", component="roof_shingles")

    def list_images(self, claim_id):
        return []

    def get_image_bytes(self, claim_id, filename):
        return png()

    def analyze(self, claim_id):
        self.demo_analyze_calls += 1

    def history(self, claim_id):
        return []

    # own-claim side
    def list_policies(self):
        return ["policy_A", "policy_B"]

    def list_user_claims(self):
        return list(self.user_ids)

    def get_user_claim(self, claim_id):
        return Claim.model_validate({**CLAIM_JSON, "claim_id": claim_id, "description": "Roof damaged."})

    def list_user_images(self, claim_id):
        return ["photo_1.jpg"]

    def get_user_image_bytes(self, claim_id, filename):
        return png()

    def analyze_user_claim(self, claim_id):
        self.user_analyze_calls += 1
        record = AssessmentRecord(
            id=len(self.records) + 1,
            created_at=datetime(2026, 10, 19, 10, 0, tzinfo=timezone.utc),
            assessment=ClaimAssessment(
                claim_id=claim_id,
                decision="needs_review",
                reasons=["No weather record confirms a wind event on 2026-06-10."],
                visual=[VisualFinding(damage_type="wind", severity="moderate", component="roof_shingles")],
                weather=WeatherEvidence(found=False, source="open-meteo", location="Mumbai, Maharashtra, India",
                                        details="Mumbai on 2026-06-10: max gust 35 km/h"),
                summary="The weather data does not confirm a wind event.",
                model_version="fake-vlm",
            ),
        )
        self.records.insert(0, record)
        return record

    def user_history(self, claim_id):
        return list(self.records)

    def delete_user_claim(self, claim_id):
        self.deleted.append(claim_id)
        self.user_ids.remove(claim_id)


@pytest.fixture
def run_app(monkeypatch):
    def start(fake, before_run=None) -> AppTest:
        st.cache_data.clear()
        st.cache_resource.clear()
        monkeypatch.setattr("claimguard.ui.api_client.ApiClient", lambda *args, **kwargs: fake)
        at = AppTest.from_file(str(APP), default_timeout=30)
        if before_run:
            before_run(at)
        return at.run()
    return start


def texts(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def switch_to_own(at: AppTest) -> AppTest:
    return at.sidebar.radio[0].set_value("Try your own claim").run()


def test_demo_mode_is_still_the_default(run_app):
    at = run_app(FakeClient())

    assert not at.exception
    assert at.sidebar.radio[0].value == "Demo claims"
    assert "Claim claim_001" in texts(at.subheader)


def test_own_mode_starts_on_the_new_claim_form(run_app):
    at = switch_to_own(run_app(FakeClient()))

    assert not at.exception
    assert "Try your own claim" in texts(at.subheader)
    assert len(at.date_input) == 1 and len(at.text_input) == 1
    assert at.sidebar.selectbox[0].value == "__new__"


def test_picking_a_saved_claim_shows_its_card_and_location(run_app):
    at = switch_to_own(run_app(FakeClient()))
    at.sidebar.selectbox(key="own_claim_choice").select("user_abc").run()

    assert not at.exception
    assert "Claim user_abc" in texts(at.subheader)
    assert "Mumbai, India" in texts(at.caption)
    assert len(at.image) == 1


def test_run_analysis_uses_the_user_claim_route_and_shows_the_live_weather(run_app):
    fake = FakeClient()
    at = switch_to_own(run_app(fake))
    at.sidebar.selectbox(key="own_claim_choice").select("user_abc").run()

    at.sidebar.button[0].click().run()

    assert not at.exception
    assert (fake.user_analyze_calls, fake.demo_analyze_calls) == (1, 0)
    assert "Needs human review" in texts(at.header)
    assert "does not confirm a wind event" in texts(at.info)
    assert "Open-Meteo.com" in texts(at.caption)          # attribution + honest limits are shown
    assert "max gust 35 km/h" in texts(at.markdown)


def test_a_freshly_submitted_claim_is_analyzed_automatically_once(run_app):
    fake = FakeClient()

    at = run_app(fake)
    at = at.sidebar.radio[0].set_value("Try your own claim").run()
    at.session_state["pending_own_choice"] = "user_abc"
    at.session_state["auto_analyze"] = "user_abc"
    at.run()

    assert not at.exception
    assert fake.user_analyze_calls == 1
    at.run()   # any later rerun must not analyze again
    assert fake.user_analyze_calls == 1


def test_delete_removes_the_claim_and_returns_to_the_form(run_app):
    fake = FakeClient()
    at = switch_to_own(run_app(fake))
    at.sidebar.selectbox(key="own_claim_choice").select("user_abc").run()

    delete = next(b for b in at.sidebar.button if b.label == "Delete my photos and claim")
    delete.click().run()

    assert fake.deleted == ["user_abc"]
    assert not at.exception
    assert "were deleted" in texts(at.success)
    assert at.sidebar.selectbox[0].value == "__new__"


def test_switching_back_to_demo_claims_works(run_app):
    at = switch_to_own(run_app(FakeClient()))
    at.sidebar.radio[0].set_value("Demo claims").run()

    assert not at.exception and "Claim claim_001" in texts(at.subheader)
