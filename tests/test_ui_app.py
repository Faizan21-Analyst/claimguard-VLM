"""Smoke tests that drive the real Streamlit script with a fake API client (no browser, no network)."""
import io
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from claimguard.api.models import AssessmentRecord, ReviewRecord  # noqa: E402
from claimguard.schemas.claim import Claim, ClaimAssessment  # noqa: E402
from claimguard.schemas.evidence import PolicyEvidence, WeatherEvidence  # noqa: E402
from claimguard.schemas.visual import VisualFinding  # noqa: E402
from claimguard.ui.api_client import ApiUnreachable, ServiceUnavailable  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeClient:
    """Same methods as ApiClient. Records every call so tests can count them."""

    def __init__(self, has_history=False, list_error=None, analyze_error=None):
        self.list_error = list_error
        self.analyze_error = analyze_error
        self.records: list[AssessmentRecord] = []
        self.analyze_calls = 0
        self.review_calls: list[tuple] = []
        if has_history:
            self._add_record()

    def _add_record(self) -> AssessmentRecord:
        record = AssessmentRecord(
            id=len(self.records) + 1,
            created_at=datetime(2026, 10, 13, 10, 0, tzinfo=timezone.utc),
            assessment=ClaimAssessment(
                claim_id="claim_001",
                decision="needs_review",
                reasons=["Usable photos show wind, but the claim is for hail damage."],
                visual=[VisualFinding(damage_type="wind", severity="severe", component="roof_shingles",
                                      findings=["missing shingles"], confidence=0.95)],
                policy_citations=[PolicyEvidence(clause_id="4.2", page=1, text="Hail is covered.", relevance=0.86)],
                weather=WeatherEvidence(found=True, event="hailstorm", date="2026-06-10", source="synthetic"),
                summary="The photos show wind damage, not hail.",
                model_version="fake-vlm",
            ),
        )
        self.records.insert(0, record)  # newest first, like the API
        return record

    def list_claims(self):
        if self.list_error:
            raise self.list_error
        return ["claim_001", "claim_002"]

    def get_claim(self, claim_id):
        return Claim(claim_id=claim_id, policy_id="policy_A", claimed_damage_type="hail",
                     component="roof_shingles", incident_date="2026-06-10", description="Roof hit by hail.")

    def list_images(self, claim_id):
        return ["roof.png"]

    def get_image_bytes(self, claim_id, filename):
        return png()

    def analyze(self, claim_id):
        self.analyze_calls += 1
        if self.analyze_error:
            raise self.analyze_error
        return self._add_record()

    def history(self, claim_id):
        return list(self.records)

    def review(self, assessment_id, action, note=""):
        self.review_calls.append((assessment_id, action, note))
        for i, record in enumerate(self.records):
            if record.id == assessment_id:
                review = ReviewRecord(id=len(record.reviews) + 1, action=action, note=note,
                                      created_at=datetime(2026, 10, 13, 11, 0, tzinfo=timezone.utc))
                self.records[i] = record.model_copy(update={"reviews": [*record.reviews, review]})
                return self.records[i]


@pytest.fixture
def run_app(monkeypatch):
    def start(fake: FakeClient) -> AppTest:
        st.cache_data.clear()
        st.cache_resource.clear()
        monkeypatch.setattr("claimguard.ui.api_client.ApiClient", lambda *args, **kwargs: fake)
        return AppTest.from_file(str(APP), default_timeout=30).run()
    return start


def texts(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def test_unanalyzed_claim_shows_claim_card_and_a_hint(run_app):
    at = run_app(FakeClient())

    assert not at.exception
    assert "Claim claim_001" in texts(at.subheader)
    assert "has not been analyzed yet" in texts(at.info)
    assert "Roof hit by hail." in texts(at.markdown)
    assert len(at.image) == 1  # the claim's one photo is shown


def test_run_analysis_shows_decision_reasons_and_summary(run_app):
    fake = FakeClient()
    at = run_app(fake)

    at.sidebar.button[0].click().run()

    assert not at.exception
    assert fake.analyze_calls == 1
    assert "Needs human review" in texts(at.header)
    assert "claim is for hail" in texts(at.markdown)
    assert "wind damage, not hail" in texts(at.info)


def test_clicking_around_after_analysis_never_runs_it_again(run_app):
    fake = FakeClient()
    at = run_app(fake)
    at.sidebar.button[0].click().run()

    at.text_area[0].set_value("typing a note").run()
    at.radio[0].set_value("request_more").run()

    assert fake.analyze_calls == 1
    assert len(fake.records) == 1  # no extra audit records


def test_override_without_a_note_is_refused_in_the_ui(run_app):
    fake = FakeClient(has_history=True)
    at = run_app(fake)

    at.radio[0].set_value("override")
    at.button[0].click().run()

    assert "needs a note" in texts(at.error)
    assert fake.review_calls == []


def test_accepting_saves_a_review_and_confirms(run_app):
    fake = FakeClient(has_history=True)
    at = run_app(fake)

    at.radio[0].set_value("accept")
    at.button[0].click().run()

    assert fake.review_calls == [(1, "accept", "")]
    assert "Review saved." in texts(at.success)
    assert "Accepted" in texts(at.caption)  # the assessment header now shows the review status


def test_api_down_shows_a_friendly_message_and_does_not_crash(run_app):
    at = run_app(FakeClient(list_error=ApiUnreachable("Cannot reach the ClaimGuard API at http://x.")))

    assert not at.exception
    assert "Cannot reach the ClaimGuard API" in texts(at.error)


def test_failed_analysis_shows_a_friendly_message(run_app):
    fake = FakeClient(analyze_error=ServiceUnavailable("The assessment service is temporarily unavailable."))
    at = run_app(fake)

    at.sidebar.button[0].click().run()

    assert not at.exception
    assert "temporarily unavailable" in texts(at.error)
    assert fake.records == []