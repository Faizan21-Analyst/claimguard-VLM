"""The live-claim additions to the pipeline, and proof that the demo path is unchanged."""
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from claimguard.agent import graph as graph_module
from claimguard.agent.explain import _build_prompt, summarize
from claimguard.agent.fusion import FusionResult, fuse
from claimguard.agent.graph import analyze_claim, build_graph
from claimguard.schemas.claim import Claim
from claimguard.schemas.evidence import PolicyEvidence, WeatherEvidence
from claimguard.schemas.visual import VisualFinding

COVERS_HAIL = PolicyEvidence(clause_id="4.2", page=1, text="Hail damage to the roof is covered.", relevance=0.9)


def hail_claim(**extra) -> Claim:
    return Claim(claim_id="c1", policy_id="policy_A", claimed_damage_type="hail", component="roof_shingles",
                 incident_date=date(2026, 6, 10), **extra)


def hail_finding() -> VisualFinding:
    return VisualFinding(damage_type="hail", severity="moderate", component="roof_shingles", image_quality="good")


class FakeVLM:
    name = "fake-vlm"

    def analyze(self, image_path):
        return hail_finding(), True


@pytest.fixture
def claim_dir(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"x")
    (tmp_path / "claim.json").write_text(hail_claim().model_dump_json(), encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def stub_policy_search(monkeypatch):
    # No vector database needed: these tests are about the weather step and the summary.
    monkeypatch.setattr(graph_module, "policy_search_tool", lambda claim, chroma_dir=None: [COVERS_HAIL])


# ---------- graph ----------

def test_default_graph_still_reads_the_demo_weather_file(claim_dir):
    (claim_dir / "external").mkdir()
    (claim_dir / "external" / "weather.json").write_text(
        json.dumps([{"event": "hailstorm", "date": "2026-06-10", "location": "Demo City", "source": "synthetic"}]),
        encoding="utf-8",
    )
    calls = []

    def old_style_summarizer(result, claim):          # exactly the old two-argument shape
        calls.append((result.decision, claim.claim_id))
        return "summary"

    assessment = analyze_claim(build_graph(FakeVLM(), summarizer=old_style_summarizer), claim_dir)

    assert assessment.decision == "supported"
    assert assessment.weather.found and assessment.weather.source == "synthetic"
    assert assessment.weather.details is None
    assert assessment.summary == "summary" and calls == [("supported", "c1")]


def test_live_graph_uses_the_given_weather_tool_and_gives_the_summary_the_facts(claim_dir):
    seen = {}

    def live_weather(claim, directory):
        seen["claim"] = claim.claim_id
        return WeatherEvidence(event="hail", date="2026-06-10", location="Pune, India", found=True,
                               source="open-meteo", details="Pune on 2026-06-10: max gust 80 km/h")

    def summarizer(result, claim, facts=None):
        seen["facts"] = facts
        return "ok"

    graph = build_graph(FakeVLM(), summarizer=summarizer, weather_tool=live_weather, rich_summary=True)
    assessment = analyze_claim(graph, claim_dir)

    assert seen["claim"] == "c1"
    assert assessment.weather.source == "open-meteo"
    assert seen["facts"][0] == "Photo 1: hail damage, severity moderate, part roof shingles."
    assert seen["facts"][1].startswith("Weather data supports the claimed event: Pune on 2026-06-10")
    assert any(r.startswith("Weather data: Pune") for r in assessment.reasons)   # also in the official record


def test_facts_describe_unconfirmed_weather_and_unreadable_photos(claim_dir):
    class UnclearVLM:
        name = "unclear"

        def analyze(self, image_path):
            return VisualFinding(damage_type="unclear", severity="unclear", component="unclear",
                                 image_quality="poor"), True

    captured = {}
    graph = build_graph(
        UnclearVLM(),
        summarizer=lambda result, claim, facts=None: captured.setdefault("facts", facts) and "ok",
        weather_tool=lambda claim, d: WeatherEvidence(found=False, source="open-meteo", details="calm day"),
        rich_summary=True,
    )
    analyze_claim(graph, claim_dir)

    assert captured["facts"][0] == "Photo 1: could not be judged (unclear or poor quality)."
    assert captured["facts"][1] == "Weather data does not confirm the claimed event: calm day"


def test_the_decision_never_depends_on_the_summary(claim_dir):
    weather = lambda claim, d: WeatherEvidence(event="hail", found=True, source="open-meteo", details="d")
    plain = analyze_claim(build_graph(FakeVLM(), summarizer=None, weather_tool=weather, rich_summary=True), claim_dir)
    fancy = analyze_claim(
        build_graph(FakeVLM(), summarizer=lambda r, c, facts=None: "different words", weather_tool=weather,
                    rich_summary=True),
        claim_dir,
    )
    assert plain.decision == fancy.decision == "supported"
    assert plain.reasons == fancy.reasons


def test_a_failing_summary_still_gives_a_decision_on_the_live_path(claim_dir):
    def boom(result, claim, facts=None):
        raise RuntimeError("LLM down")

    graph = build_graph(FakeVLM(), summarizer=boom, rich_summary=True,
                        weather_tool=lambda c, d: WeatherEvidence(event="hail", found=True, source="open-meteo", details="d"))
    assessment = analyze_claim(graph, claim_dir)

    assert assessment.decision == "supported" and assessment.summary == ""


# ---------- decision rules ----------

def test_weather_numbers_are_added_to_the_reasons_only_when_they_exist():
    live = WeatherEvidence(event="hail", found=True, source="open-meteo", details="gust 80 km/h")
    mock = WeatherEvidence(event="hailstorm", date="2026-06-10", found=True, source="synthetic")

    with_details = fuse(hail_claim(), [hail_finding()], [COVERS_HAIL], live)
    without = fuse(hail_claim(), [hail_finding()], [COVERS_HAIL], mock)

    assert with_details.decision == without.decision == "supported"
    assert with_details.reasons[-1] == "Weather data: gust 80 km/h"
    assert len(with_details.reasons) == len(without.reasons) + 1
    assert not any("Weather data" in r for r in without.reasons)


def test_unconfirmed_live_weather_goes_to_human_review_with_the_numbers():
    weather = WeatherEvidence(found=False, source="open-meteo", details="max gust 20 km/h")

    result = fuse(hail_claim(), [hail_finding()], [COVERS_HAIL], weather)

    assert result.decision == "needs_review"
    assert result.reasons[0].startswith("No weather record confirms a hail event")
    assert result.reasons[1] == "Weather data: max gust 20 km/h"
    assert result.missing_evidence == ["weather confirmation"]


def test_a_failed_weather_lookup_never_produces_a_supported_decision():
    for source in ("weather_service_error", "location_not_found", "no_location", "no_data", "future_date"):
        result = fuse(hail_claim(), [hail_finding()], [COVERS_HAIL], WeatherEvidence(found=False, source=source))
        assert result.decision == "needs_review", source


# ---------- reasoning-model prompt ----------

def fusion_result() -> FusionResult:
    return FusionResult(decision="needs_review", reasons=["No weather record confirms a hail event on 2026-06-10."])


def test_prompt_is_unchanged_without_facts():
    prompt = _build_prompt(fusion_result(), hail_claim())
    assert "Evidence behind the decision" not in prompt
    assert prompt == _build_prompt(fusion_result(), hail_claim(), None) == _build_prompt(fusion_result(), hail_claim(), [])


def test_prompt_lists_the_facts_when_given():
    prompt = _build_prompt(fusion_result(), hail_claim(), ["Photo 1: hail damage.", "Weather data does not confirm it: calm"])

    assert "Evidence behind the decision" in prompt
    assert "- Photo 1: hail damage." in prompt and "- Weather data does not confirm it: calm" in prompt
    assert "say so plainly" in prompt


def test_summarize_sends_the_facts_to_the_model():
    sent = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    sent["prompt"] = kwargs["messages"][0]["content"]
                    choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="A plain summary."))
                    return SimpleNamespace(choices=[choice])

    text = summarize(fusion_result(), hail_claim(), client=FakeClient(), facts=["Photo 1: hail damage."])

    assert text == "A plain summary." and "Photo 1: hail damage." in sent["prompt"]
