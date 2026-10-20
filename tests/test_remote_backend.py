"""Tests for the remote GPU backend, the fallback, and the honesty rules around them.

No network, no GPU, no gradio_client: the Hugging Face client is replaced by a small fake.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from claimguard.agent.graph import build_graph
from claimguard.api.main import create_app
from claimguard.schemas.claim import Claim
from claimguard.ui.api_client import ApiClient, ServiceUnavailable
from claimguard.vlm import factory, remote_backend
from claimguard.vlm.base import SAFE_FALLBACK, ModelUnavailable, VLMBackend
from claimguard.vlm.fallback import FallbackVLM
from claimguard.vlm.images import image_to_png_bytes
from claimguard.vlm.remote_backend import RemoteFinetunedVLM

VALID = {"damage_type": "hail", "severity": "moderate", "component": "roof_shingles", "image_quality": "good"}
VALID_TEXT = json.dumps(VALID)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------- fakes ----------

class FakeJob:
    def __init__(self, outcome):
        self.outcome = outcome
        self.cancelled = False

    def result(self, timeout=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def cancel(self):
        self.cancelled = True


class FakeClient:
    """Stands in for gradio_client.Client: records what was sent, answers from a script."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.api_names = []
        self.sent = []
        self.jobs = []

    def submit(self, photo, api_name=None):
        self.api_names.append(api_name)
        self.sent.append(Path(photo).read_bytes())   # the temporary file must exist while the call is made
        job = FakeJob(self.outcomes.pop(0))
        self.jobs.append(job)
        return job


class FakeBackend(VLMBackend):
    def __init__(self, name, *outputs):
        self.name = name
        self.outputs = list(outputs)
        self.calls = 0

    def _raw(self, image_path):
        self.calls += 1
        out = self.outputs.pop(0) if self.outputs else VALID_TEXT
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "roof.png"
    Image.new("RGB", (300, 200), "gray").save(path)
    return path


@pytest.fixture(autouse=True)
def _remote_settings(monkeypatch):
    vlm = SimpleNamespace(
        remote_space="someone/claimguard-vision", remote_token=None, remote_api_name="/analyze",
        remote_timeout_s=5.0, local_max_side=64,
    )
    monkeypatch.setattr(remote_backend, "load_settings", lambda: SimpleNamespace(vlm=vlm))
    monkeypatch.setattr(remote_backend, "_handle_file", lambda path: path)


# ---------- the remote backend ----------

def test_remote_answer_is_parsed_like_any_other_backend(photo):
    client = FakeClient(VALID_TEXT)
    finding, ok = RemoteFinetunedVLM(client=client).analyze(photo)
    assert ok is True and finding.damage_type == "hail"
    assert client.api_names == ["/analyze"]


def test_photo_is_sent_as_lossless_png_at_training_size(photo):
    client = FakeClient(VALID_TEXT)
    RemoteFinetunedVLM(client=client).analyze(photo)
    sent = client.sent[0]
    assert sent.startswith(PNG_MAGIC)
    import io
    assert max(Image.open(io.BytesIO(sent)).size) <= 64   # local_max_side in the test settings


def test_out_of_quota_is_unavailable_not_an_unclear_photo(photo):
    client = FakeClient(RuntimeError("You have exceeded your GPU quota (60s requested vs. 12s left)"))
    with pytest.raises(ModelUnavailable, match="no free time left"):
        RemoteFinetunedVLM(client=client).analyze(photo)
    assert client.jobs[0].cancelled is True


def test_timeout_is_unavailable(photo):
    with pytest.raises(ModelUnavailable, match="did not answer in time"):
        RemoteFinetunedVLM(client=FakeClient(TimeoutError("took too long"))).analyze(photo)


def test_any_transport_error_is_unavailable_never_a_value_error(photo):
    # A ValueError here would be swallowed by analyze() and turned into an "unclear" photo.
    with pytest.raises(ModelUnavailable):
        RemoteFinetunedVLM(client=FakeClient(ValueError("Could not fetch config for someone/space"))).analyze(photo)


def test_error_text_from_the_server_never_leaks_into_the_message(photo):
    client = FakeClient(RuntimeError("Authorization failed for token hf_SECRET123"))
    with pytest.raises(ModelUnavailable) as caught:
        RemoteFinetunedVLM(client=client).analyze(photo)
    assert "hf_SECRET123" not in str(caught.value)


def test_empty_model_output_is_bad_output_so_the_photo_is_unclear(photo):
    finding, ok = RemoteFinetunedVLM(client=FakeClient("   ", "   ")).analyze(photo)
    assert ok is False and finding == SAFE_FALLBACK


def test_missing_space_name_fails_fast(monkeypatch):
    vlm = SimpleNamespace(remote_space="", remote_token=None, remote_api_name="/analyze",
                          remote_timeout_s=5.0, local_max_side=64)
    monkeypatch.setattr(remote_backend, "load_settings", lambda: SimpleNamespace(vlm=vlm))
    with pytest.raises(ValueError, match="VLM_REMOTE_SPACE"):
        RemoteFinetunedVLM()


# ---------- the fallback ----------

def test_primary_answers_when_it_can():
    primary, secondary = FakeBackend("remote_ft:x"), FakeBackend("groq:y")
    finding, ok, name = FallbackVLM(primary, secondary).analyze_traced(Path("p.jpg"))
    assert ok and name == "remote_ft:x" and secondary.calls == 0


def test_fallback_is_used_and_named_when_primary_is_unavailable():
    primary = FakeBackend("remote_ft:x", ModelUnavailable("no GPU time"))
    secondary = FakeBackend("groq:y")
    finding, ok, name = FallbackVLM(primary, secondary).analyze_traced(Path("p.jpg"))
    assert ok and finding.damage_type == "hail"
    assert "groq:y" in name and "fallback" in name and "remote_ft:x" in name   # the record says who really answered


def test_primary_is_skipped_during_the_cooldown_then_retried():
    now = [100.0]
    primary = FakeBackend("remote_ft:x", ModelUnavailable("no GPU time"))
    secondary = FakeBackend("groq:y")
    vlm = FallbackVLM(primary, secondary, cooldown_s=300, clock=lambda: now[0])

    vlm.analyze_traced(Path("p.jpg"))          # primary fails once
    vlm.analyze_traced(Path("p.jpg"))          # inside the cooldown: primary is not even asked
    assert primary.calls == 1 and secondary.calls == 2

    now[0] += 301
    _, _, name = vlm.analyze_traced(Path("p.jpg"))   # cooldown over: primary is tried again and now works
    assert primary.calls == 2 and name == "remote_ft:x"


def test_bad_output_from_the_primary_does_not_trigger_the_fallback():
    primary = FakeBackend("remote_ft:x", "not json", "still not json")
    secondary = FakeBackend("groq:y")
    finding, ok, name = FallbackVLM(primary, secondary).analyze_traced(Path("p.jpg"))
    assert ok is False and finding.damage_type == "unclear" and name == "remote_ft:x" and secondary.calls == 0


def test_a_failing_fallback_is_not_hidden():
    primary = FakeBackend("remote_ft:x", ModelUnavailable("no GPU time"))
    secondary = FakeBackend("groq:y", RuntimeError("groq is down"))
    with pytest.raises(RuntimeError, match="groq is down"):
        FallbackVLM(primary, secondary).analyze_traced(Path("p.jpg"))


# ---------- what ends up in the saved assessment ----------

def _claim():
    return Claim(
        claim_id="c1", policy_id="auto", claimed_damage_type="hail", component="roof_shingles",
        incident_date=None, location=None, description="test",
    )


def _claim_dir(tmp_path, photos=2):
    images = tmp_path / "claim" / "images"
    images.mkdir(parents=True)
    for number in range(1, photos + 1):
        Image.new("RGB", (40, 30), "gray").save(images / f"photo_{number}.jpg")
    return tmp_path / "claim"


def _run(vlm, tmp_path):
    graph = build_graph(
        vlm, summarizer=None, weather_tool=lambda claim, claim_dir: None, policy_tool=lambda claim, chroma_dir=None: []
    )
    return graph.invoke({"claim": _claim(), "claim_dir": _claim_dir(tmp_path)})["assessment"]


def test_assessment_names_the_fallback_model(tmp_path):
    primary = FakeBackend("remote_ft:x", ModelUnavailable("no GPU time"))
    vlm = FallbackVLM(primary, FakeBackend("groq:y"))
    assert "groq:y (fallback: remote_ft:x unavailable)" in _run(vlm, tmp_path).model_version


def test_assessment_names_the_plain_backend_as_before(tmp_path):
    assert _run(FakeBackend("groq:y"), tmp_path).model_version == "groq:y"


def test_a_duck_typed_fake_without_analyze_traced_still_works(tmp_path):
    class Duck:
        name = "duck"

        def analyze(self, path):
            return FakeBackend("d").analyze(path)[0], True

    assert _run(Duck(), tmp_path).model_version == "duck"


# ---------- factory ----------

def _settings(monkeypatch, backend, fallback="groq"):
    vlm = SimpleNamespace(backend=backend, remote_fallback=fallback, remote_cooldown_s=1.0)
    monkeypatch.setattr("configs.settings.load_settings", lambda: SimpleNamespace(vlm=vlm))
    monkeypatch.setattr(factory, "_build", lambda name: FakeBackend(name))


def test_app_default_wraps_remote_ft_with_the_fallback(monkeypatch):
    _settings(monkeypatch, "remote_ft")
    vlm = factory.get_vlm()
    assert isinstance(vlm, FallbackVLM) and vlm.primary.name == "remote_ft" and vlm.secondary.name == "groq"


def test_fallback_can_be_switched_off(monkeypatch):
    _settings(monkeypatch, "remote_ft", fallback="none")
    assert not isinstance(factory.get_vlm(), FallbackVLM)


def test_other_backends_are_never_wrapped(monkeypatch):
    _settings(monkeypatch, "groq")
    assert not isinstance(factory.get_vlm(), FallbackVLM)


def test_an_explicit_name_is_never_wrapped(monkeypatch):
    # Evaluation scripts ask for a backend by name: the score must belong to exactly that model.
    _settings(monkeypatch, "remote_ft")
    assert not isinstance(factory.get_vlm("remote_ft"), FallbackVLM)


def test_unknown_backend_still_lists_the_choices():
    with pytest.raises(ValueError, match="remote_ft"):
        factory.get_vlm("nope")


# ---------- API and UI client ----------

def test_api_answers_503_when_the_model_is_unavailable(tmp_path):
    claims = tmp_path / "claims"
    (claims / "c1").mkdir(parents=True)
    (claims / "c1" / "claim.json").write_text(_claim().model_dump_json(), encoding="utf-8")

    class Down:
        def invoke(self, state):
            raise ModelUnavailable("The remote GPU has no free time left today.")

    app = create_app(
        db_path=tmp_path / "a.db", claims_dir=claims, graph=Down(),
        user_db_path=tmp_path / "u.db", user_claims_dir=tmp_path / "u", policies_dir=tmp_path,
    )
    with TestClient(app) as client:
        response = client.post("/claims/c1/analyze")
    assert response.status_code == 503
    assert "GPU" not in response.json()["detail"]   # no internals in what the browser sees


def test_ui_client_turns_503_into_a_plain_message():
    client = ApiClient("http://api", transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    with pytest.raises(ServiceUnavailable, match="image-analysis model"):
        client.analyze("c1")


# ---------- image preparation ----------

def test_png_helper_shrinks_and_is_lossless(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (2000, 1000), (10, 200, 30)).save(path)
    import io
    out = Image.open(io.BytesIO(image_to_png_bytes(path, 512)))
    assert out.format == "PNG" and max(out.size) == 512 and out.getpixel((5, 5)) == (10, 200, 30)


# ---------- the Space folder ----------

def test_service_code_does_not_depend_on_the_claimguard_package():
    import ast

    root = Path(__file__).resolve().parents[1] / "service"
    for name in ("inference.py", "app.py"):
        imported = []
        for node in ast.walk(ast.parse((root / name).read_text(encoding="utf-8"))):   # real imports, not comments
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not [m for m in imported if m.split(".")[0] in ("claimguard", "configs")], name


def test_built_space_uses_the_training_prompt(tmp_path, monkeypatch):
    from claimguard.vlm.prompt import FT_PROMPT
    from scripts import build_space

    monkeypatch.setattr(build_space, "BUILD_DIR", tmp_path / "space")
    folder = build_space.build(bundle_adapter=False)
    assert (folder / "ft_prompt.txt").read_text(encoding="utf-8") == FT_PROMPT
    assert {p.name for p in folder.iterdir()} >= {"app.py", "inference.py", "requirements.txt", "README.md"}
