import base64
import io
import json
from pathlib import Path
from typing import get_args

import pytest
from PIL import Image

from claimguard.schemas.visual import Component, DamageType, ImageQuality, Severity
from claimguard.vlm.base import SAFE_FALLBACK, VLMBackend
from claimguard.vlm.factory import get_vlm
from claimguard.vlm.images import image_to_data_url
from claimguard.vlm.parsing import extract_json
from claimguard.vlm.prompt import VLM_PROMPT

VALID = {
    "damage_type": "hail",
    "severity": "moderate",
    "component": "roof_shingles",
    "findings": ["circular impact marks"],
    "image_quality": "good",
    "confidence": 0.8,
}
VALID_TEXT = json.dumps(VALID)
IMG = Path("unused.jpg")  # FakeVLM never opens the file


class FakeVLM(VLMBackend):
    name = "fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def _raw(self, image_path):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


# ---------- extract_json ----------

def test_extract_plain_json():
    assert extract_json(VALID_TEXT) == VALID


def test_extract_handles_code_fence():
    assert extract_json(f"```json\n{VALID_TEXT}\n```") == VALID


def test_extract_handles_think_block():
    assert extract_json(f"<think>hmm {{not json}}</think>{VALID_TEXT}") == VALID


def test_extract_handles_text_around_json():
    assert extract_json(f"Sure! Here you go:\n{VALID_TEXT}\nHope that helps.") == VALID


def test_extract_raises_when_no_json():
    with pytest.raises(ValueError):
        extract_json("I cannot see any damage.")


def test_extract_raises_on_unclosed_think_block():
    with pytest.raises(ValueError):
        extract_json(f"<think>still thinking {VALID_TEXT}")


# ---------- analyze ----------

def test_valid_output_returns_finding_and_true():
    vlm = FakeVLM([VALID_TEXT])
    finding, ok = vlm.analyze(IMG)
    assert ok is True and finding.damage_type == "hail" and vlm.calls == 1


def test_garbage_then_valid_recovers_on_retry():
    vlm = FakeVLM(["not json at all", VALID_TEXT])
    finding, ok = vlm.analyze(IMG, retries=1)
    assert ok is True and finding.severity == "moderate" and vlm.calls == 2


def test_always_garbage_returns_fallback_and_false():
    vlm = FakeVLM(["nope", "still nope"])
    finding, ok = vlm.analyze(IMG, retries=1)
    assert ok is False and finding == SAFE_FALLBACK and vlm.calls == 2


def test_bad_label_returns_fallback_and_false():
    bad = json.dumps({**VALID, "damage_type": "meteor"})
    finding, ok = FakeVLM([bad, bad]).analyze(IMG, retries=1)
    assert ok is False and finding.damage_type == "unclear"


def test_retries_zero_makes_one_call():
    vlm = FakeVLM(["nope"])
    _, ok = vlm.analyze(IMG, retries=0)
    assert ok is False and vlm.calls == 1


def test_value_error_from_backend_is_retried():
    vlm = FakeVLM([ValueError("model output was cut off"), VALID_TEXT])
    _, ok = vlm.analyze(IMG, retries=1)
    assert ok is True


def test_real_errors_are_not_swallowed():
    with pytest.raises(RuntimeError):
        FakeVLM([RuntimeError("network down")]).analyze(IMG)


def test_fallback_is_a_fresh_copy():
    finding, _ = FakeVLM(["x", "x"]).analyze(IMG)
    finding.findings.append("mutated")
    assert SAFE_FALLBACK.findings == []


# ---------- prompt ----------

def test_prompt_contains_every_allowed_label():
    for literal in (DamageType, Severity, Component, ImageQuality):
        for label in get_args(literal):
            assert label in VLM_PROMPT


# ---------- images ----------

def _decode(data_url: str) -> Image.Image:
    header, payload = data_url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def test_large_image_is_shrunk_and_converted_to_jpeg(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "red").save(path)
    img = _decode(image_to_data_url(path, max_side=1024))
    assert max(img.size) <= 1024 and img.format == "JPEG"


def test_small_image_is_not_enlarged(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (200, 100), "blue").save(path)
    assert _decode(image_to_data_url(path, max_side=1024)).size == (200, 100)


# ---------- factory ----------

def test_unknown_backend_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown VLM backend"):
        get_vlm("does_not_exist")