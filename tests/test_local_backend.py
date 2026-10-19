"""Tests for the local VLM backend. torch / transformers / peft are faked, so no GPU or download is needed."""
import contextlib
import sys
import types
from types import SimpleNamespace

import pytest
from PIL import Image

from claimguard.vlm import local_backend
from claimguard.vlm.factory import get_vlm
from claimguard.vlm.local_backend import LocalQwenVLM


class _Ids:
    shape = (1, 3)


class _Inputs(dict):
    def to(self, device):
        return self


class _Out:
    def __getitem__(self, _):
        return self


def _install_fakes(monkeypatch, reply: str):
    """Fake torch/transformers/peft modules; returns a counter of model loads."""
    calls = {"loads": 0}

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return _Out()

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][1]["text"]            # FT_PROMPT is sent
            return _Inputs(input_ids=_Ids())

        def batch_decode(self, out, skip_special_tokens=True):
            return [reply]

    class FakeVL:
        @staticmethod
        def from_pretrained(name, dtype=None):
            calls["loads"] += 1
            return FakeModel()

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.float16, torch.bfloat16, torch.float32 = "f16", "bf16", "f32"
    torch.inference_mode = lambda: contextlib.nullcontext()
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda name: FakeProcessor())
    transformers.Qwen3VLForConditionalGeneration = FakeVL
    peft = types.ModuleType("peft")
    peft.PeftModel = SimpleNamespace(from_pretrained=lambda model, path: model)
    for name, mod in (("torch", torch), ("transformers", transformers), ("peft", peft)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


@pytest.fixture
def adapter_dir(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    return d


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "roof.png"
    Image.new("RGB", (80, 60), "gray").save(p)
    return p


@pytest.fixture(autouse=True)
def _settings(monkeypatch, adapter_dir):
    vlm = SimpleNamespace(
        local_base_model="fake/base", local_adapter_dir=adapter_dir,
        local_max_side=64, local_max_new_tokens=16, local_dtype="auto",
    )
    monkeypatch.setattr(local_backend, "load_settings", lambda: SimpleNamespace(vlm=vlm))
    monkeypatch.setattr(local_backend, "_CACHE", {})


def test_missing_adapter_fails_fast(monkeypatch, tmp_path):
    vlm = SimpleNamespace(local_base_model="b", local_adapter_dir=tmp_path / "nope",
                          local_max_side=64, local_max_new_tokens=16, local_dtype="auto")
    monkeypatch.setattr(local_backend, "load_settings", lambda: SimpleNamespace(vlm=vlm))
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        LocalQwenVLM()
    LocalQwenVLM(use_adapter=False)   # base model needs no adapter folder


def test_bad_dtype_rejected(monkeypatch, adapter_dir):
    vlm = SimpleNamespace(local_base_model="b", local_adapter_dir=adapter_dir,
                          local_max_side=64, local_max_new_tokens=16, local_dtype="int4")
    monkeypatch.setattr(local_backend, "load_settings", lambda: SimpleNamespace(vlm=vlm))
    with pytest.raises(ValueError, match="VLM_LOCAL_DTYPE"):
        LocalQwenVLM()


def test_factory_names(adapter_dir):
    assert get_vlm("local_ft").name == "local_ft:adapter"
    assert get_vlm("local_base").name == "local_base:fake/base"
    with pytest.raises(ValueError, match="local_ft"):
        get_vlm("nope")


def test_model_loads_once_and_json_parses(monkeypatch, photo):
    calls = _install_fakes(monkeypatch, '```json\n{"damage_type":"hail","severity":"moderate",'
                                        '"component":"roof_shingles","image_quality":"good"}\n```')
    vlm = LocalQwenVLM()
    finding1, ok1 = vlm.analyze(photo)
    finding2, ok2 = vlm.analyze(photo)
    assert ok1 and ok2 and finding1.damage_type == "hail" and finding2.severity == "moderate"
    assert finding1.findings == [] and finding1.confidence == 0.5   # defaults: model does not output them
    assert calls["loads"] == 1


def test_empty_output_falls_back_to_unclear(monkeypatch, photo):
    _install_fakes(monkeypatch, "   ")
    finding, ok = LocalQwenVLM().analyze(photo)
    assert ok is False and finding.damage_type == "unclear" and finding.image_quality == "poor"


def test_out_of_vocabulary_label_is_rejected(monkeypatch, photo):
    _install_fakes(monkeypatch, '{"damage_type":"tornado","severity":"minor","component":"roof_shingles"}')
    finding, ok = LocalQwenVLM().analyze(photo)
    assert ok is False and finding.damage_type == "unclear"
