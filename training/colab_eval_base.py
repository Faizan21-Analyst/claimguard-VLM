"""Evaluate the BASE Qwen3-VL-2B-Instruct (no fine-tuning) on the golden set. Run on a Colab GPU.

Colab cells:
    !pip -q install -U "transformers>=4.57" accelerate scikit-learn pydantic pillow
    # upload colab_bundle.zip (left sidebar > Files), then:
    !unzip -q -o colab_bundle.zip -d /content/claimguard
    %cd /content/claimguard
    !PYTHONPATH=src python training/colab_eval_base.py

Result: results/base_golden.json (every prediction, for error analysis) + printed metrics.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimguard.schemas.visual import VisualFinding  # noqa: E402
from claimguard.vlm.prompt import FT_PROMPT  # noqa: E402

FIELDS = ["damage_type", "severity", "component", "image_quality"]
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def parse_prediction(text: str) -> dict | None:
    """Model text -> the 4 label fields, or None if it is not valid JSON for our schema."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        finding = VisualFinding(**json.loads(m.group(0)))  # also rejects labels outside the Literals
    except Exception:  # noqa: BLE001 - bad JSON or bad label both count as invalid
        return None
    return finding.model_dump(include=set(FIELDS))


def compute_metrics(rows: list[dict], preds: list[dict | None]) -> dict:
    from sklearn.metrics import f1_score

    out = {"n": len(rows), "valid_json_rate": sum(p is not None for p in preds) / len(rows)}
    for f in FIELDS:
        y_true = [r["target"][f] for r in rows]
        y_pred = [p[f] if p else "INVALID" for p in preds]  # invalid output = wrong answer
        out[f"{f}_acc"] = sum(a == b for a, b in zip(y_true, y_pred)) / len(rows)
        out[f"{f}_macro_f1"] = f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)
    for name, flag in (("real", False), ("synthetic_poor", True)):
        idx = [i for i, r in enumerate(rows) if r["synthetic_poor"] == flag]
        if idx:
            out[f"damage_type_acc_{name}"] = sum(
                preds[i] is not None and preds[i]["damage_type"] == rows[i]["target"]["damage_type"] for i in idx) / len(idx)
    return out


def load_model():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
    return model, AutoProcessor.from_pretrained(MODEL_ID)


def generate(model, processor, image_path: Path, max_side: int = 768) -> str:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_side, max_side))
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": FT_PROMPT}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=ROOT / "data" / "golden" / "golden.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "base_golden.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    model, processor = load_model()

    raw, preds, t0 = [], [], time.time()
    for i, r in enumerate(rows, 1):
        text = generate(model, processor, args.jsonl.parent / r["image"])
        raw.append(text)
        preds.append(parse_prediction(text))
        if i % 10 == 0:
            print(f"{i}/{len(rows)}  {time.time() - t0:.0f}s")

    metrics = compute_metrics(rows, preds)
    metrics["seconds_per_image"] = (time.time() - t0) / len(rows)
    print(json.dumps(metrics, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": MODEL_ID, "metrics": metrics, "rows": [
        {"id": r["id"], "target": r["target"], "raw": t, "pred": p} for r, t, p in zip(rows, raw, preds)]},
        indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
