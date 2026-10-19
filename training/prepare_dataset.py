"""Module 9b: validate labels.csv, dedupe, split (train / val / golden) and write the JSONL files.

    python training/prepare_dataset.py
    python training/prepare_dataset.py --golden-frac 0.12 --val-frac 0.12 --poor-frac 0.10

Outputs (all images are copied next to their JSONL, paths in the JSONL are relative to it):
    data/processed/train.jsonl, val.jsonl   + data/processed/images/...   (zip data/processed for Colab)
    data/golden/golden.jsonl                + data/golden/images/...      (never trained on)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import get_args

from PIL import Image, ImageEnhance, ImageFilter

from claimguard.schemas.visual import Component, DamageType, ImageQuality, Severity, VisualFinding

ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGES = ROOT / "data" / "raw" / "images"
LABELS = ROOT / "data" / "raw" / "labels.csv"
PROCESSED = ROOT / "data" / "processed"
GOLDEN = ROOT / "data" / "golden"
TARGET_KEYS = {"damage_type", "severity", "component", "image_quality"}  # what the model is trained to output
ALLOWED = {
    "damage_type": set(get_args(DamageType)), "severity": set(get_args(Severity)),
    "component": set(get_args(Component)), "image_quality": set(get_args(ImageQuality)),
}


def load_and_validate(path: Path) -> list[dict]:
    """Read labels.csv; report EVERY problem at once instead of stopping at the first."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    errors = []
    for n, r in enumerate(rows, start=2):  # line 1 is the header
        for col, allowed in ALLOWED.items():
            if r.get(col) not in allowed:
                errors.append(f"line {n}: {col}={r.get(col)!r} not in {sorted(allowed)}")
        img = RAW_IMAGES / r["image"]
        if not img.exists():
            errors.append(f"line {n}: file missing: {img}")
            continue
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001 - any decode problem means a bad file
            errors.append(f"line {n}: cannot open {img.name}: {e}")
    if errors:
        print(f"{len(errors)} problem(s) in {path}:")
        for e in errors[:50]:
            print("  ", e)
        if len(errors) > 50:
            print(f"   ... and {len(errors) - 50} more")
        sys.exit(1)
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """Drop byte-identical images: the same photo in train and golden would inflate every score."""
    seen, keep = {}, []
    for r in sorted(rows, key=lambda r: r["image"]):
        h = hashlib.sha256((RAW_IMAGES / r["image"]).read_bytes()).hexdigest()
        if h in seen:
            print(f"  duplicate: {r['image']} == {seen[h]} (dropped)")
            continue
        seen[h] = r["image"]
        r["sha256"] = h
        keep.append(r)
    print(f"dedupe: {len(rows)} -> {len(keep)} images")
    return keep


def split_groups(rows: list[dict], golden_frac: float, val_frac: float, seed: int) -> dict[str, list[dict]]:
    """Stratified by damage_type, and every `group` (one original photo) lands in exactly one split."""
    rng = random.Random(seed)
    by_type = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_type[r["damage_type"]][r["group"]].append(r)
    out = {"train": [], "val": [], "golden": []}
    for dtype, groups in sorted(by_type.items()):
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(keys)
        n_golden = max(1, round(n * golden_frac)) if n >= 5 else 0
        n_val = max(1, round(n * val_frac)) if n >= 5 else 0
        if n < 5:
            print(f"  WARNING: only {n} groups for damage_type={dtype}; all go to train (no val/golden)")
        for i, k in enumerate(keys):
            split = "golden" if i < n_golden else "val" if i < n_golden + n_val else "train"
            out[split].extend(groups[k])
    return out


def degrade(img: Image.Image, kind: str) -> Image.Image:
    """Make a photo that a human could not judge. Strong on purpose: mild blur would teach the wrong label."""
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=max(img.size) / 10))
    if kind == "dark":
        return ImageEnhance.Brightness(img).enhance(0.03)
    w, h = img.size  # "tiny": as if taken from far away / heavily compressed
    return img.resize((max(6, w // 40), max(6, h // 40))).resize((w, h))


def target_json(damage_type: str, severity: str, component: str, image_quality: str) -> dict:
    """Build the training target through VisualFinding so it can never drift from the schema."""
    finding = VisualFinding(damage_type=damage_type, severity=severity, component=component,
                            image_quality=image_quality)
    return finding.model_dump(include=TARGET_KEYS)


def write_split(name: str, rows: list[dict], out_dir: Path, jsonl: Path, poor_frac: float, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}-{name}")
    records = []
    for r in rows:
        rel = f"images/{r['image']}"
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAW_IMAGES / r["image"], dst)
        records.append({"id": Path(r["image"]).stem, "image": rel, "group": r["group"], "source": r["source"],
                        "synthetic_poor": False,
                        "target": target_json(r["damage_type"], r["severity"], r["component"], r["image_quality"])})
    # synthetic "poor" photos: same group (so no leakage), label = unreadable -> "unclear" + poor
    kinds = ["blur", "dark", "tiny"]
    for i, r in enumerate(rng.sample(rows, k=round(len(rows) * poor_frac))):
        kind = kinds[i % 3]
        rel = f"images/poor/{Path(r['image']).stem}__{kind}.jpg"
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(RAW_IMAGES / r["image"]) as im:
            degrade(im.convert("RGB"), kind).save(dst, quality=90)
        records.append({"id": f"{Path(r['image']).stem}__{kind}", "image": rel, "group": r["group"],
                        "source": r["source"], "synthetic_poor": True,
                        "target": target_json("unclear", "unclear", "unclear", "poor")})
    rng.shuffle(records)
    with jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def report(name: str, records: list[dict]) -> None:
    print(f"\n[{name}] {len(records)} rows ({sum(r['synthetic_poor'] for r in records)} synthetic poor)")
    for field in ("damage_type", "severity", "image_quality"):
        c = Counter(r["target"][field] for r in records)
        print(f"  {field:<13}", dict(sorted(c.items())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-frac", type=float, default=0.12)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--poor-frac", type=float, default=0.10, help="share of each split to add as synthetic poor photos")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = dedupe(load_and_validate(LABELS))
    splits = split_groups(rows, args.golden_frac, args.val_frac, args.seed)

    # start clean so no stale image from an earlier run (e.g. a former golden image) survives
    for d in (PROCESSED / "images", GOLDEN / "images"):
        shutil.rmtree(d, ignore_errors=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    GOLDEN.mkdir(parents=True, exist_ok=True)

    recs = {
        "train": write_split("train", splits["train"], PROCESSED, PROCESSED / "train.jsonl", args.poor_frac, args.seed),
        "val": write_split("val", splits["val"], PROCESSED, PROCESSED / "val.jsonl", args.poor_frac, args.seed),
        "golden": write_split("golden", splits["golden"], GOLDEN, GOLDEN / "golden.jsonl", args.poor_frac, args.seed),
    }
    for name, r in recs.items():
        report(name, r)

    # hard guarantees: no photo (group) and no file content appears in two splits
    groups = {n: {r["group"] for r in rs} for n, rs in recs.items()}
    assert not (groups["train"] & groups["golden"]) and not (groups["val"] & groups["golden"]) \
        and not (groups["train"] & groups["val"]), "LEAK: a photo group appears in two splits"
    print("\nOK: train / val / golden share no photo groups.")
    print(f"Next: zip {PROCESSED} for Colab. Keep {GOLDEN} out of training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
