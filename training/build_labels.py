"""Module 9a: turn the downloaded Roboflow exports into data/raw/labels.csv and copy the images.

    python training/build_labels.py --downloads "C:/Users/ligio/OneDrive/Desktop/roof_downloads"
    python training/build_labels.py --downloads "..." --dry-run     # labels only, no image copy

Every rule below is deliberate. Read LABELING.md next to labels.csv for the reasoning.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMAGES = RAW / "images"
LABELS = RAW / "labels.csv"
COLUMNS = ["image", "damage_type", "severity", "component", "image_quality", "source", "license", "group"]

# folder = name of the folder inside --downloads. NOTE: "keyan_roof_damage" actually holds the
# deldetect hail dataset (see its README.dataset.txt).
SOURCES = {
    "hail": dict(folder="keyan_roof_damage", license="CC BY 4.0",
                 url="https://universe.roboflow.com/deldetect/roof-shingle-hail-damage"),
    "wind": dict(folder="roof wind damage", license="CC BY 4.0",
                 url="https://universe.roboflow.com/deldetect/roof-wind-damage"),
    "house_damage": dict(folder="house siding damage", license="CC BY 4.0",
                         url="https://universe.roboflow.com/house-damages/house-damage-boxes"),
}
# "Roof damage detection" (keyan) is NOT used: its only damage class is a generic "damage", so
# hail vs wind cannot be told apart, and no-box images are not reliably undamaged.

# Hand-collected intact roofs (plain folder of photos, no annotations). Web images, license NOT verified:
# fine for a private portfolio project, but do not redistribute. Images stay out of git.
INTACT = dict(folder="intact_roof", license="unverified (web images, not redistributed)")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Heuristic severity for roof sets (weak labels, documented in LABELING.md).
HAIL_MINOR_MAX_BOXES, HAIL_MODERATE_MAX_BOXES = 2, 8          # hail boxes are tiny: count them
WIND_MINOR_MAX_COVER, WIND_MODERATE_MAX_COVER = 0.06, 0.16     # wind boxes are big: use area share

# house_damage: damage type comes from the filename prefix, severity from the dataset's own
# traffic-light box class (Green < Amber < Red).
HOUSE_PREFIX = {"FireDamage": "fire", "WaterDamage": "water", "NoDamage": "none"}
COLOR_SEVERITY = {"Green": "minor", "Amber": "moderate", "Red": "severe"}
COLOR_RANK = {"Green": 1, "Amber": 2, "Red": 3}


def stem_of(file_name: str) -> str:
    """Roboflow appends '.rf.<hash>' to every export; the part before it is the original photo."""
    return file_name.split(".rf.")[0]


def load_coco(folder: Path):
    """Yield (split_dir, image, boxes, category_names) for every split folder with annotations."""
    for split in sorted(p for p in folder.iterdir() if p.is_dir()):
        ann = split / "_annotations.coco.json"
        if not ann.exists():
            continue
        data = json.loads(ann.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data["categories"]}
        boxes = defaultdict(list)
        for a in data["annotations"]:
            boxes[a["image_id"]].append(a)
        for im in data["images"]:
            yield split, im, boxes[im["id"]], cats


def best_per_stem(folder: Path):
    """One image per original photo (the variant with the most boxes). Returns (rows, n_dupes)."""
    best, seen = {}, 0
    for split, im, boxes, cats in load_coco(folder):
        seen += 1
        stem = stem_of(im["file_name"])
        if stem not in best or len(boxes) > len(best[stem][2]):
            best[stem] = (split, im, boxes, cats)
    return [best[s] for s in sorted(best)], seen - len(best)


def roof_severity(key: str, im: dict, boxes: list) -> str:
    if key == "hail":
        n = len(boxes)
        return "minor" if n <= HAIL_MINOR_MAX_BOXES else "moderate" if n <= HAIL_MODERATE_MAX_BOXES else "severe"
    area = sum(b["bbox"][2] * b["bbox"][3] for b in boxes)
    cover = min(1.0, area / (im["width"] * im["height"]))
    return "minor" if cover < WIND_MINOR_MAX_COVER else "moderate" if cover < WIND_MODERATE_MAX_COVER else "severe"


def roof_rows(key: str, folder: Path, skipped: Counter):
    items, dupes = best_per_stem(folder)
    skipped[f"{key}: duplicate exports of the same photo"] += dupes
    for split, im, boxes, _cats in items:
        if not boxes:  # no box does NOT mean undamaged (one such file is named "...Wind-Damage...")
            skipped[f"{key}: no boxes (cannot trust as 'none')"] += 1
            continue
        yield split, im, dict(
            damage_type=key, severity=roof_severity(key, im, boxes),
            component="roof_shingles", group=f"{key}:{stem_of(im['file_name'])}",
        )


def house_rows(folder: Path, cap: int, seed: int, skipped: Counter):
    items, dupes = best_per_stem(folder)
    skipped["house_damage: duplicate exports of the same photo"] += dupes
    by_type = defaultdict(list)
    for split, im, boxes, cats in items:
        m = re.match(r"^([A-Za-z]+)_", im["file_name"])
        prefix = m.group(1) if m else ""
        if prefix not in HOUSE_PREFIX:
            skipped[f"house_damage: prefix '{prefix}' not in schema or ambiguous"] += 1
            continue
        damage = HOUSE_PREFIX[prefix]
        names = {cats[b["category_id"]] for b in boxes}
        if damage == "none":
            severity = "none"
        else:
            colors = [n for n in names if n in COLOR_RANK]
            if not colors:
                skipped["house_damage: no severity colour box"] += 1
                continue
            severity = COLOR_SEVERITY[max(colors, key=COLOR_RANK.get)]
        by_type[damage].append((split, im, dict(
            damage_type=damage, severity=severity,
            component="other", group=f"house_damage:{stem_of(im['file_name'])}",
        )))
    rng = random.Random(seed)
    for damage, rows in sorted(by_type.items()):
        rng.shuffle(rows)
        skipped[f"house_damage: over per-class cap ({damage})"] += max(0, len(rows) - cap)
        yield from rows[:cap]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", required=True, type=Path)
    ap.add_argument("--house-cap", type=int, default=60, help="max images per damage type from house_damage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    skipped: Counter = Counter()
    out, missing = [], []
    for key, cfg in SOURCES.items():
        folder = args.downloads / cfg["folder"]
        if not folder.is_dir():
            print(f"ERROR: folder not found: {folder}")
            return 1
        rows = house_rows(folder, args.house_cap, args.seed, skipped) if key == "house_damage" \
            else roof_rows(key, folder, skipped)
        for split, im, label in rows:
            src = split / im["file_name"]
            if not args.dry_run and not src.exists():
                missing.append(str(src))
                continue
            rel = f"{key}/{im['file_name']}"
            if not args.dry_run:
                dst = IMAGES / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            out.append({"image": rel, "image_quality": "good", "source": f"{key} ({cfg['url']})",
                        "license": cfg["license"], **label})

    intact = args.downloads / INTACT["folder"]
    if intact.is_dir():
        for src in sorted(p for p in intact.iterdir() if p.suffix.lower() in IMG_EXT):
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", src.name)  # e.g. 'images4).jpg' -> 'images4_.jpg'
            rel = f"intact_roof/{safe}"
            if not args.dry_run:
                dst = IMAGES / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            out.append({"image": rel, "damage_type": "none", "severity": "none", "component": "roof_shingles",
                        "image_quality": "good", "source": "intact_roof (hand-collected)",
                        "license": INTACT["license"], "group": f"intact_roof:{safe}"})
    else:
        print(f"note: no '{INTACT['folder']}' folder found, skipping intact roofs")

    if missing:
        print(f"WARNING: {len(missing)} images listed in annotations but missing on disk, e.g. {missing[:3]}")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        with LABELS.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(out)

    print(f"\n{'(dry run) ' if args.dry_run else ''}{len(out)} labeled images -> {LABELS}")
    print("\nby damage_type / severity:")
    for (d, s), n in sorted(Counter((r["damage_type"], r["severity"]) for r in out).items()):
        print(f"  {d:<8} {s:<9} {n}")
    print("\nskipped:")
    for reason, n in sorted(skipped.items()):
        if n:
            print(f"  {n:>4}  {reason}")
    print("\nNote: no 'poor' photos exist in the sources; prepare_dataset.py adds synthetic ones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
