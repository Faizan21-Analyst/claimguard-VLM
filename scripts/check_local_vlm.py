"""Run the LOCAL Qwen3-VL model (fine-tuned by default) on one photo and show raw text, parsed result and timing.

Usage (from repo root):
    python -m scripts.check_local_vlm data/raw/images/hail/some_photo.jpg
    python -m scripts.check_local_vlm data/raw/images/hail/some_photo.jpg --base   # plain base model, no adapter
"""
import argparse
import time
from pathlib import Path

from claimguard.schemas.visual import VisualFinding
from claimguard.vlm.local_backend import LocalQwenVLM
from claimguard.vlm.parsing import extract_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--base", action="store_true", help="use the base model without the LoRA adapter")
    args = ap.parse_args()

    vlm = LocalQwenVLM(use_adapter=not args.base)
    print(f"Model: {vlm.name}\n")

    t0 = time.time()
    raw = vlm._raw(args.image)          # first call includes loading the model
    first = time.time() - t0
    t1 = time.time()
    vlm._raw(args.image)                # second call = real per-photo speed
    second = time.time() - t1

    print("--- RAW MODEL TEXT ---")
    print(raw)
    print("\n--- PARSED ---")
    try:
        print(VisualFinding.model_validate(extract_json(raw)).model_dump_json(indent=2))
    except ValueError as exc:
        print(f"FAILED to parse: {exc}")
    print(f"\nFirst call (incl. model load): {first:.1f}s | second call: {second:.1f}s")


if __name__ == "__main__":
    main()
