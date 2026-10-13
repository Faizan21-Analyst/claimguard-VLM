"""Run the Groq VLM on one photo and show the raw text and the parsed result.

Usage (from repo root):  python -m scripts.check_vlm data/raw/roof.jpg
"""
import sys
from pathlib import Path

from claimguard.schemas.visual import VisualFinding
from claimguard.vlm.groq_backend import GroqVLM
from claimguard.vlm.parsing import extract_json


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m scripts.check_vlm <image_path>")

    image = Path(sys.argv[1])
    vlm = GroqVLM()
    print(f"Model: {vlm.model}\n")

    raw = vlm._raw(image)  # one call, so the raw text and parsed result match
    print("--- RAW MODEL TEXT ---")
    print(raw)

    print("\n--- PARSED ---")
    try:
        print(VisualFinding.model_validate(extract_json(raw)).model_dump_json(indent=2))
    except ValueError as exc:
        print(f"FAILED to parse: {exc}")


if __name__ == "__main__":
    main()