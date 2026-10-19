"""Zip everything Colab needs into colab_bundle.zip (project root).

    python training/make_colab_bundle.py

Contains: src/claimguard (prompt + schema), training/*.py, data/processed (train/val), data/golden.
The golden set is included only so the eval can run; the training script must NEVER read it.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab_bundle.zip"
INCLUDE = ["src/claimguard", "training", "data/processed", "data/golden"]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def main() -> None:
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in INCLUDE:
            for p in sorted((ROOT / folder).rglob("*")):
                if p.is_file() and not SKIP_PARTS & set(p.parts) and p.suffix != ".pyc":
                    z.write(p, p.relative_to(ROOT).as_posix())
                    n += 1
    print(f"{n} files -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
