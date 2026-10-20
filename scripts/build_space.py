"""Assemble the Hugging Face Space folder from service/ and (optionally) upload it.

    python -m scripts.build_space                      # writes build/space/
    python -m scripts.build_space --bundle-adapter     # also copies models/qwen3vl_ft_A into build/space/adapter/
    python -m scripts.build_space --upload username/claimguard-vision    # then uploads build/space/ to that Space

What it does:
  * copies the files from service/
  * writes ft_prompt.txt from claimguard.vlm.prompt.FT_PROMPT, so the Space uses the exact training prompt
  * with --bundle-adapter, copies the LoRA adapter so the Space needs no second repository
    (otherwise set the ADAPTER_REPO variable of the Space to the adapter's Hugging Face id)

Create the Space once in the browser first (SDK: Gradio, hardware: ZeroGPU). Uploading needs a write token in the
HF_TOKEN environment variable. Never put the token in a file in this repository.
"""
import argparse
import os
import shutil
from pathlib import Path

from claimguard.vlm.prompt import FT_PROMPT

ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "service"
BUILD_DIR = ROOT / "build" / "space"
ADAPTER_DIR = ROOT / "models" / "qwen3vl_ft_A"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")   # all a LoRA adapter needs at inference


def build(bundle_adapter: bool) -> Path:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    for name in ("app.py", "inference.py", "requirements.txt", "README.md"):
        shutil.copy2(SERVICE_DIR / name, BUILD_DIR / name)
    (BUILD_DIR / "ft_prompt.txt").write_text(FT_PROMPT, encoding="utf-8")

    if bundle_adapter:
        target = BUILD_DIR / "adapter"
        target.mkdir()
        for name in ADAPTER_FILES:
            source = ADAPTER_DIR / name
            if not source.is_file():
                raise FileNotFoundError(f"Adapter file missing: {source}")
            shutil.copy2(source, target / name)
    return BUILD_DIR


def upload(folder: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set the HF_TOKEN environment variable to a Hugging Face WRITE token first.")
    from huggingface_hub import HfApi

    HfApi(token=token).upload_folder(
        repo_id=repo_id, repo_type="space", folder_path=str(folder), commit_message="Update ClaimGuard vision service"
    )
    print(f"Uploaded to https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-adapter", action="store_true", help=f"copy {ADAPTER_DIR.name}/ into the Space")
    ap.add_argument("--upload", metavar="USER/SPACE", help="upload the built folder to this Space")
    args = ap.parse_args()

    folder = build(args.bundle_adapter)
    print(f"Built {folder}")
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(folder)}  ({path.stat().st_size / 1024:.0f} KB)")
    if args.upload:
        upload(folder, args.upload)


if __name__ == "__main__":
    main()
