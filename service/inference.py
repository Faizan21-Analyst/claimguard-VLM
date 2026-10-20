"""Qwen3-VL-2B + the ClaimGuard LoRA adapter: one photo in, the model's raw text out.

Plain inference code. No web framework, no claimguard imports and no secrets in code, so the very same file runs
in a Hugging Face ZeroGPU Space (app.py) or behind Modal or any other host. It does the same steps as
src/claimguard/vlm/local_backend.py, so results can be compared with the local and Colab runs.

Settings come from environment variables (all optional):
    BASE_MODEL       default Qwen/Qwen3-VL-2B-Instruct
    ADAPTER_REPO     Hugging Face id of the LoRA adapter, e.g. username/claimguard-qwen3vl-lora
    ADAPTER_DIR      folder with the adapter inside this app (default "adapter"); used when ADAPTER_REPO is empty
    REQUIRE_ADAPTER  "1" (default): refuse to start without an adapter, so the base model can never be served
                     under the fine-tuned model's name. Set "0" only to test the base model on purpose.
    MAX_SIDE         longest photo side in pixels, must equal the training size (default 512)
    MAX_NEW_TOKENS   default 80
    DTYPE            float16 (default), bfloat16 or float32
    HF_TOKEN         only needed for a private adapter repo
"""
import os
from pathlib import Path

from PIL import Image, ImageOps

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "").strip()
ADAPTER_DIR = Path(os.environ.get("ADAPTER_DIR", "adapter"))
REQUIRE_ADAPTER = os.environ.get("REQUIRE_ADAPTER", "1") != "0"
MAX_SIDE = int(os.environ.get("MAX_SIDE", "512"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "80"))
DTYPE = os.environ.get("DTYPE", "float16")

# Written by scripts/build_space.py from claimguard.vlm.prompt.FT_PROMPT, so the text the model sees here is
# the text it was trained on and cannot drift from the repository.
PROMPT = (Path(__file__).parent / "ft_prompt.txt").read_text(encoding="utf-8")


def _adapter_path() -> str | None:
    if ADAPTER_REPO:
        from huggingface_hub import snapshot_download

        return snapshot_download(ADAPTER_REPO, token=os.environ.get("HF_TOKEN") or None)
    if (ADAPTER_DIR / "adapter_config.json").exists():
        return str(ADAPTER_DIR)
    return None


def load_model(device: str):
    """Load the base model and the adapter. Returns (model, processor)."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    adapter = _adapter_path()
    if adapter is None and REQUIRE_ADAPTER:
        raise RuntimeError(
            "No LoRA adapter found. Set ADAPTER_REPO to its Hugging Face id, or put it in the 'adapter' folder "
            "(it must contain adapter_config.json). Set REQUIRE_ADAPTER=0 to serve the plain base model."
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=getattr(torch, DTYPE))
    if adapter is not None:
        from peft import PeftModel

        # torch_device="cpu": on ZeroGPU, torch.cuda looks available at start-up (emulation), so PEFT would try to
        # read the adapter file straight onto a GPU that does not exist yet and fail with "No CUDA GPUs are available".
        model = PeftModel.from_pretrained(model, adapter, torch_device="cpu")
    model.to(device).eval()
    return model, AutoProcessor.from_pretrained(BASE_MODEL)


def generate(model, processor, image_path: str, device: str) -> str:
    """Raw model text for one photo. Parsing and validation happen in the ClaimGuard app, not here."""
    import torch

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))   # a no-op when the client already resized it

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
