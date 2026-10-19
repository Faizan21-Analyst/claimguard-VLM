import threading
from pathlib import Path

from PIL import Image, ImageOps

from configs.settings import load_settings
from claimguard.vlm.base import VLMBackend
from claimguard.vlm.prompt import FT_PROMPT

_VALID_DTYPES = {"auto", "float16", "bfloat16", "float32"}
_LOCK = threading.Lock()   # one generation at a time: the model is not thread-safe and RAM is limited
_CACHE: dict = {}          # (base, adapter, dtype) -> (model, processor, device); loaded once per process


class LocalQwenVLM(VLMBackend):
    """Qwen3-VL-2B running on this machine. With the LoRA adapter it is the fine-tuned model.

    use_adapter=False gives the plain base model (same prompt), handy for comparisons.
    torch / transformers / peft are imported lazily: nothing heavy loads until the first photo.
    """

    def __init__(self, use_adapter: bool = True):
        s = load_settings().vlm
        if s.local_dtype not in _VALID_DTYPES:
            raise ValueError(f"VLM_LOCAL_DTYPE must be one of {sorted(_VALID_DTYPES)}, got {s.local_dtype!r}")

        self.base_model = s.local_base_model
        self.adapter_dir = Path(s.local_adapter_dir) if use_adapter else None
        self.max_side = s.local_max_side              # must match the size used in training (512)
        self.max_new_tokens = s.local_max_new_tokens
        self.dtype_name = s.local_dtype

        if self.adapter_dir is not None and not (self.adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"LoRA adapter not found: expected {self.adapter_dir / 'adapter_config.json'}. "
                "Set VLM_LOCAL_ADAPTER_DIR or use backend 'local_base'."
            )
        self.name = f"local_ft:{self.adapter_dir.name}" if self.adapter_dir else f"local_base:{self.base_model}"

    def _load(self):
        key = (self.base_model, str(self.adapter_dir), self.dtype_name)
        if key not in _CACHE:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.dtype_name == "auto":
                dtype = torch.float16 if device == "cuda" else torch.bfloat16
            else:
                dtype = getattr(torch, self.dtype_name)

            model = Qwen3VLForConditionalGeneration.from_pretrained(self.base_model, dtype=dtype)
            if self.adapter_dir is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(self.adapter_dir))
            model.to(device).eval()
            _CACHE[key] = (model, AutoProcessor.from_pretrained(self.base_model), device)
        return _CACHE[key]

    def _raw(self, image_path: Path) -> str:
        import torch

        with _LOCK:
            model, processor, device = self._load()

            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((self.max_side, self.max_side))

            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": FT_PROMPT}],
            }]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(device)

            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        if not text.strip():
            raise ValueError("model returned an empty response")
        return text
