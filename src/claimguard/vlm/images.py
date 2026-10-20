import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


def image_to_png_bytes(image_path: Path | str, max_side: int) -> bytes:
    """The photo prepared exactly as the local model prepares it, saved lossless (PNG).

    Same steps as LocalQwenVLM: fix EXIF rotation, convert to RGB, longest side <= max_side (the size the model
    was trained on). PNG, not JPEG, so a remote GPU receives exactly the pixels the local model would see and the
    two can be compared fairly.
    """
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_data_url(image_path: Path | str, max_side: int = 1024, quality: int = 90) -> str:
    """Shrink a photo so its longest side is <= max_side and return it as a JPEG data URL.

    Always re-encodes to JPEG, so the MIME type is always image/jpeg (even for PNG input).
    Fixes phone photos that are rotated through EXIF data. Never enlarges small images.
    """
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"