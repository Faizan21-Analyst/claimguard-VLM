"""Make an uploaded photo safe to store and to show again.

Every upload is decoded, checked and RE-ENCODED as a plain JPEG. That removes hidden data
(EXIF, which can contain the GPS position of a home), rejects files that merely pretend to be
images, and means we only ever store and serve one known format.
"""
import io

from PIL import Image, ImageOps

ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_PIXELS = 40_000_000     # about 6000 x 6600: refuses decompression bombs before decoding them
MAX_SIDE = 2048             # stored size; the VLM shrinks further (512 px for the local model)


def normalize_photo(data: bytes) -> bytes:
    """Return clean JPEG bytes, or raise ValueError with a message that is safe to show the user."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise ValueError("Only JPEG and PNG photos are accepted.")
            if img.width * img.height > MAX_PIXELS:
                raise ValueError("The photo has too many pixels. Please resize it first.")
            clean = ImageOps.exif_transpose(img).convert("RGB")   # applies phone rotation, drops metadata
    except ValueError:
        raise
    except (OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError("That file is not a readable photo.") from exc

    clean.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    clean.save(out, format="JPEG", quality=90)
    return out.getvalue()
