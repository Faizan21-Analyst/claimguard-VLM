import io

import pytest
from PIL import Image

from claimguard.api import uploads
from claimguard.api.uploads import normalize_photo


def image_bytes(fmt="JPEG", size=(40, 20), exif=None) -> bytes:
    buffer = io.BytesIO()
    kwargs = {"exif": exif.tobytes()} if exif is not None else {}
    Image.new("RGB", size, "red").save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def open_result(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def test_png_is_converted_to_jpeg():
    result = open_result(normalize_photo(image_bytes("PNG")))
    assert result.format == "JPEG"


def test_hidden_metadata_is_removed():
    exif = Image.Exif()
    exif[0x010F] = "SecretCamera"   # camera make; GPS data lives in the same block
    cleaned = open_result(normalize_photo(image_bytes("JPEG", exif=exif)))

    assert len(cleaned.getexif()) == 0


def test_phone_rotation_is_applied_before_the_tag_is_dropped():
    exif = Image.Exif()
    exif[0x0112] = 6   # "rotate 90 degrees": a 40x20 photo is really 20x40
    cleaned = open_result(normalize_photo(image_bytes("JPEG", size=(40, 20), exif=exif)))

    assert cleaned.size == (20, 40)


def test_large_photos_are_shrunk():
    cleaned = open_result(normalize_photo(image_bytes("JPEG", size=(3000, 1500))))
    assert max(cleaned.size) <= uploads.MAX_SIDE


def test_text_pretending_to_be_a_photo_is_refused():
    with pytest.raises(ValueError, match="not a readable photo"):
        normalize_photo(b"<html>not an image</html>")


def test_a_truncated_photo_is_refused():
    with pytest.raises(ValueError):
        normalize_photo(image_bytes("JPEG")[:30])


def test_other_image_formats_are_refused():
    with pytest.raises(ValueError, match="Only JPEG and PNG"):
        normalize_photo(image_bytes("GIF"))


def test_absurdly_large_pixel_counts_are_refused_before_decoding(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_PIXELS", 100)
    with pytest.raises(ValueError, match="too many pixels"):
        normalize_photo(image_bytes("JPEG", size=(50, 50)))
