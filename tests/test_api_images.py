import io
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from claimguard.agent.graph import IMAGE_EXTENSIONS
from claimguard.api.main import IMAGE_TYPES, create_app
from claimguard.schemas.claim import ClaimAssessment

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CLAIMS = REPO_ROOT / "data" / "synthetic" / "claims"


def image_bytes(fmt: str, color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format=fmt)
    return buffer.getvalue()


PNG = image_bytes("PNG", "red")
JPG = image_bytes("JPEG", "blue")


class FakeGraph:
    def invoke(self, state):
        return ClaimAssessment(claim_id="claim_001", decision="supported", reasons=["x"])


@pytest.fixture
def env(tmp_path):
    """Demo claims copy where claim_001/images holds ONLY files this test controls."""
    claims_dir = tmp_path / "claims"
    shutil.copytree(DEMO_CLAIMS, claims_dir)

    claim = claims_dir / "claim_001"
    images = claim / "images"
    images.mkdir(exist_ok=True)
    for leftover in list(images.iterdir()):
        leftover.unlink()

    (images / "b.JPG").write_bytes(JPG)
    (images / "a.png").write_bytes(PNG)
    (images / "my photo.png").write_bytes(PNG)           # spaces are legal in photo names
    (images / "notes.txt").write_text("not an image", encoding="utf-8")
    (images / ".hidden.jpg").write_bytes(JPG)
    (images / "a..png").write_bytes(PNG)                  # a real file the ".." rule must still refuse
    (images / ".gitkeep").touch()
    (claim / "outside.jpg").write_bytes(JPG)              # a valid image, but NOT inside images/

    app = create_app(db_path=tmp_path / "audit.db", claims_dir=claims_dir, graph=FakeGraph())
    with TestClient(app) as client:
        yield client, claims_dir


# ---------- GET /claims/{id} ----------

def test_read_claim_returns_the_claim(env):
    client, _ = env
    response = client.get("/claims/claim_001")
    assert response.status_code == 200
    body = response.json()
    assert body["claim_id"] == "claim_001"
    assert body["claimed_damage_type"] == "hail"
    assert body["incident_date"] == "2026-06-10"


def test_read_unknown_claim_is_404(env):
    client, _ = env
    assert client.get("/claims/nope").status_code == 404


def test_read_broken_claim_is_422(env):
    client, claims_dir = env
    broken = claims_dir / "broken"
    broken.mkdir()
    (broken / "claim.json").write_text("{nope", encoding="utf-8")
    assert client.get("/claims/broken").status_code == 422


# ---------- GET /claims/{id}/images ----------

def test_image_list_is_sorted_and_only_contains_images(env):
    client, _ = env
    response = client.get("/claims/claim_001/images")
    assert response.status_code == 200
    assert response.json() == ["a.png", "b.JPG", "my photo.png"]


def test_image_list_for_claim_without_images_folder_is_empty(env):
    client, claims_dir = env
    shutil.rmtree(claims_dir / "claim_002" / "images", ignore_errors=True)
    assert client.get("/claims/claim_002/images").json() == []


def test_image_list_for_unknown_claim_is_404(env):
    client, _ = env
    assert client.get("/claims/nope/images").status_code == 404


# ---------- GET /claims/{id}/images/{filename} ----------

def test_png_is_served_with_same_bytes_and_type(env):
    client, _ = env
    response = client.get("/claims/claim_001/images/a.png")
    assert response.status_code == 200
    assert response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_jpg_with_uppercase_extension_is_served_as_jpeg(env):
    client, _ = env
    response = client.get("/claims/claim_001/images/b.JPG")
    assert response.status_code == 200
    assert response.content == JPG
    assert response.headers["content-type"] == "image/jpeg"


def test_file_name_with_spaces_works(env):
    client, _ = env
    response = client.get("/claims/claim_001/images/my%20photo.png")
    assert response.status_code == 200 and response.content == PNG


@pytest.mark.parametrize(
    "name",
    [
        "claim.json",        # wrong type
        "notes.txt",         # wrong type
        ".hidden.jpg",       # dotfile
        ".gitkeep",
        "missing.jpg",       # does not exist
        "outside.jpg",       # valid image, but lives next to claim.json, not in images/
        "..%2Foutside.jpg",  # traversal with an encoded slash
        "..%5Coutside.jpg",  # traversal with a Windows backslash
        "%2e%2e",            # just ".."
        "a..png",            # a real file, refused because ".." is anywhere in the name
    ],
)
def test_unsafe_or_unknown_image_names_are_404(env, name):
    client, _ = env
    assert client.get(f"/claims/claim_001/images/{name}").status_code == 404


def test_image_of_unknown_claim_is_404(env):
    client, _ = env
    assert client.get("/claims/nope/images/a.png").status_code == 404


def test_symlink_pointing_outside_images_is_not_served(env):
    client, claims_dir = env
    link = claims_dir / "claim_001" / "images" / "link.png"
    try:
        os.symlink(claims_dir / "claim_001" / "outside.jpg", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here (common on Windows)")
    assert client.get("/claims/claim_001/images/link.png").status_code == 404


# ---------- consistency with the pipeline ----------

def test_image_types_match_what_the_pipeline_analyzes():
    # If these drift apart, the reviewer would see photos the model never looked at.
    assert set(IMAGE_TYPES) == set(IMAGE_EXTENSIONS)