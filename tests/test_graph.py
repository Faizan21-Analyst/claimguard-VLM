import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from claimguard.agent.graph import analyze_claim, build_graph
from claimguard.agent.tools import build_policy_query
from claimguard.rag.ingest import ingest_policies
from claimguard.rag.search import search_policy
from claimguard.schemas.claim import Claim, ClaimAssessment
from claimguard.schemas.visual import VisualFinding
from claimguard.vlm.base import VLMBackend

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
POLICIES_DIR = DATA / "policies"


class FakeVLM:
    name = "fake-vlm"

    def analyze(self, image_path):
        # The fake reads the claim that owns this image and "sees" the claimed damage.
        claim_path = Path(image_path).parent.parent / "claim.json"
        claim = Claim.model_validate_json(claim_path.read_text(encoding="utf-8"))

        return (
            VisualFinding(
                damage_type=claim.claimed_damage_type,
                severity="moderate",
                component=claim.component,
                findings=["Synthetic test finding"],
                image_quality="good",
                confidence=0.9,
            ),
            True,
        )


class GarbageVLM(VLMBackend):
    name = "garbage-vlm"

    def _raw(self, image_path: Path) -> str:
        return "This is not valid JSON."


class BrokenVLM:
    name = "broken-vlm"

    def analyze(self, image_path):
        raise RuntimeError("API down")


def exploding_summarizer(result, claim):
    raise RuntimeError("LLM down")


@pytest.fixture(scope="module")
def demo_environment(tmp_path_factory):
    """Copy the demo claims, give each one a small JPEG, and ingest the policies once."""
    root = tmp_path_factory.mktemp("claimguard_graph")

    claims_dir = root / "claims"
    shutil.copytree(DATA / "claims", claims_dir)

    for claim_dir in sorted(claims_dir.glob("claim_*")):
        images_dir = claim_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for leftover in list(images_dir.iterdir()):  # drop real photos copied from your machine
            leftover.unlink()
        Image.new("RGB", (64, 64), color=(160, 160, 160)).save(
            images_dir / "demo.JPG", format="JPEG"
        )

    chroma_dir = root / "chroma"
    assert ingest_policies(POLICIES_DIR, chroma_dir) > 0

    return {"root": root, "claims_dir": claims_dir, "chroma_dir": chroma_dir}


@pytest.fixture(scope="module")
def expected_decisions():
    return json.loads((DATA / "expected_decisions.json").read_text(encoding="utf-8"))


def make_graph(env, vlm=None, summarizer=None):
    return build_graph(vlm or FakeVLM(), chroma_dir=env["chroma_dir"], summarizer=summarizer)


def test_all_six_claims_match_expected_decisions(demo_environment, expected_decisions):
    env = demo_environment
    graph = make_graph(env)
    failures = []

    for claim_id, expected in sorted(expected_decisions.items()):
        claim_dir = env["claims_dir"] / claim_id
        assessment = analyze_claim(graph, claim_dir)

        if assessment.decision != expected:
            claim = Claim.model_validate_json((claim_dir / "claim.json").read_text(encoding="utf-8"))

            # Diagnostic only: threshold 0 shows every score so a failure is easy to understand.
            raw = search_policy(
                query=build_policy_query(claim),
                policy_id=claim.policy_id,
                chroma_dir=env["chroma_dir"],
                min_relevance=0.0,
            )
            scores = [(r.clause_id, round(r.relevance, 3), r.text[:100]) for r in raw]

            failures.append(
                f"{claim_id}: expected={expected}, actual={assessment.decision}, "
                f"reasons={assessment.reasons}, retrieval_scores={scores}"
            )

    assert not failures, "\n".join(failures)


def test_no_images_gives_insufficient_evidence(demo_environment, tmp_path):
    claim_dir = tmp_path / "claim_no_images"
    shutil.copytree(demo_environment["claims_dir"] / "claim_001", claim_dir)
    for image in (claim_dir / "images").iterdir():
        image.unlink()

    assessment = analyze_claim(make_graph(demo_environment), claim_dir)

    assert assessment.decision == "insufficient_evidence"
    assert assessment.visual == []


def test_garbage_vlm_gives_insufficient_evidence(demo_environment):
    assessment = analyze_claim(
        make_graph(demo_environment, vlm=GarbageVLM()),
        demo_environment["claims_dir"] / "claim_001",
    )

    assert assessment.decision == "insufficient_evidence"
    assert assessment.model_version == "garbage-vlm"
    assert assessment.visual
    assert assessment.visual[0].damage_type == "unclear"


def test_vision_outage_is_not_hidden(demo_environment):
    graph = make_graph(demo_environment, vlm=BrokenVLM())

    with pytest.raises(RuntimeError):
        analyze_claim(graph, demo_environment["claims_dir"] / "claim_001")


def test_summary_does_not_change_decision(demo_environment):
    graph = make_graph(demo_environment, summarizer=lambda result, claim: "x")

    assessment = analyze_claim(graph, demo_environment["claims_dir"] / "claim_001")

    assert assessment.summary == "x"
    assert assessment.decision == "supported"
    assert assessment.reasons
    assert assessment.model_version == "fake-vlm"


def test_summary_failure_does_not_crash_or_change_decision(demo_environment):
    graph = make_graph(demo_environment, summarizer=exploding_summarizer)

    assessment = analyze_claim(graph, demo_environment["claims_dir"] / "claim_001")

    assert assessment.summary == ""
    assert assessment.decision == "supported"



def test_non_image_files_in_images_folder_are_ignored(demo_environment, tmp_path):
    claim_dir = tmp_path / "claim_001"

    shutil.copytree(
        demo_environment["claims_dir"] / "claim_001",
        claim_dir,
    )

    image_dir = claim_dir / "images"

    # Add files that must be ignored by the vision pipeline.
    (image_dir / "notes.txt").write_text(
        "not an image",
        encoding="utf-8",
    )
    (image_dir / ".gitkeep").touch()

    # Count the actual supported image files in this claim.
    supported_extensions = {".jpg", ".jpeg", ".png"}
    valid_image_count = sum(
        1
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    assessment = analyze_claim(
        make_graph(demo_environment),
        claim_dir,
    )

    # Each valid image should produce a visual finding.
    # Non-image files must not produce additional findings.
    assert len(assessment.visual) == valid_image_count



def test_model_version_matches_vlm(demo_environment):
    vlm = FakeVLM()

    assessment = analyze_claim(
        make_graph(demo_environment, vlm=vlm),
        demo_environment["claims_dir"] / "claim_001",
    )

    assert assessment.model_version == vlm.name
    assert isinstance(assessment, ClaimAssessment)