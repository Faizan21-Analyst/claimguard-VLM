import json
import logging
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from claimguard.agent.explain import summarize
from claimguard.agent.fusion import FusionResult, fuse
from claimguard.agent.tools import policy_search_tool, weather_check_tool
from claimguard.schemas.claim import Claim, ClaimAssessment
from claimguard.schemas.evidence import PolicyEvidence, WeatherEvidence
from claimguard.schemas.visual import VisualFinding

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ClaimState(TypedDict, total=False):
    claim: Claim
    claim_dir: Path
    findings: list[VisualFinding]
    policy: list[PolicyEvidence]
    weather: WeatherEvidence | None
    result: FusionResult
    assessment: ClaimAssessment


def build_graph(
    vlm,
    chroma_dir: Path | str | None = None,
    summarizer=summarize,
):
    """Build the straight-line pipeline: vision -> policy -> weather -> decide -> explain.

    `vlm` is passed in so tests can use a fake and the fine-tuned model can plug in later.
    Pass summarizer=None to skip the LLM summary.
    """

    def vision_node(state: ClaimState) -> dict:
        image_dir = state["claim_dir"] / "images"
        if not image_dir.is_dir():
            return {"findings": []}

        images = sorted(
            (
                p
                for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda p: p.name.lower(),
        )

        # analyze() already turns bad MODEL output into a safe "unclear" finding.
        # Real failures (API down, bad key) are deliberately NOT caught: a loud error
        # is better than a claim wrongly labelled "photos unclear".
        return {"findings": [vlm.analyze(p)[0] for p in images]}

    def policy_node(state: ClaimState) -> dict:
        return {"policy": policy_search_tool(state["claim"], chroma_dir=chroma_dir)}

    def weather_node(state: ClaimState) -> dict:
        return {"weather": weather_check_tool(state["claim"], state["claim_dir"])}

    def decide_node(state: ClaimState) -> dict:
        result = fuse(
            claim=state["claim"],
            findings=state.get("findings", []),
            policy=state.get("policy", []),
            weather=state.get("weather"),
        )
        return {"result": result}

    def explain_node(state: ClaimState) -> dict:
        claim = state["claim"]
        result = state["result"]

        summary = ""
        if summarizer is not None:
            try:
                summary = summarizer(result, claim) or ""
            except Exception:
                # The summary is cosmetic. It must never block a decision.
                logger.exception("Claim summary failed")

        # Decision and audit reasons come straight from fuse(); the LLM never edits them.
        assessment = ClaimAssessment(
            claim_id=claim.claim_id,
            decision=result.decision,
            reasons=list(result.reasons),
            visual=list(state.get("findings", [])),
            policy_citations=list(state.get("policy", [])),
            weather=state.get("weather"),
            missing_evidence=list(result.missing_evidence),
            summary=summary,
            model_version=vlm.name,
        )
        return {"assessment": assessment}

    builder = StateGraph(ClaimState)
    builder.add_node("vision", vision_node)
    builder.add_node("policy", policy_node)
    builder.add_node("weather", weather_node)
    builder.add_node("decide", decide_node)
    builder.add_node("explain", explain_node)

    builder.add_edge(START, "vision")
    builder.add_edge("vision", "policy")
    builder.add_edge("policy", "weather")
    builder.add_edge("weather", "decide")
    builder.add_edge("decide", "explain")
    builder.add_edge("explain", END)

    return builder.compile()


def analyze_claim(graph, claim_dir: Path | str) -> ClaimAssessment:
    """Load, validate and assess one claim folder."""
    claim_dir = Path(claim_dir)
    claim_path = claim_dir / "claim.json"

    if not claim_path.is_file():
        raise FileNotFoundError(f"Claim file not found: {claim_path}")

    claim = Claim.model_validate(json.loads(claim_path.read_text(encoding="utf-8")))
    state = graph.invoke({"claim": claim, "claim_dir": claim_dir})
    return state["assessment"]