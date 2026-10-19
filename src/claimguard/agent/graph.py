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


def _summary_facts(state: ClaimState) -> list[str]:
    """Plain facts for the reasoning model (live claims only): what each photo showed + the weather numbers."""
    facts = []
    for number, f in enumerate(state.get("findings", []), start=1):
        if f.damage_type == "unclear" or f.image_quality == "poor":
            facts.append(f"Photo {number}: could not be judged (unclear or poor quality).")
        else:
            facts.append(
                f"Photo {number}: {f.damage_type.replace('_', ' ')} damage, severity {f.severity}, "
                f"part {f.component.replace('_', ' ')}."
            )
    weather = state.get("weather")
    if weather is not None and weather.details:
        verdict = "supports the claimed event" if weather.found else "does not confirm the claimed event"
        facts.append(f"Weather data {verdict}: {weather.details}")
    return facts


def build_graph(
    vlm,
    chroma_dir: Path | str | None = None,
    summarizer=summarize,
    weather_tool=weather_check_tool,
    rich_summary: bool = False,
    policy_tool=None,
):
    """Build the straight-line pipeline: vision -> policy -> weather -> decide -> explain.

    `vlm` is passed in so tests can use a fake and the fine-tuned model can plug in later.
    Pass summarizer=None to skip the LLM summary.

    The defaults are the demo-claim pipeline (weather from a JSON file, summary from the decision
    notes only). "Try your own claim" passes a live `weather_tool` and rich_summary=True, so the
    reasoning model also sees the photo findings and the weather numbers. The decision itself
    always comes from fuse().

    `policy_tool` is how clauses are found. None = the demo way (the claim's own policy). "Try your own
    claim" passes a tool that searches the whole policy library, because that person has no policy to pick.
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
        find = policy_tool or policy_search_tool  # looked up now, not at import time
        return {"policy": find(state["claim"], chroma_dir=chroma_dir)}

    def weather_node(state: ClaimState) -> dict:
        return {"weather": weather_tool(state["claim"], state["claim_dir"])}

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
                facts = _summary_facts(state) if rich_summary else []
                summary = (summarizer(result, claim, facts=facts) if facts else summarizer(result, claim)) or ""
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