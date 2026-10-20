import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from claimguard.agent.explain import summarize
from claimguard.agent.fusion import FusionResult, fuse
from claimguard.agent.tools import policy_search_tool, weather_check_tool
from claimguard.schemas.claim import Claim, ClaimAssessment
from claimguard.schemas.evidence import PolicyEvidence, WeatherEvidence
from claimguard.schemas.visual import VisualFinding
from claimguard.vlm.base import VLMBackend

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ClaimState(TypedDict, total=False):
    claim: Claim
    claim_dir: Path
    findings: list[VisualFinding]
    models: list[str]   # the backend that really answered each photo (differs from vlm.name after a fallback)
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


def _analyze_one(vlm, path: Path) -> tuple[VisualFinding, str]:
    """(finding, name of the model that really produced it). Plain fakes without analyze_traced still work."""
    if isinstance(vlm, VLMBackend):
        finding, _, name = vlm.analyze_traced(path)
        return finding, name
    return vlm.analyze(path)[0], vlm.name


def _analyze_with_notes(vlm, notes, path: Path) -> tuple[VisualFinding, str]:
    """Like _analyze_one, plus optional notes from a second model, fetched in parallel so the claim does not wait twice.

    Notes are only added when the labelling model wrote none itself (the fine-tuned model writes 4 labels and
    nothing else). If Groq itself answered (e.g. as a fallback), its own notes are kept and no second opinion is
    added. This does not depend on `confidence`, which has a default value. The notes writer never raises, and
    nothing here is read by fuse().
    """
    if notes is None:
        return _analyze_one(vlm, path)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(notes.describe, path)
        finding, name = _analyze_one(vlm, path)   # a ModelUnavailable here still propagates, as before
    phrases, damage = pending.result()

    # Groq answered the photo itself (e.g. as the fallback): its own notes stay, and a "second opinion" from the
    # same model would mean nothing.
    if name == notes.name or name.startswith("groq:"):
        return finding, name

    update: dict = {}
    if phrases and not finding.findings:      # the fine-tuned model writes no notes, so these fill the gap
        update.update(findings=phrases, notes_source=notes.name)
    if damage is not None:
        update["second_opinion_damage"] = damage
    return (finding.model_copy(update=update) if update else finding), name


def _model_label(models: list[str], default: str) -> str:
    """One string for the audit record: every distinct model that answered, in order."""
    unique = list(dict.fromkeys(models))
    return " + ".join(unique) if unique else default


def build_graph(
    vlm,
    chroma_dir: Path | str | None = None,
    summarizer=summarize,
    weather_tool=weather_check_tool,
    rich_summary: bool = False,
    policy_tool=None,
    notes=None,
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

    `notes` is an optional second model that writes the visible-evidence notes (see vlm/notes.py). None = off.
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
        results = [_analyze_with_notes(vlm, notes, p) for p in images]
        return {"findings": [f for f, _ in results], "models": [name for _, name in results]}

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
            model_version=_model_label(state.get("models", []), vlm.name),
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