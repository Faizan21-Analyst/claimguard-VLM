"""Plain-English summary of a decision.

Cosmetic only: it never changes the decision, and a failure here returns "".
"""
import logging
import re

from groq import Groq

from configs.settings import load_settings
from claimguard.agent.fusion import FusionResult
from claimguard.schemas.claim import Claim

logger = logging.getLogger(__name__)

# Reasoning models spend tokens thinking before they answer, so leave plenty of room.
MAX_TOKENS = 1500
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _build_prompt(result: FusionResult, claim: Claim) -> str:
    notes = "\n".join(f"- {reason}" for reason in result.reasons)
    missing = ", ".join(result.missing_evidence) or "none"
    component = claim.component.replace("_", " ")
    return (
        "Rewrite these insurance claim review notes as 2 to 3 plain sentences for an adjuster.\n"
        "Use ONLY the facts below. Do not add numbers, amounts, payouts or new facts, "
        "and do not change the decision.\n\n"
        f"Claim: {claim.claimed_damage_type} damage to {component}\n"
        f"Decision: {result.decision}\n"
        f"Notes:\n{notes}\n"
        f"Missing evidence: {missing}\n"
    )


def _clean(text: str) -> str:
    """Remove <think> blocks. A reasoning block that never closed means the output is unusable."""
    if "<think>" in text and "</think>" not in text:
        return ""
    return _THINK_BLOCK.sub("", text).strip()


def summarize(result: FusionResult, claim: Claim, client=None) -> str:
    """Try the main model, then the fallback. Return "" if both fail."""
    settings = load_settings().groq
    client = client or Groq(api_key=settings.api_key.get_secret_value(), timeout=30.0)
    prompt = _build_prompt(result, claim)

    for model in (settings.model, settings.fallback_model):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            choice = response.choices[0]

            if choice.finish_reason == "length":
                logger.warning("Summary from %s was cut off; trying next model", model)
                continue

            text = _clean(choice.message.content or "")
            if text:
                return text
            logger.warning("Summary from %s was empty; trying next model", model)

        except Exception:
            # Deliberately broad: a summary is optional and must never block a decision.
            logger.warning("Summary request to %s failed", model, exc_info=True)

    return ""