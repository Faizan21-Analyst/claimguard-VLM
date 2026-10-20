from typing import get_args

from claimguard.schemas.visual import Component, DamageType, ImageQuality, Severity


def _options(literal) -> str:
    """Allowed labels, read straight from the schema so the prompt can never drift from it."""
    return ", ".join(get_args(literal))


VLM_PROMPT = (
    "You are inspecting a photo submitted with a home insurance claim.\n"
    "Describe ONLY what is visible in the photo. Do not decide coverage, cost, "
    "payout, fault or percentages. If you are unsure, use \"unclear\".\n\n"
    "Return ONE JSON object and nothing else, with exactly these keys:\n"
    f"- damage_type: one of {_options(DamageType)} (use \"none\" if no damage is visible)\n"
    f"- severity: one of {_options(Severity)}\n"
    f"- component: one of {_options(Component)}\n"
    "- findings: a list of up to 6 short phrases describing visible evidence\n"
    f"- image_quality: one of {_options(ImageQuality)} "
    "(use \"poor\" if the photo is blurry, dark, or too far away)\n"
    "- confidence: a number from 0 to 1\n"
)

# Short prompt for the fine-tuned local model. Used for BOTH training and inference, so the
# model sees exactly the same instruction in both. findings/confidence are not trained;
# VisualFinding fills them with defaults.
FT_PROMPT = (
    "You are inspecting a photo submitted with a home insurance claim.\n"
    "Describe ONLY what is visible in the photo. If you are unsure, use \"unclear\".\n\n"
    "Return ONE JSON object and nothing else, with exactly these keys:\n"
    f"- damage_type: one of {_options(DamageType)}\n"
    f"- severity: one of {_options(Severity)}\n"
    f"- component: one of {_options(Component)}\n"
    f"- image_quality: one of {_options(ImageQuality)}\n"
)

# For the optional notes writer (a second model next to the fine-tuned one). It is NOT asked for severity, part,
# quality or confidence: it only describes what it sees, plus one damage type used as a weak second opinion.
NOTES_PROMPT = (
    "You are looking at a photo submitted with a home insurance claim.\n"
    "Describe ONLY what is visible in the photo. Do not decide coverage, cost, payout or fault.\n\n"
    "Return ONE JSON object and nothing else, with exactly these keys:\n"
    f"- damage_type: one of {_options(DamageType)}\n"
    "- findings: a list of up to 6 short phrases (under 12 words each) describing visible evidence\n"
)