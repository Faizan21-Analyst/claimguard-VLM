"""Word lists and small helpers the decision rules use. No AI in this file."""
import re

# One precise pattern per peril. Word boundaries matter: a plain "wind" check would
# wrongly match "window". Perils not listed here (none, unclear, wear_and_tear) can
# never match a clause, so the claim ends up as "insufficient evidence".
PERIL_PATTERNS: dict[str, re.Pattern[str]] = {
    "hail": re.compile(r"\bhail(?:storms?|stones?)?\b", re.IGNORECASE),
    "wind": re.compile(r"\bwind(?:storms?|s)?\b", re.IGNORECASE),
    "water": re.compile(r"\bwater\b", re.IGNORECASE),
    "fire": re.compile(r"\bfires?\b", re.IGNORECASE),
}

# Deliberately specific phrases. A single word like "excluded" is too risky:
# "is covered when the event is not otherwise excluded" is a COVERAGE clause.
EXCLUSION_PHRASES = (
    "is excluded",
    "are excluded",
    "not covered",
    "not be covered",
    "does not cover",
    "do not cover",
    "no coverage",
)
COVERAGE_PHRASES = (
    "is covered",
    "are covered",
    "we cover",
    "this policy covers",
)

# Only these perils can be confirmed by a weather record.
WEATHER_PERILS = frozenset({"hail", "wind"})


def _normalize(text: str) -> str:
    """Lowercase and collapse every run of whitespace (PDF text has line breaks mid-sentence)."""
    return " ".join(text.lower().split())


def clause_mentions_peril(text: str, peril: str) -> bool:
    pattern = PERIL_PATTERNS.get(peril)
    return bool(pattern and pattern.search(text))


def says_excluded(text: str) -> bool:
    norm = _normalize(text)
    return any(phrase in norm for phrase in EXCLUSION_PHRASES)


def says_covered(text: str) -> bool:
    norm = _normalize(text)
    return any(phrase in norm for phrase in COVERAGE_PHRASES)