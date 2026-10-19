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


# A sentence ends at . ! ? followed by a space and a capital letter or an opening bracket.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")


def quote_for(text: str, peril: str, section_title: str | None = None, max_chars: int = 420) -> str:
    """The one sentence of a clause a reviewer should read for this peril: an exact piece of the clause.

    Preference: a sentence that names the peril AND grants or excludes coverage, then any sentence
    that names the peril, then the first sentence. The clause heading is skipped. Never invents words.
    """
    body = " ".join(text.split())          # collapse line breaks only; keep the original capitals
    heading = " ".join(section_title.split()) if section_title else ""
    if heading and body.lower().startswith(heading.lower()):
        body = body[len(heading):].lstrip(" :-")
    if not body:
        return ""

    sentences = [s.strip() for s in _SENTENCE_END.split(body) if s.strip()]
    about_peril = [s for s in sentences if clause_mentions_peril(s, peril)]
    decisive = [s for s in about_peril if says_excluded(s) or says_covered(s)]
    chosen = (decisive or about_peril or sentences)[0]

    if len(chosen) > max_chars:  # cut at a word boundary; the result is still an exact prefix of the sentence
        chosen = chosen[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + " ..."
    return chosen