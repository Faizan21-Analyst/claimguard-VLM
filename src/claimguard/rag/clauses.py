
import re


CLAUSE_START = re.compile(
    r"(?m)^[ \t]*(\d+\.\d+)[ \t]+(?=[A-Z])"
)


def split_clauses(text: str) -> list[tuple[str, str]]:
    """Split extracted policy text into numbered clauses."""
    text = text.strip()

    if not text:
        return []

    matches = list(CLAUSE_START.finditer(text))

    if not matches:
        return [("0", text)]

    clauses = []

    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        clause_id = match.group(1)
        clause_text = text[start:end].strip()

        if clause_text:
            clauses.append((clause_id, clause_text))

    return clauses
