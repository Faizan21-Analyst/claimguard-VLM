from pydantic import BaseModel, Field


class PolicyEvidence(BaseModel):
    clause_id: str
    page: int | None = None
    text: str
    relevance: float = Field(default=0.0, ge=0, le=1)
    # Where the clause comes from, so a reviewer can find it in the document.
    # All optional: assessments saved before these existed still load.
    policy_id: str | None = None        # file name without .pdf
    policy_name: str | None = None      # the document's title, e.g. "HomeShield Homeowners Policy (Form HS-100)"
    section_title: str | None = None    # the clause heading, e.g. "Hail"
    quote: str | None = None            # the exact sentence of the clause that matters for this claim


class WeatherEvidence(BaseModel):
    event: str | None = None
    date: str | None = None
    location: str | None = None
    found: bool = False
    source: str = "mock"
    details: str | None = None   # live weather only: the numbers behind the answer
