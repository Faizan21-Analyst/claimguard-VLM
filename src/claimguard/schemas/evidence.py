from pydantic import BaseModel, Field


class PolicyEvidence(BaseModel):
    clause_id: str
    page: int | None = None
    text: str
    relevance: float = Field(default=0.0, ge=0, le=1)


class WeatherEvidence(BaseModel):
    event: str | None = None
    date: str | None = None
    location: str | None = None
    found: bool = False
    source: str = "mock"