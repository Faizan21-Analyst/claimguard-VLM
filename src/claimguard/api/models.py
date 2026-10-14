from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator,model_validator

from claimguard.schemas.claim import ClaimAssessment


class ReviewRequest(BaseModel):
    action: Literal["accept", "request_more", "override"]
    note: str = ""

    @field_validator("note")
    @classmethod
    def validate_note(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def require_override_note(self):
        if self.action == "override" and not self.note:
            raise ValueError("An override requires a non-empty note.")
        return self

class ReviewRecord(BaseModel):
    id: int
    action: Literal["accept", "request_more", "override"]
    note: str
    created_at: datetime


class AssessmentRecord(BaseModel):
    id: int
    created_at: datetime
    assessment: ClaimAssessment
    reviews: list[ReviewRecord] = []
