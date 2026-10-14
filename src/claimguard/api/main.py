import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import groq
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from configs.settings import load_settings
from claimguard.api import db
from claimguard.api.models import AssessmentRecord, ReviewRecord, ReviewRequest
from claimguard.schemas.claim import Claim, ClaimAssessment

logger = logging.getLogger(__name__)

SAFE_CLAIM_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def create_app(
    db_path: Path | str | None = None,
    claims_dir: Path | str | None = None,
    graph: Any = None,
) -> FastAPI:
    """Build the API. Dependencies are arguments so tests can pass a temp DB and a fake graph."""
    settings = load_settings()
    database_path = Path(db_path or settings.api.db_path)
    claims_root = Path(claims_dir or settings.api.claims_dir).resolve()
    graph_instance = graph

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db(database_path)
        yield

    app = FastAPI(title="ClaimGuard AI API", version="1.0.0", lifespan=lifespan)

    def find_claim_dir(claim_id: str) -> Path:
        if not SAFE_CLAIM_ID.fullmatch(claim_id):
            raise HTTPException(status_code=404, detail="Claim not found")

        candidate = (claims_root / claim_id).resolve()
        try:
            candidate.relative_to(claims_root)
        except ValueError:
            raise HTTPException(status_code=404, detail="Claim not found")

        if not (candidate / "claim.json").is_file():
            raise HTTPException(status_code=404, detail="Claim not found")

        return candidate

    def get_graph():
        nonlocal graph_instance
        if graph_instance is None:
            from claimguard.agent.graph import build_graph
            from claimguard.vlm.factory import get_vlm

            graph_instance = build_graph(get_vlm())
        return graph_instance

    def make_record(row: dict) -> AssessmentRecord:
        reviews = [
            ReviewRecord.model_validate(item)
            for item in db.get_reviews(database_path, row["id"])
        ]
        return AssessmentRecord(
            id=row["id"],
            created_at=row["created_at"],
            assessment=ClaimAssessment.model_validate_json(row["assessment_json"]),
            reviews=reviews,
        )

    def load_record(assessment_id: int) -> AssessmentRecord:
        row = db.get_assessment(database_path, assessment_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Assessment not found")
        return make_record(row)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/claims", response_model=list[str])
    def list_claims() -> list[str]:
        if not claims_root.is_dir():
            return []
        return sorted(
            folder.name
            for folder in claims_root.iterdir()
            if folder.is_dir()
            and SAFE_CLAIM_ID.fullmatch(folder.name)
            and (folder / "claim.json").is_file()
        )

    @app.post("/claims/{claim_id}/analyze", response_model=AssessmentRecord)
    def analyze(claim_id: str) -> AssessmentRecord:
        claim_dir = find_claim_dir(claim_id)

        try:
            claim = Claim.model_validate(
                json.loads((claim_dir / "claim.json").read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as exc:
            logger.exception("Invalid claim file for %s", claim_id)
            raise HTTPException(status_code=422, detail="The claim file is invalid.") from exc

        try:
            output = get_graph().invoke({"claim": claim, "claim_dir": claim_dir})

            # Supports both the real LangGraph output and a fake graph returning
            # a ClaimAssessment directly in API tests.
            if isinstance(output, ClaimAssessment):
                assessment = output
            elif isinstance(output, dict) and "assessment" in output:
                assessment = ClaimAssessment.model_validate(output["assessment"])
            else:
                raise TypeError("Graph returned an unexpected result.")

        except groq.APIError as exc:
            logger.exception("Groq API call failed for claim %s", claim_id)
            raise HTTPException(
                status_code=502,
                detail="The assessment service is temporarily unavailable.",
            ) from exc

        return load_record(db.save_assessment(database_path, assessment))

    @app.get("/claims/{claim_id}/assessments", response_model=list[AssessmentRecord])
    def claim_history(claim_id: str) -> list[AssessmentRecord]:
        find_claim_dir(claim_id)
        return [make_record(row) for row in db.list_assessments(database_path, claim_id)]

    @app.get("/assessments/{assessment_id}", response_model=AssessmentRecord)
    def read_assessment(assessment_id: int) -> AssessmentRecord:
        return load_record(assessment_id)

    @app.post("/assessments/{assessment_id}/review", response_model=AssessmentRecord)
    def create_review(assessment_id: int, request: ReviewRequest) -> AssessmentRecord:
        load_record(assessment_id)  # 404 if the assessment does not exist
        # Pydantic already rejected blank override notes.
        db.save_review(database_path, assessment_id, request.action, request.note)
        return load_record(assessment_id)

    return app


app = create_app()