import json
import logging
import re
import secrets
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, get_args

import groq
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from configs.settings import load_settings
from claimguard.agent.perils import PERIL_PATTERNS, WEATHER_PERILS
from claimguard.api import db, user_claims_db
from claimguard.api.models import AssessmentRecord, ReviewRecord, ReviewRequest
from claimguard.api.uploads import normalize_photo
from claimguard.schemas.claim import Claim, ClaimAssessment
from claimguard.schemas.visual import Component
from claimguard.vlm.base import ModelUnavailable

logger = logging.getLogger(__name__)

SAFE_CLAIM_ID = re.compile(r"^[A-Za-z0-9_-]+$")

# Only these file types can ever be listed or served. A test checks this matches the pipeline.
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

# "Try your own claim" accepts the perils the decision rules can actually reason about.
CLAIMABLE_PERILS = tuple(PERIL_PATTERNS)  # hail, wind, water, fire
COMPONENTS = get_args(Component)

# "Try your own claim" has no policy picker: the whole policy library is searched automatically.
AUTO_POLICY = "auto"


def create_app(
    db_path: Path | str | None = None,
    claims_dir: Path | str | None = None,
    graph: Any = None,
    user_db_path: Path | str | None = None,
    user_claims_dir: Path | str | None = None,
    policies_dir: Path | str | None = None,
    live_graph: Any = None,
) -> FastAPI:
    settings = load_settings()
    database_path = Path(db_path or settings.api.db_path)
    claims_root = Path(claims_dir or settings.api.claims_dir).resolve()
    graph_instance = graph

    # "Try your own claim" lives apart from the demo claims and the audit database.
    user_db = Path(user_db_path or settings.api.user_db_path or database_path.parent / "user_claims.db")
    user_root = Path(user_claims_dir or settings.api.user_claims_dir or database_path.parent / "user_claims").resolve()
    policies_root = Path(policies_dir or settings.api.policies_dir)
    max_photos = settings.api.max_photos
    max_photo_mb = settings.api.max_photo_mb
    live_graph_instance = live_graph

    def warm_up() -> None:
        """Load the heavy parts (torch, the embedding model, the search index) while the app is waking up,
        so the first claim after a cold start does not have to wait for them. An optimisation only."""
        try:
            from claimguard.agent.library import prepare_library
            from claimguard.rag.ingest import get_embedding_model

            r = settings.rag
            get_embedding_model(r.embedding_model)
            prepare_library(r.library_dir, r.library_chroma_dir)
            logging.getLogger("uvicorn.error").info("Warm-up finished")
        except Exception:  # noqa: BLE001 - the first claim simply loads these itself instead
            logger.warning("Warm-up failed", exc_info=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db(database_path)
        user_claims_db.init_user_db(user_db)
        if graph is None and live_graph is None:  # tests inject their own graph and need no warm-up
            threading.Thread(target=warm_up, daemon=True).start()
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

    def load_claim(claim_id: str) -> tuple[Claim, Path]:
        claim_dir = find_claim_dir(claim_id)
        try:
            claim = Claim.model_validate(
                json.loads((claim_dir / "claim.json").read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as exc:
            logger.exception("Invalid claim file for %s", claim_id)
            raise HTTPException(status_code=422, detail="The claim file is invalid.") from exc
        return claim, claim_dir

    def servable(name: str) -> bool:
        """The single rule for which file names may ever be listed or served."""
        return (
            Path(name).suffix.lower() in IMAGE_TYPES
            and not name.startswith(".")
            and ".." not in name
            and "/" not in name
            and "\\" not in name
        )

    def image_names(claim_dir: Path) -> list[str]:
        images_dir = claim_dir / "images"
        if not images_dir.is_dir():
            return []
        return sorted(
            (p.name for p in images_dir.iterdir() if p.is_file() and servable(p.name)),
            key=str.lower,
        )

    def find_image(claim_dir: Path, filename: str) -> tuple[Path, str]:
        """Resolve one photo, or raise 404. Never serves anything but an image inside images/."""
        not_found = HTTPException(status_code=404, detail="Image not found")

        if not servable(filename):
            raise not_found

        images_root = (claim_dir / "images").resolve()
        candidate = (images_root / filename).resolve()
        try:
            candidate.relative_to(images_root)
        except ValueError:
            raise not_found  # e.g. a symlink pointing outside images/

        if not candidate.is_file():
            raise not_found
        return candidate, IMAGE_TYPES[Path(filename).suffix.lower()]

    def get_graph():
        nonlocal graph_instance
        if graph_instance is None:
            from claimguard.agent.graph import build_graph
            from claimguard.vlm.factory import get_vlm
            from claimguard.vlm.notes import get_notes_writer

            graph_instance = build_graph(get_vlm(), notes=get_notes_writer())
        return graph_instance

    def make_record(row: dict) -> AssessmentRecord:
        reviews = [ReviewRecord.model_validate(item) for item in db.get_reviews(database_path, row["id"])]
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
            f.name for f in claims_root.iterdir()
            if f.is_dir() and SAFE_CLAIM_ID.fullmatch(f.name) and (f / "claim.json").is_file()
        )

    @app.get("/claims/{claim_id}", response_model=Claim)
    def read_claim(claim_id: str) -> Claim:
        claim, _ = load_claim(claim_id)
        return claim

    @app.get("/claims/{claim_id}/images", response_model=list[str])
    def list_images(claim_id: str) -> list[str]:
        return image_names(find_claim_dir(claim_id))

    @app.get("/claims/{claim_id}/images/{filename}")
    def read_image(claim_id: str, filename: str) -> FileResponse:
        path, media_type = find_image(find_claim_dir(claim_id), filename)
        return FileResponse(path, media_type=media_type, headers={"X-Content-Type-Options": "nosniff"})

    @app.post("/claims/{claim_id}/analyze", response_model=AssessmentRecord)
    def analyze(claim_id: str) -> AssessmentRecord:
        claim, claim_dir = load_claim(claim_id)

        try:
            output = get_graph().invoke({"claim": claim, "claim_dir": claim_dir})
            if isinstance(output, ClaimAssessment):
                assessment = output
            elif isinstance(output, dict) and "assessment" in output:
                assessment = ClaimAssessment.model_validate(output["assessment"])
            else:
                raise TypeError("Graph returned an unexpected result.")
        except ModelUnavailable as exc:
            logger.warning("Image model unavailable for claim %s: %s", claim_id, exc)
            raise HTTPException(status_code=503, detail="The image-analysis model is not available right now.") from exc
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
        load_record(assessment_id)  # 404 if it does not exist
        db.save_review(database_path, assessment_id, request.action, request.note)
        return load_record(assessment_id)

    # =====================================================================
    # "Try your own claim": uploaded photos + live weather. The demo endpoints above
    # are untouched; these use their own store and their own graph.
    # =====================================================================

    def get_live_graph():
        nonlocal live_graph_instance
        if live_graph_instance is None:
            from claimguard.agent.graph import build_graph
            from claimguard.agent.library import make_library_policy_tool
            from claimguard.agent.weather_live import make_live_weather_tool
            from claimguard.vlm.factory import get_vlm
            from claimguard.vlm.notes import get_notes_writer

            w = settings.weather
            r = settings.rag
            live_graph_instance = build_graph(
                get_vlm(),
                weather_tool=make_live_weather_tool(
                    wind_gust_kmh=w.wind_gust_kmh, hail_codes=w.hail_code_list(), timeout=w.timeout_s
                ),
                # No policy to pick: every document of the library is searched.
                policy_tool=make_library_policy_tool(r.library_dir, r.library_chroma_dir, r.library_top_k),
                rich_summary=True,
                notes=get_notes_writer(),
            )
        return live_graph_instance

    def rejected(message: str, status: int = 422) -> HTTPException:
        # Our own validation messages are written to be shown to the person who submitted the form.
        return HTTPException(status_code=status, detail=message)

    def find_user_claim(claim_id: str) -> tuple[Claim, Path]:
        not_found = HTTPException(status_code=404, detail="Claim not found")
        if not SAFE_CLAIM_ID.fullmatch(claim_id):
            raise not_found
        row = user_claims_db.get_user_claim(user_db, claim_id)
        if row is None:
            raise not_found
        claim = Claim(
            claim_id=row["id"],
            policy_id=row["policy_id"],
            claimed_damage_type=row["claimed_damage_type"],
            component=row["component"],
            incident_date=row["incident_date"],
            location=row["location"],
            description=row["description"],
        )
        return claim, user_root / claim_id

    @app.get("/policies", response_model=list[str])
    def list_policies() -> list[str]:
        return sorted(p.stem for p in policies_root.glob("*.pdf")) if policies_root.is_dir() else []

    @app.post("/user-claims", response_model=Claim, status_code=201)
    async def create_user_claim(
        policy_id: str = Form(AUTO_POLICY),
        claimed_damage_type: str = Form(...),
        component: str = Form("unclear"),
        incident_date: str = Form(""),
        location: str = Form(""),
        description: str = Form(""),
        photos: list[UploadFile] = File(default=[]),
    ) -> Claim:
        # The person does not choose a policy: "auto" means every document of the policy library is searched.
        # A specific demo policy name is still accepted (older clients), but the live search ignores it.
        if policy_id != AUTO_POLICY and policy_id not in list_policies():
            raise rejected("Unknown policy.")
        if claimed_damage_type not in CLAIMABLE_PERILS:
            raise rejected(f"The claimed damage must be one of: {', '.join(CLAIMABLE_PERILS)}.")
        if component not in COMPONENTS:
            raise rejected("Unknown part of the home.")

        location = " ".join(location.split())
        description = description.strip()
        if len(location) > 120 or len(description) > 1000:
            raise rejected("The location or the description is too long.")

        when = None
        if incident_date.strip():
            try:
                when = date.fromisoformat(incident_date.strip())
            except ValueError:
                raise rejected("The incident date is not valid.")
            if when > date.today() + timedelta(days=1):  # one day of slack for time zones
                raise rejected("The incident date cannot be in the future.")
        if claimed_damage_type in WEATHER_PERILS and (when is None or not location):
            raise rejected(
                "For hail and wind claims, please give the incident date and the location so the weather can be checked."
            )

        if not photos:
            raise rejected("Please add at least one photo.")
        if len(photos) > max_photos:
            raise rejected(f"Please upload at most {max_photos} photos.")

        # Validate and clean EVERY photo before anything is written to disk or the database.
        limit = max_photo_mb * 1024 * 1024
        clean: list[bytes] = []
        for number, upload in enumerate(photos, start=1):
            data = await upload.read(limit + 1)
            if len(data) > limit:
                raise rejected(f"Photo {number} is larger than {max_photo_mb} MB.", 413)
            try:
                clean.append(normalize_photo(data))
            except ValueError as exc:
                raise rejected(f"Photo {number}: {exc}") from exc

        claim_id = f"user_{secrets.token_hex(5)}"
        images_dir = user_root / claim_id / "images"
        try:
            images_dir.mkdir(parents=True)
            saved: list[tuple[str, int]] = []
            for number, jpeg in enumerate(clean, start=1):
                name = f"photo_{number}.jpg"
                (images_dir / name).write_bytes(jpeg)
                saved.append((name, len(jpeg)))
            user_claims_db.create_user_claim(
                user_db,
                claim_id,
                {
                    "policy_id": policy_id,
                    "claimed_damage_type": claimed_damage_type,
                    "component": component,
                    "incident_date": when.isoformat() if when else None,
                    "location": location or None,
                    "description": description,
                },
                saved,
            )
        except Exception:
            shutil.rmtree(user_root / claim_id, ignore_errors=True)  # never leave half a claim behind
            raise
        return find_user_claim(claim_id)[0]

    @app.get("/user-claims", response_model=list[str])
    def list_user_claims() -> list[str]:
        return user_claims_db.list_user_claim_ids(user_db)

    @app.get("/user-claims/{claim_id}", response_model=Claim)
    def read_user_claim(claim_id: str) -> Claim:
        return find_user_claim(claim_id)[0]

    @app.get("/user-claims/{claim_id}/images", response_model=list[str])
    def list_user_images(claim_id: str) -> list[str]:
        find_user_claim(claim_id)
        return user_claims_db.list_user_images(user_db, claim_id)

    @app.get("/user-claims/{claim_id}/images/{filename}")
    def read_user_image(claim_id: str, filename: str) -> FileResponse:
        _, claim_dir = find_user_claim(claim_id)
        path, media_type = find_image(claim_dir, filename)
        return FileResponse(path, media_type=media_type, headers={"X-Content-Type-Options": "nosniff"})

    @app.post("/user-claims/{claim_id}/analyze", response_model=AssessmentRecord)
    def analyze_user_claim(claim_id: str) -> AssessmentRecord:
        claim, claim_dir = find_user_claim(claim_id)

        try:
            output = get_live_graph().invoke({"claim": claim, "claim_dir": claim_dir})
            if isinstance(output, ClaimAssessment):
                assessment = output
            elif isinstance(output, dict) and "assessment" in output:
                assessment = ClaimAssessment.model_validate(output["assessment"])
            else:
                raise TypeError("Graph returned an unexpected result.")
        except ModelUnavailable as exc:
            logger.warning("Image model unavailable for user claim %s: %s", claim_id, exc)
            raise HTTPException(status_code=503, detail="The image-analysis model is not available right now.") from exc
        except groq.APIError as exc:
            logger.exception("Groq API call failed for user claim %s", claim_id)
            raise HTTPException(
                status_code=502,
                detail="The assessment service is temporarily unavailable.",
            ) from exc

        return load_record(db.save_assessment(database_path, assessment))

    @app.get("/user-claims/{claim_id}/assessments", response_model=list[AssessmentRecord])
    def user_claim_history(claim_id: str) -> list[AssessmentRecord]:
        find_user_claim(claim_id)
        return [make_record(row) for row in db.list_assessments(database_path, claim_id)]

    @app.delete("/user-claims/{claim_id}", status_code=204)
    def delete_user_claim(claim_id: str) -> Response:
        """Remove the uploaded photos and the form data. Audit assessments are append-only and stay."""
        find_user_claim(claim_id)
        user_claims_db.delete_user_claim(user_db, claim_id)
        shutil.rmtree(user_root / claim_id, ignore_errors=True)
        return Response(status_code=204)

    return app


app = create_app()