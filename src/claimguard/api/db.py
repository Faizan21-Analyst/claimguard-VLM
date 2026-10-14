import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from claimguard.schemas.claim import ClaimAssessment


@contextmanager
def _connect(path: Path | str) -> Iterator[sqlite3.Connection]:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(path: Path | str) -> None:
    with _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                decision TEXT NOT NULL,
                assessment_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_assessments_claim_created
            ON assessments (claim_id, created_at DESC, id DESC);

            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id INTEGER NOT NULL,
                action TEXT NOT NULL
                    CHECK (action IN ('accept', 'request_more', 'override')),
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (assessment_id)
                    REFERENCES assessments(id)
                    ON DELETE RESTRICT
            );

            CREATE INDEX IF NOT EXISTS idx_reviews_assessment
            ON reviews (assessment_id, id);

            -- Audit rule enforced by the database itself: assessments are append-only.
            CREATE TRIGGER IF NOT EXISTS assessments_no_update
            BEFORE UPDATE ON assessments
            BEGIN
                SELECT RAISE(ABORT, 'assessments are append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS assessments_no_delete
            BEFORE DELETE ON assessments
            BEGIN
                SELECT RAISE(ABORT, 'assessments are append-only');
            END;
            """
        )


def save_assessment(path: Path | str, assessment: ClaimAssessment) -> int:
    with _connect(path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO assessments
                (claim_id, model_version, decision, assessment_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                assessment.claim_id,
                assessment.model_version,
                assessment.decision,
                assessment.model_dump_json(),
                _utc_now(),
            ),
        )
        return int(cursor.lastrowid)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def get_assessment(path: Path | str, assessment_id: int) -> dict[str, Any] | None:
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM assessments WHERE id = ?",
            (assessment_id,),
        ).fetchone()
        return _row_to_dict(row)


def list_assessments(path: Path | str, claim_id: str) -> list[dict[str, Any]]:
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM assessments
            WHERE claim_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (claim_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def save_review(
    path: Path | str,
    assessment_id: int,
    action: str,
    note: str = "",
) -> int:
    with _connect(path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO reviews (assessment_id, action, note, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (assessment_id, action, note, _utc_now()),
        )
        return int(cursor.lastrowid)


def get_reviews(path: Path | str, assessment_id: int) -> list[dict[str, Any]]:
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT id, assessment_id, action, note, created_at
            FROM reviews
            WHERE assessment_id = ?
            ORDER BY id ASC
            """,
            (assessment_id,),
        ).fetchall()
        return [dict(row) for row in rows]