"""Storage for claims submitted through "Try your own claim".

A separate SQLite file from the audit database (claimguard.db) and from the demo claims, so the
demos can never be touched by user data. Only form data and photo file names live here; the
photos themselves are files on disk. Assessments of these claims are still written to the normal
append-only audit database, so review and history work the same way.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claimguard.api.db import _connect  # same connection helper as the audit database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_user_db(path: Path | str) -> None:
    with _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_claims (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                claimed_damage_type TEXT NOT NULL,
                component TEXT NOT NULL,
                incident_date TEXT,
                location TEXT,
                description TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS user_claim_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT NOT NULL REFERENCES user_claims(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                UNIQUE (claim_id, filename)
            );
            """
        )


def create_user_claim(path: Path | str, claim_id: str, fields: dict[str, Any], images: list[tuple[str, int]]) -> None:
    """Insert the claim and its (filename, size) photo rows in one transaction."""
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO user_claims
                (id, created_at, policy_id, claimed_damage_type, component, incident_date, location, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim_id, _utc_now(), fields["policy_id"], fields["claimed_damage_type"], fields["component"],
                fields.get("incident_date"), fields.get("location"), fields.get("description", ""),
            ),
        )
        conn.executemany(
            "INSERT INTO user_claim_images (claim_id, filename, size_bytes) VALUES (?, ?, ?)",
            [(claim_id, name, size) for name, size in images],
        )


def list_user_claim_ids(path: Path | str) -> list[str]:
    with _connect(path) as conn:
        rows = conn.execute("SELECT id FROM user_claims ORDER BY created_at DESC, rowid DESC").fetchall()
        return [row["id"] for row in rows]


def get_user_claim(path: Path | str, claim_id: str) -> dict[str, Any] | None:
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM user_claims WHERE id = ?", (claim_id,)).fetchone()
        return dict(row) if row is not None else None


def list_user_images(path: Path | str, claim_id: str) -> list[str]:
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT filename FROM user_claim_images WHERE claim_id = ? ORDER BY id", (claim_id,)
        ).fetchall()
        return [row["filename"] for row in rows]


def delete_user_claim(path: Path | str, claim_id: str) -> bool:
    """Remove the claim and its photo rows. Audit assessments are append-only and stay."""
    with _connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.execute("DELETE FROM user_claims WHERE id = ?", (claim_id,))
        return cursor.rowcount > 0
