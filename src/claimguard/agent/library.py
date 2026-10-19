"""Policy search for "Try your own claim": the whole policy library, no policy to pick.

A person submitting their own claim does not know which document says what. So instead of asking, the system
builds the library on first use (if the folder is empty), keeps its search index in step with the PDFs, and
looks through every document for the clauses about the claimed damage.

Safety rule: a clause that grants or excludes cover for the claimed damage is NEVER dropped by the result
limit. Only neutral clauses (deductibles, evidence rules...) are trimmed. An exclusion can therefore not be
lost just because the library grew.
"""
import hashlib
import threading
from pathlib import Path

import chromadb

from claimguard.agent.perils import PERIL_PATTERNS, says_covered, says_excluded
from claimguard.agent.tools import build_policy_query, with_quotes
from claimguard.rag.ingest import COLLECTION_NAME, ingest_policies
from claimguard.rag.library_builder import build_library
from claimguard.rag.search import search_library
from claimguard.schemas.claim import Claim
from claimguard.schemas.evidence import PolicyEvidence

FINGERPRINT_FILE = "library.fingerprint"

# The API runs requests in threads: only one of them may build or re-index at a time.
_prepare_lock = threading.Lock()


def _fingerprint(library_dir: Path) -> str:
    """Changes whenever a PDF is added, removed, renamed or edited."""
    digest = hashlib.sha256()
    for pdf in sorted(library_dir.glob("*.pdf")):
        stat = pdf.stat()
        digest.update(f"{pdf.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def prepare_library(library_dir: Path | str, chroma_dir: Path | str) -> None:
    """Make sure the library exists and its search index matches it. Cheap when nothing changed."""
    library_dir, chroma_dir = Path(library_dir), Path(chroma_dir)

    with _prepare_lock:
        if not any(library_dir.glob("*.pdf")):
            build_library(library_dir)

        chroma_dir.mkdir(parents=True, exist_ok=True)
        marker = chroma_dir / FINGERPRINT_FILE
        current = _fingerprint(library_dir)
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == current:
            return

        # First run, or a document changed: rebuild the index from scratch so that no clause of a
        # deleted or edited document can linger.
        try:
            chromadb.PersistentClient(path=str(chroma_dir)).delete_collection(COLLECTION_NAME)
        except Exception:  # nothing to delete on a first run (the exception type differs between versions)
            pass
        ingest_policies(library_dir, chroma_dir)
        marker.write_text(current, encoding="utf-8")


def make_library_policy_tool(
    library_dir: Path | str,
    library_chroma_dir: Path | str,
    top_k: int = 12,
):
    """The `policy_tool` for build_graph(): same inputs and outputs as policy_search_tool."""

    def find_in_library(claim: Claim, chroma_dir: Path | str | None = None) -> list[PolicyEvidence]:
        pattern = PERIL_PATTERNS.get(claim.claimed_damage_type)
        if pattern is None:  # e.g. "unclear": nothing to look for
            return []

        prepare_library(library_dir, library_chroma_dir)
        found = search_library(build_policy_query(claim), pattern, library_chroma_dir)

        decisive = [c for c in found if says_excluded(c.text) or says_covered(c.text)]
        neutral = [c for c in found if not (says_excluded(c.text) or says_covered(c.text))]
        kept = decisive + neutral[: max(0, top_k - len(decisive))]
        kept.sort(key=lambda c: -c.relevance)
        return with_quotes(claim, kept)

    return find_in_library
