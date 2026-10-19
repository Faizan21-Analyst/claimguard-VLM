import re
from pathlib import Path
from collections.abc import Sequence

from configs.settings import load_settings
from claimguard.rag.ingest import (
    get_chroma_collection,
    get_embedding_model,
)
from claimguard.schemas.evidence import PolicyEvidence


def _evidence(document: str, metadata: dict, relevance: float) -> PolicyEvidence:
    """Build one citation. Older databases lack the document/heading fields; those stay None."""
    return PolicyEvidence(
        clause_id=metadata["clause_id"],
        page=metadata["page"],
        text=document,
        relevance=max(0.0, min(1.0, relevance)),
        policy_id=metadata.get("policy_id") or None,
        policy_name=metadata.get("policy_name") or None,
        section_title=metadata.get("section_title") or None,
    )


def search_policy(
    query: str,
    policy_id: str,
    chroma_dir: Path | str | None = None,
    top_k: int | None = None,
    min_relevance: float | None = None,
    must_mention: Sequence[str] | None = None, ) -> list[PolicyEvidence]:
    """Retrieve relevant clauses for one policy; abstain if none qualify."""
    settings = load_settings().rag

    chroma_dir = Path(chroma_dir or settings.chroma_dir)
    top_k = settings.top_k if top_k is None else top_k
    min_relevance = (
        settings.min_relevance
        if min_relevance is None
        else min_relevance
    )

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not 0 <= min_relevance <= 1:
        raise ValueError("min_relevance must be between 0 and 1")

    collection = get_chroma_collection(chroma_dir)

    if collection.count() == 0:
        return []

    model = get_embedding_model(settings.embedding_model)
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where={"policy_id": policy_id},
        include=["documents", "metadatas", "distances"],
    )

    evidence = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
        
    terms = [t.lower() for t in (must_mention or [])]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        relevance = 1.0 - float(distance)

        # This is the abstention rule: never return weak matches.
        if relevance < min_relevance:
            continue

        if terms and not any(t in document.lower() for t in terms):
            continue

        evidence.append(_evidence(document, metadata, relevance))

    return evidence


def search_library(
    query: str,
    must_match: re.Pattern[str],
    chroma_dir: Path | str,
    max_candidates: int = 400,
) -> list[PolicyEvidence]:
    """Search EVERY document in a policy library at once ("Try your own claim").

    The person does not know which document answers their question, so no policy is chosen up front.
    Every clause that names the claimed peril (`must_match`) is a candidate, best match first.
    A plain keyword test decides what is a candidate, not a similarity cut-off: a long, realistic
    clause scores low against a short query even when it is exactly the right one, and a strict cut-off
    would hide it (and could hide an exclusion).
    """
    collection = get_chroma_collection(chroma_dir)
    total = collection.count()
    if total == 0:
        return []

    model = get_embedding_model(load_settings().rag.embedding_model)
    query_embedding = model.encode(query, normalize_embeddings=True, convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max_candidates, total),
        include=["documents", "metadatas", "distances"],
    )

    return [
        _evidence(document, metadata, 1.0 - float(distance))
        for document, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if must_match.search(document)
    ]
