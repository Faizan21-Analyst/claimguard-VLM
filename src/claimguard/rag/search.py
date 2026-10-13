
from pathlib import Path
from collections.abc import Sequence

from configs.settings import load_settings
from claimguard.rag.ingest import (
    get_chroma_collection,
    get_embedding_model,
)
from claimguard.schemas.evidence import PolicyEvidence


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

        evidence.append(
            PolicyEvidence(
                clause_id=metadata["clause_id"],
                page=metadata["page"],
                text=document,
                relevance=max(0.0, min(1.0, relevance)),
            )
        )

    return evidence
