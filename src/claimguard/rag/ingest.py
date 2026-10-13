
from functools import lru_cache
from pathlib import Path

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

from configs.settings import load_settings
from claimguard.rag.clauses import split_clauses


COLLECTION_NAME = "policies"
MIN_CHUNK_LENGTH = 40


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load each embedding model once per process."""
    return SentenceTransformer(model_name)


def get_chroma_collection(chroma_dir: Path | str):
    client = chromadb.PersistentClient(path=str(chroma_dir))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_policies(
    policies_dir: Path | str,
    chroma_dir: Path | str | None = None,
) -> int:
    """Ingest policy PDFs and return the number of clauses upserted."""
    settings = load_settings().rag

    policies_dir = Path(policies_dir)
    chroma_dir = Path(chroma_dir or settings.chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    collection = get_chroma_collection(chroma_dir)
    model = get_embedding_model(settings.embedding_model)

    ids = []
    documents = []
    metadatas = []

    for pdf_path in sorted(policies_dir.glob("*.pdf")):
        policy_id = pdf_path.stem

        with pymupdf.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                page_text = page.get_text("text")

                for clause_id, clause_text in split_clauses(page_text):
                    if len(clause_text) < MIN_CHUNK_LENGTH:
                        continue

                    # Include the page in the ID to avoid collisions
                    # if a clause number appears on multiple pages.
                    item_id = (
                        f"{policy_id}:{clause_id}:p{page_index + 1}"
                    )

                    ids.append(item_id)
                    documents.append(clause_text)
                    metadatas.append(
                        {
                            "policy_id": policy_id,
                            "clause_id": clause_id,
                            "page": page_index + 1,
                        }
                    )

    if not documents:
        return 0

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    return len(documents)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    count = ingest_policies(
        root / "data" / "synthetic" / "policies"
    )
    print(f"Upserted {count} policy clauses.")
