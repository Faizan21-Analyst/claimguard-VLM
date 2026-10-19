import re
from functools import lru_cache
from pathlib import Path

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

from configs.settings import load_settings
from claimguard.rag.clauses import split_clauses


COLLECTION_NAME = "policies"
MIN_CHUNK_LENGTH = 40

# Real policy documents have a running header and footer ("Form HS-100 ... Page 2 of 6").
# Text this close to the top or bottom edge of a page is never part of a clause.
PAGE_TOP_MARGIN = 60
PAGE_BOTTOM_MARGIN = 50

# A banner such as "SECTION 3  PERILS WE INSURE AGAINST" sits between clauses but belongs to none.
SECTION_BANNER = re.compile(r"(?m)^[ \t]*SECTION[ \t]+\d+\b.*$")

# "Hail damage Direct physical loss ..." = a heading run into its sentence (the old demo PDFs).
# A real, separate heading never has a capitalised word in the middle of its line.
RUN_ON_HEADING = re.compile(r"[a-z] [A-Z][a-z]")


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


def policy_title(pdf, fallback: str) -> str:
    """The document's own title: PDF metadata first, then the first line of page 1."""
    title = ((pdf.metadata or {}).get("title") or "").strip()
    if title:
        return title
    if len(pdf):
        for line in pdf[0].get_text("text").splitlines():
            if line.strip():
                return line.strip()
    return fallback


def section_title(clause_text: str) -> str:
    """The clause heading when it sits on its own line, else ''."""
    first = clause_text.strip().splitlines()[0].strip() if clause_text.strip() else ""
    if not first or len(first) > 60 or first.endswith((".", ":", ";")) or RUN_ON_HEADING.search(first):
        return ""
    return first


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
            name = policy_title(pdf, fallback=policy_id)

            for page_index, page in enumerate(pdf):
                body = pymupdf.Rect(
                    0, PAGE_TOP_MARGIN, page.rect.width, page.rect.height - PAGE_BOTTOM_MARGIN
                )
                page_text = SECTION_BANNER.sub("", page.get_text("text", clip=body))

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
                            "policy_name": name,
                            "section_title": section_title(clause_text),
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
