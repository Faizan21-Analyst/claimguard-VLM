
from pathlib import Path

from claimguard.rag.ingest import ingest_policies
from claimguard.rag.search import search_policy


ROOT = Path(__file__).resolve().parents[1]

POLICIES_DIR = ROOT / "data" / "synthetic" / "policies"
CHROMA_DIR = ROOT / "data" / "processed" / "chroma_experiment"

QUERIES = [
    "hail damage to roof shingles",
    "wear and tear",
    "water damage ceiling",
    "earthquake",
]

POLICY_IDS = ["policy_A", "policy_B", "policy_C"]


def main():
    count = ingest_policies(POLICIES_DIR, CHROMA_DIR)
    print(f"Ingested {count} clauses\n")

    for policy_id in POLICY_IDS:
        for query in QUERIES:
            results = search_policy(
                query=query,
                policy_id=policy_id,
                chroma_dir=CHROMA_DIR,
                top_k=4,
                min_relevance=0.0,
            )

            print(f"\nPolicy: {policy_id} | Query: {query}")

            if not results:
                print("  No results above the experimental threshold")
                continue

            for item in results:
                print(
                    f"  Clause {item.clause_id} | "
                    f"Page {item.page} | "
                    f"Relevance {item.relevance:.3f}"
                )
                print(f"    {item.text[:160]}")


if __name__ == "__main__":
    main()
