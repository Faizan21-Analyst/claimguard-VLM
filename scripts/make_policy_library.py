"""Write the policy library PDFs ("Try your own claim") to data/policy_library/.

You normally do not need to run this: the API builds the library the first time a claim is analyzed and
re-indexes it whenever a PDF changes. Run it to regenerate the files or to look at them:

    python scripts/make_policy_library.py

To use your own policy documents instead, put real PDFs into data/policy_library/. The clauses must start
with a number such as "4.2 Hail" on their own line, like the generated ones.
"""
from pathlib import Path

from claimguard.rag.library_builder import build_library

LIBRARY_DIR = Path(__file__).resolve().parents[1] / "data" / "policy_library"


def main() -> None:
    for path in build_library(LIBRARY_DIR):
        print(f"Created {path}")


if __name__ == "__main__":
    main()
