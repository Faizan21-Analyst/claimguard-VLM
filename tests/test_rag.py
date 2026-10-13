from pathlib import Path

import pytest

from claimguard.rag.ingest import ingest_policies
from claimguard.rag.search import search_policy

POLICIES_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "policies"
HAIL = "hail damage to roof shingles"


@pytest.fixture(scope="module")
def chroma_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chroma")
    ingest_policies(POLICIES_DIR, d)
    return d


def test_policy_a_hail_returns_42_first(chroma_dir):
    results = search_policy(HAIL, "policy_A", chroma_dir=chroma_dir)
    assert results and results[0].clause_id == "4.2"


def test_policy_b_hail_returns_41_first(chroma_dir):
    results = search_policy(HAIL, "policy_B", chroma_dir=chroma_dir)
    assert results and results[0].clause_id == "4.1"


def test_policy_c_hail_returns_nothing(chroma_dir):
    assert search_policy(HAIL, "policy_C", chroma_dir=chroma_dir) == []


def test_unrelated_query_returns_nothing(chroma_dir):
    assert search_policy("earthquake", "policy_A", chroma_dir=chroma_dir) == []


def test_keyword_filter_removes_wrong_peril(chroma_dir):
    # Threshold off, so only the keyword filter can remove the fire clauses.
    results = search_policy(HAIL, "policy_C", chroma_dir=chroma_dir,
                            min_relevance=0.0, must_mention=["hail"])
    assert results == []


def test_results_only_come_from_requested_policy(chroma_dir):
    results = search_policy(HAIL, "policy_A", chroma_dir=chroma_dir, min_relevance=0.0)
    assert len(results) > 1
    assert all(r.relevance <= 1 for r in results)