
from claimguard.rag.clauses import split_clauses


def test_two_clauses_split_correctly():
    text = (
        "4.2 Hail and windstorm\n"
        "Hail damage to roofs is covered.\n"
        "6.1 Wear and tear\n"
        "Gradual deterioration is excluded."
    )

    clauses = split_clauses(text)

    assert [clause_id for clause_id, _ in clauses] == [
        "4.2",
        "6.1",
    ]
    assert clauses[0][1].startswith("Hail and windstorm")
    assert clauses[1][1].startswith("Wear and tear")


def test_text_without_clause_numbers_uses_zero():
    text = "This policy covers eligible accidental damage."

    assert split_clauses(text) == [("0", text)]


def test_clause_id_remains_a_string():
    clauses = split_clauses(
        "4.2 Hail and windstorm\n"
        "Hail damage to roofs is covered."
    )

    assert clauses[0][0] == "4.2"
    assert isinstance(clauses[0][0], str)
