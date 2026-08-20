from dense.encode import QUERY_PREFIX, apply_query_prefix


def test_prefix_is_prepended_exactly_once():
    result = apply_query_prefix("search engine ranking")
    assert result == QUERY_PREFIX + "search engine ranking"


def test_prefix_matches_bge_instruction_convention():
    assert QUERY_PREFIX == "Represent this sentence for searching relevant passages: "
