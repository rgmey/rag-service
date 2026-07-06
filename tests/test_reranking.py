from app.rag.reranking import rerank


def test_rerank_returns_all_candidates_unchanged_if_fewer_than_top_k():
    candidates = ["a", "b"]
    assert rerank("q", candidates, top_k=5) == candidates


def test_rerank_returns_empty_list_for_no_candidates():
    assert rerank("q", [], top_k=3) == []


def test_rerank_falls_back_to_original_order_on_failure(mocker):
    mocker.patch("app.rag.reranking._score_passages", side_effect=RuntimeError("boom"))
    candidates = ["a", "b", "c", "d"]

    result = rerank("q", candidates, top_k=2)

    assert result == candidates[:2]


def test_rerank_falls_back_when_score_count_mismatches_candidates(mocker):
    # Model returned the wrong number of scores — don't trust a partial/misaligned result.
    mocker.patch("app.rag.reranking._score_passages", return_value=[1, 2])
    candidates = ["a", "b", "c", "d"]

    result = rerank("q", candidates, top_k=2)

    assert result == candidates[:2]


def test_rerank_sorts_by_score_descending(mocker):
    mocker.patch("app.rag.reranking._score_passages", return_value=[2, 9, 5, 1])
    candidates = ["low", "high", "mid", "lowest"]

    result = rerank("q", candidates, top_k=2)

    assert result == ["high", "mid"]
