from knowledge_agent.retrieval_evaluation import metrics


def test_matching_document_without_supporting_clause_is_not_a_hit():
    result = metrics(["D03@2026-02-01#p001"], ["D03@2026-02-01#p002"])
    assert result["hit@6"] is False
    assert result["recall@6"] == 0


def test_one_side_of_conflict_is_not_full_recall():
    result = metrics(["D01#p001", "D02#p001"], ["D01#p001", "unrelated"])
    assert result["hit@1"] is True
    assert result["recall@6"] == 0.5
    assert result["all_required@6"] is False


def test_outside_candidate_does_not_become_a_positive_hit():
    result = metrics([], ["unrelated"])
    assert result == {"outside_returned_candidates": True}
