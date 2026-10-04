import math

import pytest

from app.evaluation.metrics import average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k

REC = [1, 2, 3, 4, 5]
REL = {2, 5, 9}


def test_precision_hand_computed():
    assert precision_at_k(REC, REL, 5) == pytest.approx(2 / 5)
    assert precision_at_k(REC, REL, 2) == pytest.approx(1 / 2)


def test_precision_penalises_short_lists():
    assert precision_at_k([2], {2}, 5) == pytest.approx(1 / 5)


def test_recall_hand_computed():
    assert recall_at_k(REC, REL, 5) == pytest.approx(2 / 3)
    assert recall_at_k(REC, REL, 1) == 0.0


def test_average_precision_hand_computed():
    # hits at rank 2 (P=1/2) and rank 5 (P=2/5); normaliser min(|rel|, k) = 3
    assert average_precision_at_k(REC, REL, 5) == pytest.approx((0.5 + 0.4) / 3)


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(REC, REL, 5) == pytest.approx(dcg / idcg)


def test_perfect_ranking_scores_one():
    for fn in (precision_at_k, average_precision_at_k, ndcg_at_k):
        assert fn([7, 8, 9], {7, 8, 9}, 3) == pytest.approx(1.0)
    assert recall_at_k([7, 8, 9], {7, 8, 9}, 3) == pytest.approx(1.0)


def test_no_hits_scores_zero():
    for fn in (precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k):
        assert fn([1, 2, 3], {9}, 3) == 0.0


def test_ndcg_rewards_earlier_hits():
    assert ndcg_at_k([9, 1, 2], {9}, 3) > ndcg_at_k([1, 2, 9], {9}, 3)


def test_empty_relevant_and_bad_k():
    assert recall_at_k(REC, set(), 5) == 0.0
    assert ndcg_at_k(REC, set(), 5) == 0.0
    with pytest.raises(ValueError):
        precision_at_k(REC, REL, 0)
