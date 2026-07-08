import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.eval import threshold_at_fpr, fpr_at_threshold, recall_at_fpr, auroc


def test_threshold_respects_fpr_cap():
    benign = [float(i) for i in range(100)]  # 0..99
    T = threshold_at_fpr(benign, 0.01)         # allow flagging 1 benign
    assert fpr_at_threshold(benign, T) <= 0.01


def test_zero_fpr_flags_nothing_benign():
    benign = [1.0, 2.0, 3.0]
    T = threshold_at_fpr(benign, 0.0)
    assert fpr_at_threshold(benign, T) == 0.0


def test_recall_at_fpr_separable():
    benign = [0.0] * 100
    pos = [1.0] * 20
    res = recall_at_fpr(pos, benign, 0.01)
    assert res["recall"] == 1.0
    assert res["empirical_fpr"] <= 0.01


def test_recall_at_fpr_overlap():
    benign = [float(i) for i in range(100)]
    pos = [50.0] * 10
    res = recall_at_fpr(pos, benign, 0.01)
    assert 0.0 <= res["recall"] <= 1.0


def test_auroc_perfect_and_chance():
    assert auroc([1, 1, 1], [0, 0, 0]) == 1.0
    assert auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
