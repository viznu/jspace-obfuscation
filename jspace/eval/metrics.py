"""Detection metrics, made directly comparable across monitors.

Convention in this literature (Baker 2025, Zolkowski 2025): calibrate every
monitor to a fixed FPR on benign traces, then report recall (TPR) at that FPR.
That is what makes CoT-only vs J-space-only vs hybrid recall comparable.

Pure-stdlib so it runs and is unit-tested with no numpy/torch.
"""
from __future__ import annotations

import math
from typing import Sequence


def threshold_at_fpr(benign_scores: Sequence[float], target_fpr: float) -> float:
    """Smallest threshold T such that FPR(benign) <= target_fpr.

    A benign trace counts as a false positive when its score >= T. We want the
    fraction of benign scores >= T to be at most `target_fpr`, so T is just above
    the (1 - target_fpr) quantile of the benign scores.
    """
    if not benign_scores:
        raise ValueError("need at least one benign score to calibrate")
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be in [0, 1]")

    s = sorted(benign_scores)
    n = len(s)
    # Number of benign traces we are allowed to flag.
    allowed = math.floor(target_fpr * n)
    if allowed <= 0:
        # No benign flag permitted: threshold just above the max benign score.
        return _nextafter_up(s[-1])
    # Flag the top `allowed` benign scores: threshold at the (n-allowed)-th score.
    T = s[n - allowed]
    # Ties: if scores equal T sit below the cut too, bump above them so FPR holds.
    if s.count(T) + (n - allowed) - s[: n - allowed].count(T) > allowed:
        # There are ties spanning the boundary; place T just above the value.
        return _nextafter_up(T) if T == s[-1] else T
    return T


def fpr_at_threshold(benign_scores: Sequence[float], threshold: float) -> float:
    if not benign_scores:
        return 0.0
    flagged = sum(1 for x in benign_scores if x >= threshold)
    return flagged / len(benign_scores)


def recall_at_threshold(positive_scores: Sequence[float], threshold: float) -> float:
    if not positive_scores:
        return 0.0
    caught = sum(1 for x in positive_scores if x >= threshold)
    return caught / len(positive_scores)


def recall_at_fpr(
    positive_scores: Sequence[float],
    benign_scores: Sequence[float],
    target_fpr: float = 0.01,
) -> dict:
    """Primary metric: recall on positives at a calibrated FPR<=target on benign.

    Returns the threshold, the achieved (empirical) FPR, and the recall.
    """
    T = threshold_at_fpr(benign_scores, target_fpr)
    return {
        "threshold": T,
        "target_fpr": target_fpr,
        "empirical_fpr": fpr_at_threshold(benign_scores, T),
        "recall": recall_at_threshold(positive_scores, T),
        "n_pos": len(positive_scores),
        "n_neg": len(benign_scores),
    }


def roc_points(
    positive_scores: Sequence[float], benign_scores: Sequence[float]
) -> list[tuple[float, float]]:
    """(FPR, TPR) points across all candidate thresholds, sorted by FPR."""
    thresholds = sorted(set(list(positive_scores) + list(benign_scores)))
    pts = [(0.0, 0.0)]
    for T in thresholds:
        pts.append(
            (fpr_at_threshold(benign_scores, T), recall_at_threshold(positive_scores, T))
        )
    pts.append((1.0, 1.0))
    pts = sorted(set(pts))
    return pts


def auroc(positive_scores: Sequence[float], benign_scores: Sequence[float]) -> float:
    """Area under ROC via the rank-sum (Mann-Whitney U) identity, ties = 0.5."""
    if not positive_scores or not benign_scores:
        return float("nan")
    wins = 0.0
    for p in positive_scores:
        for n in benign_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positive_scores) * len(benign_scores))


def recall_above_max_negative(positive_scores, negative_scores) -> float:
    """Fraction of positives strictly above the highest negative.

    This is what "FPR<=1%" actually means when there are only ~10 negatives, so
    report it honestly under its real name instead of implying fine calibration.
    """
    if not positive_scores:
        return 0.0
    if not negative_scores:
        return 1.0
    m = max(negative_scores)
    return sum(1 for x in positive_scores if x > m) / len(positive_scores)


def bootstrap_ci(metric_fn, positive_scores, negative_scores,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a paired-sample metric like AUROC.

    Resamples positives and negatives with replacement. Uses a seeded PRNG
    (stdlib `random`) so results are reproducible.
    """
    import random

    pos = list(positive_scores)
    neg = list(negative_scores)
    if not pos or not neg:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        rp = [pos[rng.randrange(len(pos))] for _ in pos]
        rn = [neg[rng.randrange(len(neg))] for _ in neg]
        stats.append(metric_fn(rp, rn))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


def bootstrap_delta_auroc(posA, negA, posB, negB, n_boot: int = 2000, seed: int = 0):
    """Paired CI for AUROC(arm A) - AUROC(arm B), arms A and B evaluated on the SAME
    items (posA[i]/posB[i] are item i under two lenses). Resample item indices ONCE
    per replicate and apply to both arms, so the CI reflects the paired delta (not
    the sum of two independent SEs). Returns (point, lo, hi).
    """
    import random

    if len(posA) != len(posB) or len(negA) != len(negB):
        raise ValueError("paired arms must be aligned item-for-item")
    if not posA or not negA:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    point = auroc(posA, negA) - auroc(posB, negB)
    npn, nnn, stats = len(posA), len(negA), []
    for _ in range(n_boot):
        ip = [rng.randrange(npn) for _ in range(npn)]
        inn = [rng.randrange(nnn) for _ in range(nnn)]
        d = (auroc([posA[k] for k in ip], [negA[k] for k in inn])
             - auroc([posB[k] for k in ip], [negB[k] for k in inn]))
        stats.append(d)
    stats.sort()
    return (point, stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1])


def permutation_delta_pvalue(posA, negA, posB, negB, n_perm: int = 2000, seed: int = 0):
    """Two-sided p for ΔAUROC(A-B) by permuting the lens label within each item's
    paired score (the null: the two lenses are exchangeable on every item)."""
    import random

    if not posA or not negA:
        return float("nan")
    rng = random.Random(seed)
    obs = abs(auroc(posA, negA) - auroc(posB, negB))
    hits = 0
    for _ in range(n_perm):
        pa, pb, na, nb = [], [], [], []
        for a, b in zip(posA, posB):
            if rng.random() < 0.5:
                pa.append(a); pb.append(b)
            else:
                pa.append(b); pb.append(a)
        for a, b in zip(negA, negB):
            if rng.random() < 0.5:
                na.append(a); nb.append(b)
            else:
                na.append(b); nb.append(a)
        if abs(auroc(pa, na) - auroc(pb, nb)) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def _nextafter_up(x: float) -> float:
    return math.nextafter(x, float("inf"))
