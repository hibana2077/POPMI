from __future__ import annotations
"""McNemar test for paired classification outputs + Holm-Bonferroni correction utility."""
import numpy as np
from typing import Tuple, List
from math import comb
from scipy.stats import chi2

__all__ = ["mcnemar_test", "holm_bonferroni"]

def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray, labels: np.ndarray, exact: bool = False) -> Tuple[int,int,float,float]:
    """Return (b, c, stat, p_value) where:
    b = # samples A correct, B wrong
    c = # samples B correct, A wrong
    If exact: use binomial test (two-sided), else chi-square with continuity correction.
    """
    pred_a = pred_a.astype(int)
    pred_b = pred_b.astype(int)
    labels = labels.astype(int)
    a_correct = pred_a == labels
    b_correct = pred_b == labels
    b_ct = int(np.logical_and(a_correct, ~b_correct).sum())
    c_ct = int(np.logical_and(~a_correct, b_correct).sum())
    if b_ct + c_ct == 0:
        return b_ct, c_ct, 0.0, 1.0
    if exact:
        # Two-sided binomial: count extremes
        n = b_ct + c_ct
        k = min(b_ct, c_ct)
        from math import comb
        p = 0.0
        for i in range(0, k+1):
            p += comb(n, i)
        p *= 2 ** (-n)
        return b_ct, c_ct, np.nan, min(1.0, 2*p)
    # Chi-square with continuity correction
    stat = (abs(b_ct - c_ct) - 1) ** 2 / (b_ct + c_ct)
    p_value = 1 - chi2.cdf(stat, df=1)
    return b_ct, c_ct, stat, p_value

def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Return list of significance decisions after Holm-Bonferroni correction."""
    m = len(p_values)
    order = np.argsort(p_values)
    decisions = [False]*m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            decisions[idx] = True
        else:
            break
    return decisions
