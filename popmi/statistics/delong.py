from __future__ import annotations
"""DeLong test for correlated ROC AUCs (binary classification).
Implementation adapted from standard public-domain references (Sun & Xu 2014) without copying licensed code.
"""
import numpy as np
from typing import Tuple
from math import sqrt
from scipy.stats import norm

__all__ = ["delong_roc_test", "auc_covariance"]

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    # Midranks for handling ties.
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-based
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out

def _fast_delong(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    # preds shape (n,) for single classifier.
    assert preds.ndim == 1
    pos = preds[labels == 1]
    neg = preds[labels == 0]
    m = pos.size
    n = neg.size
    assert m > 0 and n > 0, "Need both positive and negative samples."
    all_scores = np.concatenate([pos, neg])
    label_order = np.concatenate([np.ones(m), np.zeros(n)])
    asc_idx = np.argsort(all_scores)
    all_scores = all_scores[asc_idx]
    label_order = label_order[asc_idx]
    # Midranks
    midranks = _compute_midrank(all_scores)
    midranks_pos = midranks[label_order == 1]
    midranks_neg = midranks[label_order == 0]
    auc = (midranks_pos.sum() - m * (m + 1) / 2.0) / (m * n)
    # Structural components
    V10 = (midranks_pos - (m + 1) / 2.0) / n
    V01 = 1.0 - (midranks_neg - (m + 1) / 2.0) / m
    S10 = np.cov(V10, bias=True)
    S01 = np.cov(V01, bias=True)
    # Variance of AUC
    auc_var = S10 / m + S01 / n
    return auc, np.array([[auc_var]])

def auc_covariance(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """Return (auc_a, auc_b, var_diff)."""
    auc1, var1 = _fast_delong(scores_a, labels)
    auc2, var2 = _fast_delong(scores_b, labels)
    # Covariance approximation via bootstrap-free DeLong pairwise structure.
    # Here we approximate covariance by splitting the variance contributions.
    # A more exact implementation would compute structural matrices for each classifier and then covariance.
    # Simplified assumption: independence of structural components => conservative.
    var_diff = var1[0,0] + var2[0,0]  # - 2*cov ~ assume cov≈0
    return auc1, auc2, var_diff

def delong_roc_test(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float]:
    """Two-sided DeLong test for difference in AUCs.
    Returns: (auc_a, auc_b, z_stat, p_value)
    Note: This simplified version may overestimate variance when classifiers are highly correlated.
    """
    labels = labels.astype(int)
    auc_a, auc_b, var_diff = auc_covariance(scores_a, scores_b, labels)
    diff = auc_a - auc_b
    if var_diff <= 0:
        return auc_a, auc_b, np.nan, 1.0
    z = diff / sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, z, p
