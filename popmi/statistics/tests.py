"""Statistical test placeholders (implementations to be completed).
Provides function signatures for DeLong, McNemar, and bootstrap CI.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Callable

# DeLong test placeholder

def delong_ci(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.05) -> Tuple[float,float,float]:
    auc = float('nan')  # TODO implement
    return auc, float('nan'), float('nan')

def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    return float('nan')  # TODO

def bootstrap_ci(metric_fn: Callable[[np.ndarray,np.ndarray], float], y_true: np.ndarray, y_pred: np.ndarray,
                 n_boot: int = 1000, alpha: float = 0.05, rng: int = 42) -> Tuple[float,float,float]:
    rng_state = np.random.default_rng(rng)
    stats = []
    n = y_true.shape[0]
    for _ in range(n_boot):
        idx = rng_state.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    stats = np.array(stats)
    point = metric_fn(y_true, y_pred)
    lo = np.quantile(stats, alpha/2)
    hi = np.quantile(stats, 1-alpha/2)
    return point, lo, hi
