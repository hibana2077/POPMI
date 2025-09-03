from __future__ import annotations
import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
import numpy as np

def multiclass_balanced_accuracy(preds: Tensor, targets: Tensor) -> float:
    p = preds.detach().cpu().numpy()
    t = targets.detach().cpu().numpy()
    return balanced_accuracy_score(t, p)

def multiclass_macro_f1(preds: Tensor, targets: Tensor) -> float:
    return f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='macro')

def binary_auroc(probs: Tensor, targets: Tensor) -> float:
    try:
        return roc_auc_score(targets.cpu().numpy(), probs.cpu().numpy())
    except ValueError:
        return float('nan')

def multilabel_auroc(probs: Tensor, targets: Tensor) -> float:
    try:
        return roc_auc_score(targets.cpu().numpy(), probs.cpu().numpy(), average='macro')
    except ValueError:
        return float('nan')
