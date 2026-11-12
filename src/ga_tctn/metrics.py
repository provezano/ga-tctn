"""Weighted F1 as in scikit-learn's f1_score(average="weighted"), without the dependency."""

from __future__ import annotations

import numpy as np


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Per-class F1 averaged with weights equal to each class's support in ``y_true``.

    Classes absent from ``y_true`` get weight zero; 0/0 precision or recall counts as 0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    denom = support + predicted
    f1 = np.zeros(n_classes)
    np.divide(2.0 * tp, denom, out=f1, where=denom > 0)
    return float((f1 * support).sum() / support.sum())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    return float((y_true == np.asarray(y_pred)).mean()) if y_true.size else 0.0
