"""Evaluation protocol of the GA-TCTN paper (Sec. 4.2).

For each repetition: n labelled documents per class, 10% of all documents as a
validation set (unlabelled during propagation, used to score hyper-parameters),
and the rest as the unlabelled/test set. Ten repetitions; scores are averaged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .metrics import weighted_f1
from .tctn import TCTN, TCTNConfig, document_scores, initial_term_scores


@dataclass
class Split:
    labeled: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def feasible(y: np.ndarray, n_per_class: int) -> bool:
    """A labelled-set size is used only if every class has at least ``n_per_class`` documents.

    This matches the sizes reported in the paper (e.g. Re1, whose smallest class has 10
    documents, appears with 10 labelled documents per class).
    """
    return bool(np.bincount(y).min() >= n_per_class)


def make_splits(
    y: np.ndarray,
    n_per_class: int,
    n_splits: int = 10,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> list[Split]:
    if not feasible(y, n_per_class):
        raise ValueError(f"some class has fewer than {n_per_class} documents")
    n = len(y)
    n_val = int(round(val_fraction * n))
    splits = []
    for child in np.random.SeedSequence([seed, n_per_class]).spawn(n_splits):
        rng = np.random.default_rng(child)
        labeled = np.concatenate([
            rng.choice(np.flatnonzero(y == c), size=n_per_class, replace=False)
            for c in range(int(y.max()) + 1)
        ])
        rest = np.setdiff1d(np.arange(n), labeled)
        rest = rng.permutation(rest)
        splits.append(Split(labeled=np.sort(labeled), validation=np.sort(rest[:n_val]), test=np.sort(rest[n_val:])))
    return splits


@dataclass
class Score:
    validation: float
    test: float
    per_split_validation: list[float] = field(default_factory=list)
    per_split_test: list[float] = field(default_factory=list)


class Evaluator:
    """Scores a hyper-parameter triple (sf, k, alpha) over all repetitions, with memoisation.

    The term network does not depend on the labels, so S(sf, k) is shared by all
    repetitions, and the initial scores of all repetitions are propagated together.
    """

    def __init__(self, X: sp.spmatrix, y: np.ndarray, n_classes: int, splits: list[Split],
                 config: TCTNConfig | None = None, k_max: int = 60):
        self.model = TCTN(X, n_classes, config, k_max=k_max)
        self.y = y
        self.C = n_classes
        self.splits = splits
        self.Y = np.hstack([
            initial_term_scores(self.model.X[s.labeled], y[s.labeled], n_classes) for s in splits
        ])
        self.cache: dict[tuple[int, int, float], Score] = {}
        self.n_evaluations = 0

    @staticmethod
    def key(sf: int, k: int, alpha: float) -> tuple[int, int, float]:
        return int(sf), int(k), round(float(alpha), 12)

    def __call__(self, sf: int, k: int, alpha: float) -> Score:
        key = self.key(sf, k, alpha)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        self.n_evaluations += 1
        F = self.model.term_scores(self.Y, key[0], key[1], key[2])
        D = document_scores(self.model.X, F, self.model.config.doc_weighting)
        val, test = [], []
        for i, s in enumerate(self.splits):
            pred = D[:, i * self.C:(i + 1) * self.C].argmax(axis=1)
            val.append(weighted_f1(self.y[s.validation], pred[s.validation], self.C))
            test.append(weighted_f1(self.y[s.test], pred[s.test], self.C))
        score = Score(float(np.mean(val)), float(np.mean(test)), val, test)
        self.cache[key] = score
        return score
