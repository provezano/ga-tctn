"""Transductive Classification through Term Networks (TCTN).

Steps (Rossi, Rezende & Lopes 2015; Sec. 2.3 of the GA-TCTN paper):
1. term network from document co-occurrence (see ``network``),
2. initial class scores of terms from the labelled documents (Eq. 1),
3. LLGC-style propagation F <- alpha S F + (1 - alpha) Y (Algorithm 1),
4. document scores f_d = sum_t freq(d, t) f_t and arg-max (Eq. 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .network import TermNetworkFactory


def initial_term_scores(X_labeled: sp.spmatrix, y_labeled: np.ndarray, n_classes: int) -> np.ndarray:
    """Eq. 1: y[t, c] = sum_{d in D_L, class c} freq(d, t) / sum_{d in D_L} freq(d, t)."""
    X_labeled = sp.csr_matrix(X_labeled)
    onehot = sp.csr_matrix(
        (np.ones(len(y_labeled)), (np.arange(len(y_labeled)), y_labeled)),
        shape=(len(y_labeled), n_classes),
    )
    num = np.asarray((X_labeled.T @ onehot).todense())
    den = np.asarray(X_labeled.sum(axis=0)).ravel()
    out = np.zeros_like(num)
    pos = den > 0
    out[pos] = num[pos] / den[pos, None]
    return out


def propagate(S, Y: np.ndarray, alpha: float, max_iter: int = 400, tol: float = 1e-10) -> np.ndarray:
    """Iterate F <- alpha S F + (1 - alpha) Y starting from F = Y.

    Stops after ``max_iter`` iterations (the notebook's fixed 400) or earlier once the
    largest change is below ``tol``. Y may hold several label sets side by side as
    columns; propagation is linear, so they are processed in one pass.
    """
    F = Y
    if alpha == 0.0:
        return Y.copy()
    base = (1.0 - alpha) * Y
    # When alpha**max_iter is below tol the iteration ends at the fixed point
    # (I - alpha S) F = (1 - alpha) Y to within tol, so solve that system directly
    # if it saves many iterations. alpha = 1 (or tol = 0) always iterates.
    if 0.0 < tol and alpha < 1.0 and alpha ** max_iter < tol and np.log(tol) / np.log(alpha) > 30:
        n = Y.shape[0]
        if sp.issparse(S):
            lu = spla.splu(sp.csc_matrix(sp.identity(n) - alpha * S))
            return lu.solve(np.asarray(base))
        return sla.solve(np.eye(n) - alpha * np.asarray(S), base)
    for _ in range(max_iter):
        F_new = alpha * (S @ F) + base
        if np.max(np.abs(F_new - F)) < tol:
            return F_new
        F = F_new
    return F


def document_scores(X_docs: sp.spmatrix, F: np.ndarray, doc_weighting: str = "tf") -> np.ndarray:
    """Eq. 2 scores. ``doc_weighting="binary"`` uses term presence instead of frequency."""
    if doc_weighting == "binary":
        X_docs = sp.csr_matrix(X_docs, copy=True)
        X_docs.data[:] = 1.0
    elif doc_weighting != "tf":
        raise ValueError(f"unknown doc_weighting={doc_weighting!r}")
    return np.asarray(X_docs @ F)


@dataclass
class TCTNConfig:
    include_self: bool = True
    symmetrize: str = "union"
    weighting: str = "minmax_columns"
    kappa_variant: str = "paper"
    tie_break: str = "argpartition"
    doc_weighting: str = "tf"
    max_iter: int = 400
    tol: float = 1e-10

    @classmethod
    def preset(cls, name: str) -> "TCTNConfig":
        if name == "notebook":
            return cls()
        if name == "clean":
            return cls(include_self=False, symmetrize="union", weighting="clip", tie_break="index")
        raise ValueError(f"unknown preset {name!r} (use 'notebook' or 'clean')")


class TCTN:
    """Transductive classifier over a fixed collection (all documents known up front)."""

    def __init__(self, X: sp.spmatrix, n_classes: int, config: TCTNConfig | None = None, k_max: int = 60):
        self.X = sp.csr_matrix(X, dtype=np.float64)
        self.n_classes = n_classes
        self.config = config or TCTNConfig()
        self.networks = TermNetworkFactory(
            self.X,
            k_max=k_max,
            include_self=self.config.include_self,
            symmetrize=self.config.symmetrize,
            weighting=self.config.weighting,
            kappa_variant=self.config.kappa_variant,
            tie_break=self.config.tie_break,
        )

    def term_scores(self, Y: np.ndarray, sf: int, k: int, alpha: float) -> np.ndarray:
        S = self.networks.S(sf, k)
        return propagate(S, Y, alpha, self.config.max_iter, self.config.tol)

    def predict(self, labeled_idx: np.ndarray, y_labeled: np.ndarray, sf: int, k: int, alpha: float) -> np.ndarray:
        """Class index for every document of the collection."""
        Y = initial_term_scores(self.X[labeled_idx], y_labeled, self.n_classes)
        F = self.term_scores(Y, sf, k, alpha)
        return document_scores(self.X, F, self.config.doc_weighting).argmax(axis=1)
