"""Term network construction for TCTN.

Terms are linked by a similarity computed from the document co-occurrence
contingency table (Table 2 of the GA-TCTN paper), keeping each term's top-k
most similar terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

SIMILARITY_NAMES = {
    1: "Piatetsky-Shapiro",
    2: "Mutual Information",
    3: "Kappa",
    4: "Yule's Q",
    5: "Support",
}


@dataclass
class Contingency:
    """Joint and marginal document probabilities of term pairs.

    p11[i, j] = p(t_i, t_j) and p1[i] = p(t_i); the other cells follow from them.
    """

    p11: np.ndarray
    p1: np.ndarray

    @classmethod
    def from_documents(cls, X: sp.spmatrix) -> "Contingency":
        Xb = sp.csr_matrix(X, dtype=np.float64, copy=True)
        Xb.data[:] = 1.0
        n = Xb.shape[0]
        co = (Xb.T @ Xb).toarray() / n
        p1 = np.diag(co).copy()
        return cls(p11=co, p1=p1)

    def cells(self):
        p11 = self.p11
        pi = self.p1[:, None]
        pj = self.p1[None, :]
        p10 = pi - p11
        p01 = pj - p11
        p00 = 1.0 - pi - pj + p11
        return p11, p10, p01, p00, pi, pj


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros(np.broadcast(num, den).shape)
    np.divide(num, den, out=out, where=den != 0)
    return out


def similarity(ct: Contingency, sf: int, kappa_variant: str = "paper") -> np.ndarray:
    """Dense term x term similarity matrix for similarity function ``sf`` (1..5).

    Undefined cells (zero denominators, 0 * log 0) are set to 0.
    ``kappa_variant="notebook"`` reproduces the sign used in the 2018 notebook
    (p(ti)p(tj) - p(~ti)p(~tj) instead of the sum); the paper's formula is the default.
    """
    p11, p10, p01, p00, pi, pj = ct.cells()
    qi, qj = 1.0 - pi, 1.0 - pj
    if sf == 1:
        return p11 - pi * pj
    if sf == 2:
        out = np.zeros_like(p11)
        for pxy, px, py in ((p11, pi, pj), (p10, pi, qj), (p01, qi, pj), (p00, qi, qj)):
            den = px * py
            mask = (pxy > 0) & (den > 0)
            ratio = np.ones_like(p11)
            np.divide(pxy, den, out=ratio, where=mask)
            out += np.where(mask, pxy * np.log2(ratio, out=np.zeros_like(ratio), where=mask), 0.0)
        return out
    if sf == 3:
        if kappa_variant == "notebook":
            chance = pi * pj - qi * qj
        else:
            chance = pi * pj + qi * qj
        return _safe_div(p11 + p00 - chance, 1.0 - chance)
    if sf == 4:
        a = p11 * p00
        b = p10 * p01
        return _safe_div(a - b, a + b)
    if sf == 5:
        return p11.copy()
    raise ValueError(f"similarity function must be in 1..5, got {sf}")


def neighbour_order(sim: np.ndarray, k_max: int, include_self: bool) -> np.ndarray:
    """For each term, the indices of its ``k_max`` most similar terms, most similar first.

    Ties are broken by term index. With ``include_self`` the term itself competes
    for a slot, as in the 2018 notebook (np.argpartition over the full row).
    """
    s = sim.copy()
    if not include_self:
        np.fill_diagonal(s, -np.inf)
    k_max = min(k_max, s.shape[1] - (0 if include_self else 1))
    # A stable sort of -s gives descending similarity with ties broken by index.
    return np.argsort(-s, axis=1, kind="stable")[:, :k_max]


def argpartition_neighbours(sim: np.ndarray, k: int, include_self: bool) -> np.ndarray:
    """Top-k neighbours chosen with np.argpartition, exactly as the 2018 notebook does.

    Among terms with equal similarity at the k-th position, the choice is whatever
    numpy's introselect returns; it is deterministic for a given numpy version.
    """
    s = sim
    if not include_self:
        s = sim.copy()
        np.fill_diagonal(s, -np.inf)
    k = min(k, s.shape[1] - (0 if include_self else 1))
    return np.argpartition(s, -k, axis=1)[:, -k:]


def build_weights(
    sim: np.ndarray,
    order: np.ndarray,
    k: int,
    symmetrize: str = "union",
    weighting: str = "minmax_columns",
) -> np.ndarray | sp.csr_matrix:
    """Top-k term network weight matrix.

    symmetrize: "union" links i-j if either is in the other's top-k (the notebook's
    'symmetric' kNN); "none" keeps the directed top-k lists.
    weighting: "minmax_columns" applies the notebook's column-wise min-max scaling to
    the dense masked similarity matrix (returns a dense array); "clip" keeps the raw
    similarity on the edges and sets negative weights to zero (returns CSR).
    """
    m = sim.shape[0]
    k = min(k, order.shape[1])
    rows = np.repeat(np.arange(m), k)
    cols = order[:, :k].ravel()
    mask = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(m, m))
    if symmetrize == "union":
        mask = mask.maximum(mask.T)
    elif symmetrize != "none":
        raise ValueError(f"unknown symmetrize={symmetrize!r}")
    mask = mask.tocoo()
    vals = sim[mask.row, mask.col]
    if weighting == "clip":
        vals = np.maximum(vals, 0.0)
        W = sp.csr_matrix((vals, (mask.row, mask.col)), shape=(m, m))
        W.eliminate_zeros()
        return W
    if weighting == "minmax_columns":
        W = np.zeros((m, m))
        W[mask.row, mask.col] = vals
        lo = W.min(axis=0)
        hi = W.max(axis=0)
        rng = hi - lo
        rng[rng == 0] = 1.0  # same convention as sklearn's MinMaxScaler
        return (W - lo) / rng
    raise ValueError(f"unknown weighting={weighting!r}")


def normalize(W: np.ndarray | sp.spmatrix) -> np.ndarray | sp.csr_matrix:
    """S = D^-1/2 W D^-1/2 with D the row sums of W; isolated terms get zero rows/columns."""
    d = np.asarray(W.sum(axis=1)).ravel()
    inv = np.zeros_like(d)
    pos = d > 0
    inv[pos] = 1.0 / np.sqrt(d[pos])
    if sp.issparse(W):
        Dm = sp.diags(inv)
        return (Dm @ W @ Dm).tocsr()
    return inv[:, None] * W * inv[None, :]


class TermNetworkFactory:
    """Caches the similarity matrices and neighbour lists so S(sf, k) is cheap to rebuild."""

    def __init__(
        self,
        X: sp.spmatrix,
        k_max: int = 60,
        include_self: bool = True,
        symmetrize: str = "union",
        weighting: str = "minmax_columns",
        kappa_variant: str = "paper",
        tie_break: str = "argpartition",
        cache_size: int = 64,
    ):
        self.ct = Contingency.from_documents(X)
        self.k_max = k_max
        self.include_self = include_self
        self.symmetrize = symmetrize
        self.weighting = weighting
        self.kappa_variant = kappa_variant
        if tie_break not in ("argpartition", "index"):
            raise ValueError(f"unknown tie_break={tie_break!r}")
        self.tie_break = tie_break
        self._sim: dict[int, np.ndarray] = {}
        self._order: dict[int, np.ndarray] = {}
        self._S: dict[tuple[int, int], object] = {}
        self._cache_size = cache_size

    @property
    def n_terms(self) -> int:
        return self.ct.p1.size

    def sim(self, sf: int) -> np.ndarray:
        if sf not in self._sim:
            self._sim[sf] = similarity(self.ct, sf, self.kappa_variant)
            self._order[sf] = neighbour_order(self._sim[sf], self.k_max, self.include_self)
        return self._sim[sf]

    def S(self, sf: int, k: int):
        key = (sf, k)
        if key in self._S:
            S = self._S.pop(key)
            self._S[key] = S  # move to most-recent position
            return S
        sim = self.sim(sf)
        if self.tie_break == "argpartition":
            order = argpartition_neighbours(sim, k, self.include_self)
        else:
            order = self._order[sf]
        W = build_weights(sim, order, k, self.symmetrize, self.weighting)
        S = normalize(W)
        if len(self._S) >= self._cache_size:
            self._S.pop(next(iter(self._S)))
        self._S[key] = S
        return S
