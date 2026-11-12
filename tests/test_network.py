import numpy as np
import pytest
import scipy.sparse as sp

from ga_tctn.network import Contingency, build_weights, neighbour_order, normalize, similarity


def brute(X, i, j):
    """Contingency cells for terms i, j computed by counting documents."""
    a, b = X[:, i] > 0, X[:, j] > 0
    n = len(a)
    return (a & b).sum() / n, (a & ~b).sum() / n, (~a & b).sum() / n, (~a & ~b).sum() / n


@pytest.fixture
def X():
    rng = np.random.default_rng(3)
    return (rng.random((50, 8)) < 0.35) * rng.integers(1, 4, (50, 8))


def test_similarities_match_formulas(X):
    ct = Contingency.from_documents(sp.csr_matrix(X))
    sims = {sf: similarity(ct, sf) for sf in range(1, 6)}
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            p11, p10, p01, p00 = brute(X, i, j)
            pi, pj = p11 + p10, p11 + p01
            qi, qj = 1 - pi, 1 - pj
            assert sims[1][i, j] == pytest.approx(p11 - pi * pj)
            mi = sum(pxy * np.log2(pxy / (px * py)) for pxy, px, py in
                     ((p11, pi, pj), (p10, pi, qj), (p01, qi, pj), (p00, qi, qj)) if pxy > 0)
            assert sims[2][i, j] == pytest.approx(mi)
            chance = pi * pj + qi * qj
            assert sims[3][i, j] == pytest.approx((p11 + p00 - chance) / (1 - chance))
            assert sims[4][i, j] == pytest.approx((p11 * p00 - p10 * p01) / (p11 * p00 + p10 * p01))
            assert sims[5][i, j] == pytest.approx(p11)
    for s in sims.values():
        np.testing.assert_allclose(s, s.T, atol=1e-12)
    assert sims[2].min() >= -1e-12  # mutual information is non-negative


def test_neighbour_order_and_self():
    sim = np.array([[1.0, 0.2, 0.5], [0.2, 1.0, 0.2], [0.5, 0.2, 1.0]])
    with_self = neighbour_order(sim, 2, include_self=True)
    without = neighbour_order(sim, 2, include_self=False)
    np.testing.assert_array_equal(with_self[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(without, [[2, 1], [0, 2], [0, 1]])  # tie at row 1 broken by index


@pytest.mark.parametrize("weighting", ["clip", "minmax_columns"])
def test_weights_and_normalisation(weighting):
    rng = np.random.default_rng(1)
    A = rng.normal(size=(12, 12))
    sim = (A + A.T) / 2
    order = neighbour_order(sim, 5, include_self=False)
    W = build_weights(sim, order, 3, "union", weighting)
    Wd = W.toarray() if sp.issparse(W) else W
    assert Wd.min() >= 0
    if weighting == "clip":
        np.testing.assert_allclose(Wd, Wd.T)
        assert ((Wd > 0).sum(axis=1) <= 12).all()
    else:
        assert Wd.max() <= 1 + 1e-12
    S = normalize(W)
    Sd = S.toarray() if sp.issparse(S) else S
    d = Wd.sum(axis=1)
    i, j = np.nonzero(Wd)
    np.testing.assert_allclose(Sd[i, j], Wd[i, j] / np.sqrt(d[i] * d[j]))


def test_argpartition_neighbours_select_same_values_as_stable_sort():
    from ga_tctn.network import argpartition_neighbours

    rng = np.random.default_rng(5)
    sim = rng.integers(0, 4, (30, 30)).astype(float)  # many ties
    sim = (sim + sim.T) / 2
    for k in (1, 4, 9):
        a = argpartition_neighbours(sim, k, include_self=True)
        b = neighbour_order(sim, k, include_self=True)
        rows = np.arange(30)[:, None]
        np.testing.assert_allclose(np.sort(sim[rows, a], 1), np.sort(sim[rows, b], 1))
