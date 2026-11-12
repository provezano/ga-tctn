import numpy as np
import pytest
import scipy.sparse as sp

from ga_tctn.tctn import TCTN, TCTNConfig, initial_term_scores, propagate


def test_initial_scores_eq1():
    X = sp.csr_matrix(np.array([[2, 0, 1], [1, 0, 3], [0, 0, 4]], dtype=float))
    Y = initial_term_scores(X, np.array([0, 1, 1]), 2)
    np.testing.assert_allclose(Y, [[2 / 3, 1 / 3], [0, 0], [1 / 8, 7 / 8]])


def test_propagation_reaches_fixed_point():
    rng = np.random.default_rng(0)
    A = rng.random((10, 10))
    W = (A + A.T) / 2
    d = W.sum(1)
    S = W / np.sqrt(np.outer(d, d))
    Y = rng.random((10, 3))
    alpha = 0.6
    F = propagate(S, Y, alpha, max_iter=1000, tol=1e-13)
    closed = (1 - alpha) * np.linalg.solve(np.eye(10) - alpha * S, Y)
    np.testing.assert_allclose(F, closed, atol=1e-10)
    np.testing.assert_array_equal(propagate(S, Y, 0.0), Y)


def test_fixed_iterations_match_plain_loop():
    rng = np.random.default_rng(2)
    S = rng.random((6, 6)) / 6
    Y = rng.random((6, 2))
    F = Y
    for _ in range(400):
        F = 0.97 * S @ F + 0.03 * Y
    np.testing.assert_allclose(propagate(S, Y, 0.97, max_iter=400, tol=0.0), F)


@pytest.mark.parametrize("preset", ["notebook", "clean"])
@pytest.mark.parametrize("sf", [1, 2, 3, 4, 5])
def test_separable_toy_is_classified(toy, preset, sf):
    model = TCTN(toy.X, 2, TCTNConfig.preset(preset), k_max=10)
    lab = np.array([0, 1, 2, 30, 31, 32])
    pred = model.predict(lab, toy.y[lab], sf, 5, 0.5)
    assert (pred == toy.y).mean() > 0.9


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("alpha", [0.3, 0.8, 0.95, 0.999])
def test_direct_solve_matches_iteration(sparse, alpha):
    rng = np.random.default_rng(4)
    A = rng.random((40, 40)) * (rng.random((40, 40)) < 0.2)
    W = A + A.T
    d = W.sum(1)
    S = W / np.sqrt(np.outer(d, d))
    Y = rng.random((40, 6))
    plain = Y
    for _ in range(400):
        plain = alpha * S @ plain + (1 - alpha) * Y
    S_in = sp.csr_matrix(S) if sparse else S
    np.testing.assert_allclose(propagate(S_in, Y, alpha), plain, atol=1e-8)
