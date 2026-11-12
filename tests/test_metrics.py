import numpy as np
import pytest

from ga_tctn.metrics import accuracy, weighted_f1


@pytest.mark.parametrize("seed", range(5))
def test_weighted_f1_matches_sklearn(seed):
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, 200)
    p = np.where(rng.random(200) < 0.6, y, rng.integers(0, 5, 200))
    p[p == 4] = 3  # one class never predicted
    expected = sk.f1_score(y, p, average="weighted", zero_division=0)
    assert weighted_f1(y, p, 5) == pytest.approx(expected, abs=1e-12)


def test_perfect_and_accuracy():
    y = np.array([0, 1, 2, 2])
    assert weighted_f1(y, y, 3) == 1.0
    assert accuracy(y, np.array([0, 1, 0, 2])) == 0.75
