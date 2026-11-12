import numpy as np
import pytest
import scipy.sparse as sp

from ga_tctn.data import Collection


@pytest.fixture
def toy():
    """Two topics with disjoint vocabularies plus a few shared words."""
    rng = np.random.default_rng(0)
    n_per, vocab = 30, 40
    rows = []
    y = []
    for c in range(2):
        for _ in range(n_per):
            v = np.zeros(vocab)
            own = rng.choice(np.arange(c * 15, c * 15 + 15), size=6, replace=False)
            v[own] = rng.integers(1, 4, size=6)
            shared = rng.choice(np.arange(30, 40), size=2, replace=False)
            v[shared] = 1
            rows.append(v)
            y.append(c)
    X = sp.csr_matrix(np.array(rows))
    return Collection(X=X, y=np.array(y), class_names=["a", "b"], term_names=[f"w{i}" for i in range(vocab)], name="toy")
