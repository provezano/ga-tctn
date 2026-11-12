import json

import numpy as np
import pytest

from ga_tctn.cli import main
from ga_tctn.protocol import Evaluator, feasible, make_splits
from ga_tctn.search import GAConfig, genetic_algorithm, grid_search, random_search


def test_splits_are_disjoint_and_deterministic():
    y = np.repeat([0, 1, 2], [20, 30, 50])
    s1 = make_splits(y, 10, n_splits=3, seed=7)
    s2 = make_splits(y, 10, n_splits=3, seed=7)
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a.labeled, b.labeled)
        np.testing.assert_array_equal(a.test, b.test)
        assert len(a.labeled) == 30 and len(a.validation) == 10 and len(a.test) == 60
        assert np.bincount(y[a.labeled]).tolist() == [10, 10, 10]
        allidx = np.concatenate([a.labeled, a.validation, a.test])
        assert len(np.unique(allidx)) == 100
    assert not np.array_equal(s1[0].labeled, s1[1].labeled)
    assert feasible(y, 20) and not feasible(y, 21)
    with pytest.raises(ValueError):
        make_splits(y, 21)


class Fake:
    """Smooth synthetic objective with optimum at sf=4, k=21, alpha=0.3."""

    class S:
        def __init__(self, v):
            self.validation = v
            self.test = v

    def __call__(self, sf, k, alpha):
        return self.S(-abs(sf - 4) - abs(k - 21) / 10 - abs(alpha - 0.3))


def test_grid_search_covers_125():
    r = grid_search(Fake())
    assert r.n_evaluations == 125
    assert r.best == (4, 17, 0.3)


def test_ga_is_seeded_and_improves():
    a = genetic_algorithm(Fake(), np.random.default_rng(1))
    b = genetic_algorithm(Fake(), np.random.default_rng(1))
    assert a.best == b.best and a.history == b.history
    assert a.history == sorted(a.history)  # elitism keeps the best
    assert a.best[0] == 4 and abs(a.best[1] - 21) <= 5
    assert a.generations <= GAConfig().max_generations


def test_random_search_patience():
    r = random_search(Fake(), np.random.default_rng(0), max_iter=100, patience=5)
    assert r.n_evaluations <= 100
    assert len(r.history) == r.n_evaluations


def test_evaluator_memoises(toy):
    splits = make_splits(toy.y, 2, n_splits=3, seed=0)
    ev = Evaluator(toy.X, toy.y, 2, splits)
    s1 = ev(4, 5, 0.5)
    s2 = ev(4, 5, 0.5)
    assert s1 is s2 and ev.n_evaluations == 1
    assert 0.8 < s1.test <= 1.0


def test_cli_tune(tmp_path, toy, capsys):
    lines = ["@relation toy"] + [f"@attribute w{i} numeric" for i in range(toy.X.shape[1])]
    lines += ["@attribute class_atr {a,b}", "@data"]
    for row, c in zip(toy.X.toarray(), toy.y):
        lines.append(",".join(str(int(v)) for v in row) + "," + "ab"[c])
    path = tmp_path / "toy.arff"
    path.write_text("\n".join(lines) + "\n")
    out = tmp_path / "r.json"
    rc = main(["tune", str(path), "--term-fraction", "1", "--labeled", "1", "30", "31",
               "--methods", "grid", "ga", "--runs", "2", "--repetitions", "2", "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["skipped_sizes"] == [31]
    assert [e["labeled_per_class"] for e in rep["results"]] == [1, 30]
    assert set(rep["results"][0]["methods"]) == {"grid", "ga"}
