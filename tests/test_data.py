import numpy as np
import scipy.sparse as sp

from ga_tctn.data import load_arff, load_cluto, select_terms, tfidf_term_ranking

DENSE = """% comment
@relation toy
@attribute alpha numeric
@attribute 'beta' numeric
@attribute gamma numeric
@attribute class_atr {pos,neg}
@data
1,0,2,pos
0,3,0,'neg'
"""

SPARSE = """@relation toy
@attribute alpha numeric
@attribute beta numeric
@attribute class_atr {pos,neg}
@attribute gamma numeric
@data
{0 1, 2 pos, 3 2}
{1 3, 2 neg}
"""


def test_dense_and_sparse_arff_agree(tmp_path):
    a = tmp_path / "dense.arff"
    a.write_text(DENSE)
    b = tmp_path / "sparse.arff"
    b.write_text(SPARSE)
    da, sa = load_arff(str(a)), load_arff(str(b))
    assert da.term_names == ["alpha", "beta", "gamma"]
    assert sa.term_names == ["alpha", "beta", "gamma"]
    np.testing.assert_array_equal(da.X.toarray(), [[1, 0, 2], [0, 3, 0]])
    np.testing.assert_array_equal(sa.X.toarray(), da.X.toarray())
    assert da.class_names == ["pos", "neg"] and list(da.y) == [0, 1]
    assert list(sa.y) == [0, 1]


def test_cluto(tmp_path):
    (tmp_path / "c.mat").write_text("3 4 4\n1 2 3 1\n2 5\n4 1\n")
    (tmp_path / "c.mat.rclass").write_text("x\ny\nx\n")
    col = load_cluto(str(tmp_path / "c.mat"))
    np.testing.assert_array_equal(col.X.toarray(), [[2, 0, 1, 0], [0, 5, 0, 0], [0, 0, 0, 1]])
    assert list(col.y) == [0, 1, 0] and col.term_names == ["t0", "t1", "t2", "t3"]


def test_term_selection_by_tfidf():
    X = sp.csr_matrix(np.array([[5, 1, 0, 1], [0, 1, 0, 1], [0, 1, 3, 0]], dtype=float))
    score = tfidf_term_ranking(X)
    assert score[1] == 0.0  # present everywhere -> idf 0
    np.testing.assert_allclose(score[0], 5 * np.log(3))
    np.testing.assert_array_equal(select_terms(X, 0.5), [0, 2])
