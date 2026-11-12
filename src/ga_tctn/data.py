"""Loading text collections (ARFF bag-of-words or CLUTO sparse matrices) and term selection."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Collection:
    """A bag-of-words text collection.

    X is a documents x terms CSR matrix of raw term frequencies, y holds class
    indices in [0, n_classes).
    """

    X: sp.csr_matrix
    y: np.ndarray
    class_names: list[str]
    term_names: list[str]
    name: str = ""

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def subset_terms(self, cols: np.ndarray) -> "Collection":
        cols = np.asarray(cols)
        return Collection(
            X=self.X[:, cols].tocsr(),
            y=self.y,
            class_names=self.class_names,
            term_names=[self.term_names[c] for c in cols],
            name=self.name,
        )

    def summary(self) -> dict:
        counts = np.bincount(self.y, minlength=self.n_classes)
        pct = 100.0 * counts / counts.sum()
        return {
            "name": self.name,
            "documents": int(self.X.shape[0]),
            "terms": int(self.X.shape[1]),
            "avg_terms_per_doc": float(np.diff(self.X.indptr).mean()),
            "classes": self.n_classes,
            "class_sd_pct": float(pct.std(ddof=1)) if self.n_classes > 1 else 0.0,
            "majority_pct": float(pct.max()),
            "smallest_class": int(counts.min()),
        }


def _factorize(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    index: dict[str, int] = {}
    out = np.empty(len(labels), dtype=np.int64)
    for i, lab in enumerate(labels):
        if lab not in index:
            index[lab] = len(names)
            names.append(lab)
        out[i] = index[lab]
    return out, names


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


_ATTR_RE = re.compile(r"@attribute\s+('[^']*'|\"[^\"]*\"|\S+)\s+(.*)$", re.IGNORECASE)


def load_arff(path: str, class_attribute: str | None = None) -> Collection:
    """Read a dense or sparse ARFF file whose attributes are term frequencies plus one class attribute.

    The class attribute is the one named ``class_attribute`` or, by default, the last
    nominal attribute in the header. Nothing in the file is executed.
    """
    names: list[str] = []
    nominal: list[bool] = []
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    labels: list[str] = []
    class_idx = -1
    in_data = False
    n_docs = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                low = line.lower()
                if low.startswith("@attribute"):
                    m = _ATTR_RE.match(line)
                    if not m:
                        raise ValueError(f"cannot parse attribute line: {line[:80]}")
                    names.append(_unquote(m.group(1)))
                    nominal.append(m.group(2).strip().startswith("{"))
                elif low.startswith("@data"):
                    in_data = True
                    if class_attribute is not None:
                        class_idx = names.index(class_attribute)
                    else:
                        nom = [i for i, f in enumerate(nominal) if f]
                        if not nom:
                            raise ValueError("no nominal class attribute found")
                        class_idx = nom[-1]
                continue
            label = None
            if line.startswith("{"):
                body = line.strip("{}").strip()
                if body:
                    for item in body.split(","):
                        j_str, v_str = item.strip().split(None, 1)
                        j = int(j_str)
                        if j == class_idx:
                            label = _unquote(v_str)
                        else:
                            v = float(v_str)
                            if v != 0.0:
                                rows.append(n_docs)
                                cols.append(j)
                                vals.append(v)
                if label is None:
                    raise ValueError("sparse ARFF row without class value is not supported")
            else:
                parts = line.split(",")
                if len(parts) != len(names):
                    raise ValueError(f"row {n_docs} has {len(parts)} values, header has {len(names)}")
                label = _unquote(parts[class_idx])
                for j, p in enumerate(parts):
                    if j == class_idx:
                        continue
                    v = float(p)
                    if v != 0.0:
                        rows.append(n_docs)
                        cols.append(j)
                        vals.append(v)
            labels.append(label)
            n_docs += 1
    # Re-index term columns so the class attribute is dropped.
    keep = np.array([j for j in range(len(names)) if j != class_idx])
    remap = np.full(len(names), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    X = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64), (np.asarray(rows), remap[np.asarray(cols, dtype=np.int64)])),
        shape=(n_docs, len(keep)),
    )
    X.sum_duplicates()
    y, class_names = _factorize(labels)
    return Collection(X=X, y=y, class_names=class_names, term_names=[names[j] for j in keep],
                      name=os.path.splitext(os.path.basename(path))[0])


def load_cluto(mat_path: str, rclass_path: str | None = None, clabel_path: str | None = None) -> Collection:
    """Read a CLUTO sparse matrix (``.mat``) with its ``.rclass`` labels and optional ``.clabel`` terms."""
    rclass_path = rclass_path or mat_path + ".rclass"
    clabel_path = clabel_path or mat_path + ".clabel"
    with open(mat_path, "r") as fh:
        header = fh.readline().split()
        n_rows, n_cols = int(header[0]), int(header[1])
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for _ in range(n_rows):
            toks = fh.readline().split()
            indices.extend(int(t) - 1 for t in toks[0::2])
            data.extend(float(t) for t in toks[1::2])
            indptr.append(len(indices))
    X = sp.csr_matrix((np.asarray(data), np.asarray(indices, dtype=np.int64), np.asarray(indptr)),
                      shape=(n_rows, n_cols))
    X.sum_duplicates()
    with open(rclass_path, "r") as fh:
        labels = [ln.strip() for ln in fh if ln.strip()]
    if len(labels) != n_rows:
        raise ValueError(f"{rclass_path}: {len(labels)} labels for {n_rows} rows")
    if os.path.exists(clabel_path):
        with open(clabel_path, "r") as fh:
            terms = [ln.strip() for ln in fh][:n_cols]
    else:
        terms = [f"t{j}" for j in range(n_cols)]
    y, class_names = _factorize(labels)
    name = os.path.splitext(os.path.basename(mat_path))[0]
    return Collection(X=X, y=y, class_names=class_names, term_names=terms, name=name)


def load_collection(path: str, class_attribute: str | None = None) -> Collection:
    if path.endswith(".mat"):
        return load_cluto(path)
    return load_arff(path, class_attribute)


def tfidf_term_ranking(X: sp.csr_matrix) -> np.ndarray:
    """Sum over documents of tf * log(N / df) for each term (0 for unused terms)."""
    X = sp.csr_matrix(X)
    n = X.shape[0]
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
    idf = np.zeros_like(df)
    used = df > 0
    idf[used] = np.log(n / df[used])
    tf_sum = np.asarray(X.sum(axis=0)).ravel()
    return tf_sum * idf


def select_terms(X: sp.csr_matrix, fraction: float = 0.1) -> np.ndarray:
    """Indices of the top ``fraction`` of terms by summed TF-IDF, in original column order."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    score = tfidf_term_ranking(X)
    m = int(round(fraction * X.shape[1]))
    m = max(1, min(m, X.shape[1]))
    order = np.lexsort((np.arange(len(score)), -score))  # descending score, ties by column index
    return np.sort(order[:m])
