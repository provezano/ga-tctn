"""Command-line interface: ``ga-tctn info|classify|tune``."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .data import load_collection, select_terms
from .network import SIMILARITY_NAMES
from .protocol import Evaluator, feasible, make_splits
from .search import GAConfig, genetic_algorithm, grid_search, random_search
from .tctn import TCTNConfig

LABELED_SIZES = (1, 10, 20, 30, 40, 50)


def _load(args):
    col = load_collection(args.data, args.class_attribute)
    full = col.summary()
    if args.term_fraction < 1.0:
        col = col.subset_terms(select_terms(col.X, args.term_fraction))
    return col, full


def _config(args) -> TCTNConfig:
    cfg = TCTNConfig.preset(args.preset)
    if args.kappa_variant:
        cfg.kappa_variant = args.kappa_variant
    if args.doc_weighting:
        cfg.doc_weighting = args.doc_weighting
    if args.tie_break:
        cfg.tie_break = args.tie_break
    return cfg


def cmd_info(args) -> int:
    col, full = _load(args)
    out = dict(full)
    out["selected_terms"] = int(col.X.shape[1])
    out["feasible_labeled_sizes"] = [n for n in LABELED_SIZES if feasible(col.y, n)]
    print(json.dumps(out, indent=2))
    return 0


def cmd_classify(args) -> int:
    col, _ = _load(args)
    splits = make_splits(col.y, args.labeled, args.repetitions, args.val_fraction, args.seed)
    ev = Evaluator(col.X, col.y, col.n_classes, splits, _config(args))
    sc = ev(args.sf, args.k, args.alpha)
    print(json.dumps({"sf": args.sf, "similarity": SIMILARITY_NAMES[args.sf], "k": args.k, "alpha": args.alpha,
                      "labeled_per_class": args.labeled, "validation_f1": sc.validation, "test_f1": sc.test,
                      "test_f1_per_repetition": sc.per_split_test}, indent=2))
    return 0


def cmd_tune(args) -> int:
    col, full = _load(args)
    cfg = _config(args)
    sizes = [n for n in args.labeled if feasible(col.y, n)]
    skipped = [n for n in args.labeled if n not in sizes]
    report = {"collection": full, "selected_terms": int(col.X.shape[1]), "preset": args.preset,
              "config": cfg.__dict__, "seed": args.seed, "runs": args.runs, "skipped_sizes": skipped, "results": []}
    for n in sizes:
        splits = make_splits(col.y, n, args.repetitions, args.val_fraction, args.seed)
        ev = Evaluator(col.X, col.y, col.n_classes, splits, cfg)
        entry = {"labeled_per_class": n, "methods": {}}
        for method in args.methods:
            t0 = time.process_time()
            if method == "grid":
                runs = [grid_search(ev)]
            elif method == "random":
                runs = [random_search(ev, np.random.default_rng([args.seed, n, 1, r])) for r in range(args.runs)]
            elif method == "ga":
                runs = [genetic_algorithm(ev, np.random.default_rng([args.seed, n, 2, r]), GAConfig())
                        for r in range(args.runs)]
            else:
                raise SystemExit(f"unknown method {method}")
            tests = np.array([r.test for r in runs])
            vals = np.array([r.validation for r in runs])
            pick = int(np.argmax(vals))  # first run with the best validation score
            entry["methods"][method] = {
                "runs": [r.as_dict() for r in runs],
                "best_by_validation_test_f1": float(tests[pick]),
                "max_test_f1": float(tests.max()),
                "mean_test_f1": float(tests.mean()),
                "sd_test_f1": float(tests.std(ddof=1)) if len(tests) > 1 else 0.0,
                "cpu_seconds": time.process_time() - t0,
            }
            m = entry["methods"][method]
            print(f"{col.name} n={n:<3d} {method:<6s} best-by-val={m['best_by_validation_test_f1']:.4f} "
                  f"max={m['max_test_f1']:.4f} mean={m['mean_test_f1']:.4f} (sd {m['sd_test_f1']:.4f}) "
                  f"best={runs[pick].best} cpu={m['cpu_seconds']:.1f}s", file=sys.stderr, flush=True)
        report["results"].append(entry)
    text = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ga-tctn", description="TCTN transductive text classification with GA hyper-parameter tuning.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("data", help="ARFF file, or CLUTO .mat file with .rclass next to it")
        sp.add_argument("--class-attribute", default=None, help="ARFF class attribute (default: last nominal one)")
        sp.add_argument("--term-fraction", type=float, default=0.1, help="share of terms kept by summed TF-IDF (default 0.1)")
        sp.add_argument("--preset", choices=("notebook", "clean"), default="notebook",
                        help="term-network variant (default: notebook, see README)")
        sp.add_argument("--kappa-variant", choices=("paper", "notebook"), default=None)
        sp.add_argument("--doc-weighting", choices=("tf", "binary"), default=None)
        sp.add_argument("--tie-break", choices=("argpartition", "index"), default=None,
                        help="how equally similar neighbours are chosen for top-k")
        sp.add_argument("--repetitions", type=int, default=10, help="random labelled/validation/test splits (default 10)")
        sp.add_argument("--val-fraction", type=float, default=0.1)
        sp.add_argument("--seed", type=int, default=0)

    sp_info = sub.add_parser("info", help="collection statistics")
    common(sp_info)
    sp_info.set_defaults(func=cmd_info)

    sp_cls = sub.add_parser("classify", help="score one hyper-parameter triple")
    common(sp_cls)
    sp_cls.add_argument("--labeled", type=int, default=10, help="labelled documents per class")
    sp_cls.add_argument("--sf", type=int, choices=range(1, 6), required=True)
    sp_cls.add_argument("--k", type=int, required=True)
    sp_cls.add_argument("--alpha", type=float, required=True)
    sp_cls.set_defaults(func=cmd_classify)

    sp_tune = sub.add_parser("tune", help="grid / random / GA search with the paper's protocol")
    common(sp_tune)
    sp_tune.add_argument("--labeled", type=int, nargs="+", default=list(LABELED_SIZES))
    sp_tune.add_argument("--methods", nargs="+", default=["grid", "random", "ga"], choices=("grid", "random", "ga"))
    sp_tune.add_argument("--runs", type=int, default=10, help="independent runs of random search and GA (default 10)")
    sp_tune.add_argument("--out", default=None, help="write the JSON report here")
    sp_tune.set_defaults(func=cmd_tune)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
