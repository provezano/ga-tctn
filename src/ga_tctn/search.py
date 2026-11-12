"""Hyper-parameter search for TCTN: grid search, random search and the GA of GA-TCTN.

All three maximise the mean validation weighted F1 returned by an ``Evaluator``.
An individual is [SF, k, alpha] with SF in 1..5, k in 1..60, alpha in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

SF_RANGE = (1, 5)
K_RANGE = (1, 60)
GRID_K = (1, 7, 17, 37, 57)
GRID_ALPHA = (0.1, 0.3, 0.5, 0.7, 0.9)
GRID_SF = (1, 2, 3, 4, 5)


@dataclass
class SearchResult:
    method: str
    best: tuple[int, int, float]
    validation: float
    test: float
    n_evaluations: int
    generations: int = 0
    history: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        sf, k, a = self.best
        return {"method": self.method, "sf": sf, "k": k, "alpha": a, "validation_f1": self.validation,
                "test_f1": self.test, "n_evaluations": self.n_evaluations, "generations": self.generations,
                "history": self.history}


class _Tracker:
    """Keeps the best triple seen so far (strictly better validation score wins)."""

    def __init__(self, evaluate: Callable):
        self.evaluate = evaluate
        self.best = None
        self.best_score = None
        self.calls = 0

    def __call__(self, sf, k, alpha) -> float:
        self.calls += 1
        sc = self.evaluate(sf, k, alpha)
        if self.best_score is None or sc.validation > self.best_score.validation:
            self.best, self.best_score = (int(sf), int(k), float(alpha)), sc
        return sc.validation

    def result(self, method: str, generations: int = 0, history=None) -> SearchResult:
        return SearchResult(method, self.best, self.best_score.validation, self.best_score.test,
                            self.calls, generations, list(history or []))


def grid_search(evaluate: Callable) -> SearchResult:
    """125 combinations: k in {1,7,17,37,57}, alpha in {.1,...,.9}, SF in 1..5."""
    t = _Tracker(evaluate)
    for sf in GRID_SF:
        for k in GRID_K:
            for a in GRID_ALPHA:
                t(sf, k, a)
    return t.result("grid")


def _random_individual(rng: np.random.Generator) -> list:
    return [int(rng.integers(SF_RANGE[0], SF_RANGE[1] + 1)),
            int(rng.integers(K_RANGE[0], K_RANGE[1] + 1)),
            float(rng.random())]


def random_search(evaluate: Callable, rng: np.random.Generator, max_iter: int = 100, patience: int = 5) -> SearchResult:
    """Uniform sampling; stops after ``max_iter`` draws or ``patience`` draws without improvement."""
    t = _Tracker(evaluate)
    best = -np.inf
    stale = 0
    history = []
    for _ in range(max_iter):
        v = t(*_random_individual(rng))
        if v > best:
            best, stale = v, 0
        else:
            stale += 1
        history.append(best)
        if stale >= patience:
            break
    return t.result("random", history=history)


@dataclass
class GAConfig:
    population: int = 50
    max_generations: int = 100
    patience: int = 5
    cx_prob: float = 0.8
    uniform_cx_prob: float = 0.5
    mut_prob: float = 0.1
    gene_mut_prob: float = 0.1
    tournament: int = 3
    elitism: int = 1
    sigma: float = 0.2
    mu: float = 0.0


def _tournament(rng, fitness: np.ndarray, n: int, size: int) -> np.ndarray:
    cand = rng.integers(0, len(fitness), size=(n, size))
    return cand[np.arange(n), np.argmax(fitness[cand], axis=1)]


def genetic_algorithm(evaluate: Callable, rng: np.random.Generator, cfg: GAConfig | None = None) -> SearchResult:
    """Generational GA with tournament selection, uniform crossover, mixed mutation and elitism.

    Variation follows DEAP's ``varAnd``: consecutive pairs are crossed with probability
    ``cx_prob`` (each gene swapped with ``uniform_cx_prob``); then each individual is
    mutated with probability ``mut_prob``, each gene independently with ``gene_mut_prob``
    (uniform integer reset for SF and k, Gaussian step clipped to [0, 1] for alpha).
    The ``elitism`` best individuals of the previous generation replace the worst
    offspring. Stops after ``max_generations`` or ``patience`` generations without an
    improvement of the best fitness.
    """
    cfg = cfg or GAConfig()
    t = _Tracker(evaluate)
    pop = [_random_individual(rng) for _ in range(cfg.population)]
    fit = np.array([t(*ind) for ind in pop])
    best = fit.max()
    history = [float(best)]
    stale = 0
    gen = 0
    for gen in range(1, cfg.max_generations + 1):
        chosen = _tournament(rng, fit, cfg.population, cfg.tournament)
        off = [list(pop[i]) for i in chosen]
        for i in range(1, len(off), 2):
            if rng.random() < cfg.cx_prob:
                a, b = off[i - 1], off[i]
                for g in range(3):
                    if rng.random() < cfg.uniform_cx_prob:
                        a[g], b[g] = b[g], a[g]
        for ind in off:
            if rng.random() < cfg.mut_prob:
                if rng.random() < cfg.gene_mut_prob:
                    ind[0] = int(rng.integers(SF_RANGE[0], SF_RANGE[1] + 1))
                if rng.random() < cfg.gene_mut_prob:
                    ind[1] = int(rng.integers(K_RANGE[0], K_RANGE[1] + 1))
                if rng.random() < cfg.gene_mut_prob:
                    ind[2] = float(np.clip(ind[2] + rng.normal(cfg.mu, cfg.sigma), 0.0, 1.0))
        off_fit = np.array([t(*ind) for ind in off])
        if cfg.elitism > 0:
            elite = np.argsort(-fit, kind="stable")[: cfg.elitism]
            worst = np.argsort(off_fit, kind="stable")[: cfg.elitism]
            for e, w in zip(elite, worst):
                off[w] = list(pop[e])
                off_fit[w] = fit[e]
        pop, fit = off, off_fit
        if fit.max() > best:
            best, stale = fit.max(), 0
        else:
            stale += 1
        history.append(float(best))
        if stale >= cfg.patience:
            break
    return t.result("ga", generations=gen, history=history)
