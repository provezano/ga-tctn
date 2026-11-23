# GA-TCTN

This is a 2026 reimplementation of

> F. P. Coutinho, S. O. Rezende and R. G. Rossi. GA-TCTN: a framework for hyper-parameter optimization and text classification using transductive semi-supervised learning through term networks. *Learning and Nonlinear Models* 17(2):27–41, 2019. https://doi.org/10.21528/lmln-vol17-no2-art3

It is not the code used for the paper; that code was lost. The classifier (TCTN) follows the first author's own 2018 TCTN notebook. On the CSTR collection it gives the same term network and the same predictions as that notebook, once the notebook's choices are switched on. The genetic algorithm, the grid and random searches, and the evaluation protocol are rebuilt from the text of the paper.

## What the method does

TCTN (Transductive Classification through Term Networks; Rossi, Rezende and Lopes, CICLing 2015) classifies a collection of documents when only a few of them have labels.

1. **Term network.** Every pair of terms gets a similarity computed from how often they appear in the same documents (the 2x2 contingency table of document co-occurrence). Five similarity functions are available: Piatetsky-Shapiro, mutual information, kappa, Yule's Q and support. Each term is then linked to its k most similar terms.
2. **Initial term scores.** From the labelled documents, a term's score for class c is the share of its occurrences that fall in documents of class c (Eq. 1 of the paper).
3. **Propagation.** Scores spread over the network as in Learning with Local and Global Consistency: F ← α S F + (1 − α) Y, with S = D^-1/2 W D^-1/2.
4. **Classification.** A document's score for a class is the sum of its term frequencies times the term scores (Eq. 2), and the document goes to the class with the highest score.

TCTN has three hyper-parameters: the similarity function SF (1–5), the number of neighbours k (1–60) and α in [0, 1]. GA-TCTN tunes them with a genetic algorithm whose fitness is the weighted F1 on a validation set:

| Setting | Value |
|---|---|
| individual | [SF, k, α]: two integers and one real |
| population | 50 |
| generations | at most 100; stop after 5 without improvement of the best |
| selection | tournament of size 3 |
| crossover | uniform, applied to consecutive pairs with probability 0.8, each gene swapped with probability 0.5 |
| mutation | probability 0.1 per individual and 0.1 per gene; uniform integer reset for SF and k, Gaussian step N(0, 0.2) clipped to [0, 1] for α |
| elitism | 1 |

For comparison the package also runs the paper's two baselines: a 125-point grid (k ∈ {1, 7, 17, 37, 57}, α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, all five SF) and random search (at most 100 draws, stopping after 5 without improvement).

The evaluation protocol is the paper's. Each collection is reduced to the 10% of terms with the highest summed TF-IDF. For n labelled documents per class (1, 10, 20, 30, 40, 50) the documents are split 10 times at random: n per class are labelled, 10% of all documents form the validation set, and the rest are the test set. Validation and test documents are unlabelled during propagation. A hyper-parameter triple is scored by the mean weighted F1 over the 10 splits, on validation for the search and on test for reporting. A size n is skipped when some class has fewer than n documents, which reproduces the sizes the paper reports for each collection. Random search and the GA are each run 10 times.

## Install

Python 3.10 or newer.

```
python -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest
```

The only runtime dependencies are NumPy and SciPy. The tests also use scikit-learn to check the F1 computation.

## Data

The paper uses 25 collections from the ICMC-USP benchmark (Rossi, Marcacini and Rezende, 2013), distributed as ARFF files at `http://sites.labic.icmc.usp.br/text_collections/`. That site was offline when this was written (November 2026). Eleven of them (Fbis, Hitech, Re0, Re1, Tr11, Tr12, Tr23, Tr31, Tr41, Tr45 and WAP) are also in the CLUTO toolkit's dataset archive, and this script fetches them from the Internet Archive's copy of it and checks its SHA-256. Tr21, which the paper also uses, is not in that archive.

```
.venv/bin/python -I scripts/download_data.py data/              # all eleven
.venv/bin/python -I scripts/download_data.py data/ re0 tr45     # a subset
```

No data is shipped with this repository. The validation below also uses CSTR, which is not in the CLUTO archive. If you have the ICMC-USP file, put it at `data/CSTR.arff`.

The loader reads ARFF files (dense or sparse, the class being the last nominal attribute unless `--class-attribute` says otherwise) and CLUTO `.mat` files with their `.rclass` labels.

## Run

```
# statistics of a collection and the labelled-set sizes it allows
ga-tctn info data/re0.mat

# one hyper-parameter triple, 10 labelled documents per class
ga-tctn classify data/re0.mat --labeled 10 --sf 4 --k 21 --alpha 0.5

# grid, random search and GA for every feasible labelled-set size, as in the paper
ga-tctn tune data/re0.mat --out re0.json
.venv/bin/python -I scripts/compare_with_paper.py re0.json
```

`python -I -m ga_tctn` works the same way as `ga-tctn`. All randomness comes from `--seed` (default 0), so a run is repeatable. The splits depend only on the seed and n, and every method sees the same splits.

`--preset notebook` (the default) builds the network as the 2018 notebook does. `--preset clean` is a plainer variant: no self-links, raw similarities with negative values set to zero, ties broken by term index. Both are described under "Deviations and assumptions".

## Validation

`sh scripts/run_validation.sh data results/raw` reruns everything below (with the package's virtual environment on `PATH`), and `scripts/compare_with_paper.py results/raw/*.json` prints the table. Weighted F1 on the test documents, mean over the 10 splits. "Paper" columns are copied from Tables 5–7. For grid search there is one run. For random search the table gives the best of our 10 runs. For the GA it gives the best of 10 runs ("max") and the run with the best validation score ("by-val"). The last column is the mean and standard deviation over the 10 GA runs. All rows use `--seed 0` and the default `notebook` preset.

| Collection | n/class | Grid paper | Grid ours | Random paper | Random ours | GA best paper | GA best ours (max / by-val) | GA avg paper | GA avg ours (sd) |
|---|---|---|---|---|---|---|---|---|---|
| CSTR | 1 | 0.4089 | 0.4167 | 0.4072 | 0.4081 | 0.4120 | 0.4210 / 0.4171 | 0.4089 | 0.4120 (0.0071) |
| CSTR | 10 | 0.7445 | 0.7339 | 0.7434 | 0.7342 | 0.7461 | 0.7328 / 0.7276 | 0.7457 | 0.7258 (0.0082) |
| CSTR | 20 | 0.7851 | 0.7119 | 0.7860 | 0.7187 | 0.7859 | 0.7179 / 0.7158 | 0.7859 | 0.7158 (0.0021) |
| re0 | 1 | 0.2900 | 0.2422 | 0.2915 | 0.2421 | 0.2916 | 0.2434 / 0.2421 | 0.2916 | 0.2423 (0.0004) |
| re0 | 10 | 0.6125 | 0.6010 | 0.6120 | 0.5987 | 0.6151 | 0.5942 / 0.5901 | 0.6151 | 0.5900 (0.0024) |
| re1 | 1 | 0.3177 | 0.3105 | 0.3183 | 0.3139 | 0.3206 | 0.3125 / 0.3125 | 0.3203 | 0.3118 (0.0006) |
| re1 | 10 | 0.6256 | 0.6179 | 0.6265 | 0.6146 | 0.6321 | 0.6176 / 0.6155 | 0.6319 | 0.6146 (0.0021) |
| tr45 | 1 | 0.4312 | 0.3510 | 0.4355 | 0.3434 | 0.4392 | 0.3514 / 0.3514 | 0.4392 | 0.3436 (0.0051) |
| tr45 | 10 | 0.4110 | 0.5097 | 0.4109 | 0.5083 | 0.4112 | 0.5124 / 0.5095 | 0.4112 | 0.5094 (0.0026) |
| wap | 1 | 0.3327 | 0.3193 | 0.3329 | 0.3177 | 0.3342 | 0.3230 / 0.3230 | 0.3341 | 0.3194 (0.0026) |

The same table for the `clean` preset, and the seed check mentioned below, are in `results/validation.md`. Both presets together took about 8 CPU minutes.

The reimplementation lands close to the paper in some cells and not in others.

- **Close:** CSTR with 1 and 10 labelled documents per class, Re0 with 10, Re1 with 1 and 10, and WAP with 1. Grid search is within 0.015 of the paper in these cells and the GA columns within 0.025, mostly a little lower.
- **Lower than the paper:** Re0 with 1 (about 0.05 lower), Tr45 with 1 (about 0.08 lower) and CSTR with 20 (about 0.07 lower).
- **Higher than the paper:** Tr45 with 10 (about 0.10 higher).

Most of the large gaps are about as big as the variation that comes from the random splits alone. With one labelled document per class, the mean over 10 splits for a fixed triple moves by 0.09 on Tr45 and 0.09 on Re0 when only the split seed changes (0.33 to 0.43 and 0.23 to 0.32 over seeds 0 to 4). On Tr45 with 10 labelled documents it moves from 0.32 to 0.51. Two of the paper's own numbers are odd in the same way: its Tr45 score falls from 0.439 with 1 labelled document to 0.411 with 10. The CSTR gap at 20 labelled documents is the one the split seed does not explain: across seeds it ranges from 0.710 to 0.757, still below the paper's 0.785.

Other likely causes:

- The paper used the ICMC-USP ARFF versions of the collections. Re0, Re1, Tr45 and WAP here come from the CLUTO matrices, which may differ in preprocessing. Re0 and Tr45 match the paper's Table 4 on documents, classes and class balance; the term counts differ by one.
- The 2019 code is lost, and some of its choices are unknown (see the next section): which kappa formula it used, whether documents were scored by frequency or presence, whether terms were their own neighbours, and whether the network was min-max scaled. The `clean` preset changes several of these at once and moves single cells by up to 0.03, in both directions.
- The paper does not say how it picked the best of 10 runs, or whether the GA fitness pooled the 10 splits as here.

On the comparison between search methods, the paper reports GA-TCTN ahead of grid and random search in most cells, by 0.001 to 0.005. Here the best of the 10 GA runs beats grid search in 7 of 10 cells. If the GA keeps only the run with the best validation score, it is ahead in 5 cells and behind in 5. Differences of this size are well inside the split-to-split noise shown above. This reimplementation therefore neither confirms nor refutes an advantage of the GA over grid search. Most GA runs stop after 6 to 10 generations (350 to 560 evaluations) because of the 5-generation patience rule.

## Deviations and assumptions

This section lists where the code follows the 2018 notebook, where it follows the paper instead, and what it assumes where neither says. The term-network and scoring choices can be switched from the command line; the protocol choices are fixed in the code.

Term network (choices taken from the 2018 notebook, `--preset notebook`):

- A term counts as one of its own k nearest neighbours. Under every similarity, a term is at least as similar to itself as to any other term, so k = 1 gives a network of self-links only, with no propagation. The `clean` preset leaves the term itself out.
- The top-k lists are made symmetric: i and j are linked if either is in the other's list.
- The masked similarity matrix is rescaled column by column to [0, 1] (scikit-learn's `MinMaxScaler`), then normalised as D^-1/2 W D^-1/2 with D the row sums. When a column has negative similarities, this rescaling gives non-links a positive weight. The `clean` preset keeps the raw similarity on the links and sets negative weights to zero.
- Among equally similar candidates for the k-th slot, the choice is the one `numpy.argpartition` makes, exactly as in the notebook. `--tie-break index` picks the lowest term index instead. Ties are common because the similarities come from integer document counts.
- Propagation starts from F = Y and runs 400 iterations, as in the notebook, stopping early once nothing changes by more than 1e-10. When α^400 is below that tolerance, the fixed point (I − αS)F = (1 − α)Y is solved directly, which gives the same result to within 1e-10.

Where this code follows the paper rather than the notebook:

- Kappa uses the paper's formula, with chance agreement p(ti)p(tj) + p(¬ti)p(¬tj). The notebook has a minus sign there; `--kappa-variant notebook` reproduces it.
- Algorithm 1 in the paper prints F ← αSF − (1 − α)Y. The minus is taken to be a typo, because LLGC and the notebook both use a plus.
- Document scores use term frequencies (Eq. 2). The notebook ends up using term presence, because it binarises the document matrix in place while building the co-occurrence matrix. `--doc-weighting binary` reproduces that.
- The notebook uses all terms. The paper keeps the 10% with the highest summed TF-IDF, with TF-IDF taken as tf · ln(N/df). The number kept is rounded to the nearest integer.

Search and protocol (the paper is silent or unclear):

- The fitness of a triple is the mean validation weighted F1 over the 10 splits, and each GA or random-search run tunes once over all 10 splits rather than once per split.
- Weighted F1 is scikit-learn's `f1_score(average="weighted")`, which the paper's footnote cites. Eq. 3 of the paper, a harmonic mean of weighted precision and weighted recall, is not quite the same quantity.
- The validation set is 10% of all documents, rounded, drawn uniformly from the documents that are not labelled.
- "indmp = 10" in the paper is read as a per-gene mutation probability of 0.1. Gaussian steps for α are clipped to [0, 1]. Variation follows DEAP's `varAnd`, since the paper says DEAP was used. The elite individual replaces the worst offspring.
- The "best of 10 runs" in the paper is not defined. The table therefore gives both the run with the highest test F1 and the run with the highest validation F1.
- Grid-search ties go to the first triple in SF, k, α order. Random-search and GA ties go to the first triple found.
- The paper reports Friedman and Nemenyi tests over 25 collections. With five collections they are not repeated here.

## How to cite

If you use this code, please cite the paper:

F. P. Coutinho, S. O. Rezende and R. G. Rossi. GA-TCTN: a framework for hyper-parameter optimization and text classification using transductive semi-supervised learning through term networks. *Learning and Nonlinear Models*, 17(2):27–41, 2019. DOI: [10.21528/lmln-vol17-no2-art3](https://doi.org/10.21528/lmln-vol17-no2-art3). Available at https://sbia.org.br/lnlm/publicacoes/vol17-no2/vol17-no2-art3/

```bibtex
@article{coutinho2019gatctn,
  author  = {Coutinho, Felipe Provezano and Rezende, Solange Oliveira and Rossi, Rafael Geraldeli},
  title   = {{GA-TCTN}: A Framework for Hyper-parameter Optimization and Text Classification
             Using Transductive Semi-supervised Learning Through Term Networks},
  journal = {Learning and Nonlinear Models},
  volume  = {17},
  number  = {2},
  pages   = {27--41},
  year    = {2019},
  doi     = {10.21528/lmln-vol17-no2-art3}
}
```

`CITATION.cff` has the same information in machine-readable form.

## License

Apache-2.0, see `LICENSE`.
