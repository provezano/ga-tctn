#!/bin/sh
# Reproduces the validation table: every downloaded collection, both presets, seed 0.
# Usage: sh scripts/run_validation.sh [data-dir] [out-dir]
set -e
DATA=${1:-data}
OUT=${2:-results/raw}
mkdir -p "$OUT"
for preset in notebook clean; do
  for f in "$DATA"/CSTR.arff "$DATA"/re0.mat "$DATA"/tr45.mat "$DATA"/re1.mat "$DATA"/wap.mat; do
    [ -f "$f" ] || continue
    n=$(basename "$f"); n=${n%.*}
    nice -n 15 python -I -m ga_tctn tune "$f" --preset "$preset" --runs 10 --seed 0 --out "$OUT/${n}_${preset}.json"
  done
done
