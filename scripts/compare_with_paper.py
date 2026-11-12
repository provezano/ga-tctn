"""Print a Markdown table comparing `ga-tctn tune` reports with Tables 5-10 of the paper.

    python -I scripts/compare_with_paper.py results/*.json
"""

from __future__ import annotations

import json
import sys

# (collection, labelled per class) -> (Grid-TCTN, Random-TCTN, GA-TCTN best, GA-TCTN avg), weighted F1.
PAPER = {
    ("cstr", 1): (0.4089, 0.4072, 0.4120, 0.4089),
    ("cstr", 10): (0.7445, 0.7434, 0.7461, 0.7457),
    ("cstr", 20): (0.7851, 0.7860, 0.7859, 0.7859),
    ("re0", 1): (0.2900, 0.2915, 0.2916, 0.2916),
    ("re0", 10): (0.6125, 0.6120, 0.6151, 0.6151),
    ("re1", 1): (0.3177, 0.3183, 0.3206, 0.3203),
    ("re1", 10): (0.6256, 0.6265, 0.6321, 0.6319),
    ("tr45", 1): (0.4312, 0.4355, 0.4392, 0.4392),
    ("tr45", 10): (0.4110, 0.4109, 0.4112, 0.4112),
    ("wap", 1): (0.3327, 0.3329, 0.3342, 0.3341),
}


def main(paths: list[str]) -> int:
    print("| Collection | n/class | preset | Grid paper | Grid ours | Random paper | Random ours | "
          "GA best paper | GA best ours (max / by-val) | GA avg paper | GA avg ours (sd) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        rep = json.load(open(path))
        name = rep["collection"]["name"]
        for e in rep["results"]:
            n = e["labeled_per_class"]
            paper = PAPER.get((name.lower(), n))
            m = e["methods"]
            g, r, ga = m.get("grid"), m.get("random"), m.get("ga")
            pv = (lambda i: f"{paper[i]:.4f}") if paper else (lambda i: "-")
            print(f"| {name} | {n} | {rep['preset']} | {pv(0)} | {g['max_test_f1']:.4f} | {pv(1)} | "
                  f"{r['max_test_f1']:.4f} | {pv(2)} | {ga['max_test_f1']:.4f} / {ga['best_by_validation_test_f1']:.4f} | "
                  f"{pv(3)} | {ga['mean_test_f1']:.4f} ({ga['sd_test_f1']:.4f}) |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
