"""Draw the M6 figures from the result files into figures/.

    .venv/bin/python scripts/make_figures.py          # uses configs/report.yaml

Figures with no result files are skipped and listed as "not run" in
figures/INDEX.md; a figure with some runs missing says so under the plot.
Needs no GPU and no model (matplotlib only).
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from learn_to_remember.figures import make_all  # noqa: E402
from learn_to_remember.model_utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/report.yaml")
    ap.add_argument("--slug", default=None, help="overrides the spec's model slug")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    spec = load_config(os.path.join(ROOT, a.spec))
    if a.slug:
        spec["slug"] = a.slug
    res = make_all(spec, ROOT, os.path.join(ROOT, a.out))
    for name, path in res["written"].items():
        print(f"wrote {os.path.relpath(path, ROOT)}")
    if res["not_run"]:
        print("not run (no result files): " + ", ".join(res["not_run"]))


if __name__ == "__main__":
    main()
