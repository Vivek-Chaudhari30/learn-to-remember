"""Build results/report.md (RQ1-RQ5 tables) from the eval result files.

    .venv/bin/python scripts/make_report.py            # uses configs/report.yaml

Missing runs show as "not run". Run it after copying a run's result folder
into results/. It needs no GPU and no model.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from learn_to_remember.model_utils import load_config  # noqa: E402
from learn_to_remember.report import build_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/report.yaml")
    ap.add_argument("--slug", default=None, help="overrides the spec's model slug")
    ap.add_argument("--out", default="results/report.md")
    a = ap.parse_args()
    spec = load_config(os.path.join(ROOT, a.spec))
    if a.slug:
        spec["slug"] = a.slug
    text = build_report(spec, ROOT)
    with open(os.path.join(ROOT, a.out), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
