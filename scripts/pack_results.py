"""Zip the GPU run's results to bring back (no `zip` binary needed).

    python scripts/pack_results.py                       # -> gpu_results.zip
    python scripts/pack_results.py --rollouts            # also every eval trajectory (large)

Includes results/ (eval JSONs, tables, timing, train logs, samples, figures
index, report), figures/ and logs/. Never includes checkpoints/ or model
weights. rollouts/ folders (every trajectory) are left out unless --rollouts.
"""

from __future__ import annotations

import argparse
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_SUFFIXES = (".pt", ".safetensors", ".bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gpu_results.zip")
    ap.add_argument("--rollouts", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    n, size = 0, 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for top in ("results", "figures", "logs"):
            for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, top)):
                if not a.rollouts and "rollouts" in dirnames:
                    dirnames.remove("rollouts")
                for fn in filenames:
                    if fn.endswith(SKIP_SUFFIXES):      # never ship weights / optimizer state
                        continue
                    p = os.path.join(dirpath, fn)
                    z.write(p, os.path.relpath(p, ROOT))
                    n, size = n + 1, size + os.path.getsize(p)
    print(f"{out}: {n} files, {size / 2**20:.1f} MB before compression, "
          f"{os.path.getsize(out) / 2**20:.1f} MB zipped")


if __name__ == "__main__":
    main()
