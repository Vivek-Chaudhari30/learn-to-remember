#!/usr/bin/env bash
# One-time setup on a Linux CUDA server (e.g. osaka, RTX A6000). Run from the repo root:
#   bash scripts/setup_server.sh
# Creates .venv, installs a CUDA build of torch plus the project, downloads the
# models, and runs the test suite. Safe to re-run. Overrides:
#   PY=python3.12              interpreter used to create the venv (>= 3.10)
#   TORCH_INDEX=<url>          torch wheel index (default CUDA 12.4: driver >= 550)
#   HF_HOME=/big/disk/hf       where models are cached (~7 GB)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), f"need Python >= 3.10, got {sys.version}"'
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q torch --index-url "$TORCH_INDEX"
python -m pip install -q -r requirements.txt
python -m pip install -q -e . --no-deps

python - <<'PYEOF'
import torch, transformers, peft
assert torch.cuda.is_available(), "torch cannot see the GPU (driver / CUDA build mismatch?)"
cap = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | transformers {transformers.__version__} "
      f"| peft {peft.__version__}")
print(f"GPU {torch.cuda.get_device_name(0)} | capability {cap} | "
      f"bf16 {'native' if cap >= (8, 0) else 'NOT native: use DTYPE=fp16'}")
PYEOF

# Models: the two probe candidates (weights) and the 0.5B tokenizer used by tests.
python - <<'PYEOF'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
for m in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-1.7B"]:
    print("downloading", m, "->", snapshot_download(m))
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
PYEOF

python -m pytest -q 2>&1 | tail -3
df -h . | tail -1
echo "Setup done. Next: tmux new -s ltr ; bash scripts/run_all.sh probe   (see docs/RUNBOOK.md)"
