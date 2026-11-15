# learn-to-remember

Teaching a small open model (~1.5B, default `Qwen/Qwen2.5-1.5B-Instruct`) to **use a text memory well** with a hand-written GRPO + LoRA loop. The memory is a notes index over a synthetic company workspace, imperfect on purpose. The model decides which notes to read, whether to trust them, and when to fall back to the raw files. Reward = correctness + format bonus − λ · tokens read.

Successor to [GRPO-Zero](https://github.com/Vivek-Chaudhari30/GRPO-Zero), which provides the GRPO engine. Inspired by Engram's [*Learning in a world of notes*](https://engram.com/blog/learning-in-a-world-of-notes).

**Status:** M0 (bootstrap). No results yet. Everything reported here will be real measured numbers; see [`docs/RESULTS.md`](docs/RESULTS.md).

- Design spec: [`docs/PROJECT.md`](docs/PROJECT.md)
- Decision log: [`docs/DECISIONS.md`](docs/DECISIONS.md)

## Dev setup (Mac / CPU / MPS)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest -q
```
