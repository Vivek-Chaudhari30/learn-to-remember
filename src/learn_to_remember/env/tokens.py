"""Token counting and truncation for tool outputs.

The reward's cost term counts tool-output tokens, and outputs are capped in
tokens, so counting must use the policy's own tokenizer in real runs.
`WhitespaceCounter` is a dependency-free stand-in for unit tests only.
"""

from __future__ import annotations

TRUNC_MARKER = "\n[... truncated]"


class WhitespaceCounter:
    """1 token = 1 whitespace-separated word. For tests only."""

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> tuple[str, int, bool]:
        words = text.split()
        if len(words) <= max_tokens:
            return text, len(words), False
        keep = max(0, max_tokens - self.count(TRUNC_MARKER))
        out = " ".join(words[:keep]) + TRUNC_MARKER
        return out, self.count(out), True


class HFTokenCounter:
    """Counts with a Hugging Face tokenizer (e.g. Qwen2.5's)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._marker = len(self._ids(TRUNC_MARKER))

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def count(self, text: str) -> int:
        return len(self._ids(text))

    def truncate(self, text: str, max_tokens: int) -> tuple[str, int, bool]:
        ids = self._ids(text)
        if len(ids) <= max_tokens:
            return text, len(ids), False
        keep = max(0, max_tokens - self._marker)
        while True:
            # decode -> re-encode can shift a token at the cut, so re-check the cap
            out = self.tok.decode(ids[:keep]) + TRUNC_MARKER
            n = self.count(out)
            if n <= max_tokens or keep == 0:
                return out, n, True
            keep -= 1


def load_qwen_counter(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", local_only: bool = False):
    """Qwen2.5 0.5B/1.5B share one tokenizer."""
    from transformers import AutoTokenizer
    return HFTokenCounter(AutoTokenizer.from_pretrained(model_id, local_files_only=local_only))
