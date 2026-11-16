"""A small hand-written Okapi BM25, so search is transparent and deterministic.

score(q, d) = sum over query terms t of
    idf(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
with idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)) (the always-positive variant).

Ties are broken by document order (the order the index was built in), so the
same query always returns the same ranking.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ids like "C-1589" stay one token; otherwise "c" matches every contract doc
_TOKEN_RE = re.compile(r"[a-z]+-\d+|[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to "
    "was were will with what who which when where how does do did current currently now "
    "right our we us".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def terms(text: str) -> list[str]:
    """Index/query terms. Unigrams only: adding adjacent bigrams was tried and
    measured worse (top-10 recall of the target doc 268/300 vs 283/300), since
    it lengthens long list documents and the length penalty buries them."""
    return tokenize(text)


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(terms(t)) for t in texts]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.N = len(texts)
        self.avgdl = (sum(self.lens) / self.N) if self.N else 0.0
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for t in dict.fromkeys(terms(query)):             # unique, order kept
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tfs[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                out[i] = out.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        return out

    def search(self, query: str, k: int, allowed=None) -> list[tuple[int, float]]:
        """Top-k (index, score). `allowed(i) -> bool` filters documents first."""
        s = self.scores(query)
        items = s.items() if allowed is None else ((i, v) for i, v in s.items() if allowed(i))
        return sorted(items, key=lambda kv: (-kv[1], kv[0]))[:k]
