from __future__ import annotations

import re


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


class QueryAnalyzer:
    """Shared English/CJK analyzer used by BM25, fallback ranking and evaluation."""

    def __init__(self, cjk_ngram_size: int = 2) -> None:
        self.cjk_ngram_size = max(1, cjk_ngram_size)

    def tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens = _WORD.findall(lowered)
        for run in _CJK_RUN.findall(lowered):
            if len(run) <= self.cjk_ngram_size:
                tokens.append(run)
            else:
                tokens.extend(
                    run[index : index + self.cjk_ngram_size]
                    for index in range(len(run) - self.cjk_ngram_size + 1)
                )
        return tokens

    def overlap(self, left: str, right: str) -> float:
        left_tokens = set(self.tokens(left))
        if not left_tokens:
            return 0.0
        right_tokens = set(self.tokens(right))
        return len(left_tokens & right_tokens) / len(left_tokens)
