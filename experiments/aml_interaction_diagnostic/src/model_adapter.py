from typing import FrozenSet, Sequence

from src.word_units import WordUnit


class ProbabilityScorer:
    """Callable target-probability scorer for interaction teacher queries."""

    def __init__(self, adapter, text: str, words: Sequence[WordUnit], mode: str = "delete"):
        self.adapter = adapter
        self.text = text
        self.words = words
        self.mode = mode

    def __call__(self, masked_words: FrozenSet[int]) -> float:
        return self.adapter.score(self.text, self.words, masked_words, mode=self.mode)
