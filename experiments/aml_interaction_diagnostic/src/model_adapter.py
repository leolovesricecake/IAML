from typing import FrozenSet, Sequence

from src.aml_adapter import MockAmlAdapter
from src.word_units import WordUnit


class ProbabilityScorer:
    """Callable target-probability scorer for interaction teacher queries."""

    def __init__(self, adapter: MockAmlAdapter, text: str, words: Sequence[WordUnit]):
        self.adapter = adapter
        self.text = text
        self.words = words

    def __call__(self, masked_words: FrozenSet[int]) -> float:
        return self.adapter.score(self.text, self.words, masked_words)
