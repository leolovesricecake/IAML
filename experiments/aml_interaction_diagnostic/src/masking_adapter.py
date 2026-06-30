from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

from src.word_units import WordUnit


@dataclass
class MaskedInput:
    """A hard-deletion masked input and the original token positions kept."""

    input_ids: List[int]
    attention_mask: List[int]
    kept_token_indices: List[int]


def _word_token_indices(words: Sequence[WordUnit], word_indices: Iterable[int]) -> Set[int]:
    selected = set(word_indices)
    token_indices = set()
    for word in words:
        if word.index in selected:
            token_indices.update(word.subword_indices)
    return token_indices


def _select_tokens(input_ids: Sequence[int], attention_mask: Sequence[int], keep_indices: Set[int]) -> MaskedInput:
    kept = [idx for idx in range(len(input_ids)) if idx in keep_indices]
    return MaskedInput(
        input_ids=[input_ids[idx] for idx in kept],
        attention_mask=[attention_mask[idx] for idx in kept],
        kept_token_indices=kept,
    )


def mask_word_groups(
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    words: Sequence[WordUnit],
    masked_word_indices: Iterable[int],
    required_token_indices: Iterable[int] = (),
) -> MaskedInput:
    """AML-style hard deletion for comprehensiveness: delete selected word groups."""
    remove_token_indices = _word_token_indices(words, masked_word_indices)
    keep_indices = set(range(len(input_ids))) - remove_token_indices
    keep_indices.update(required_token_indices)
    return _select_tokens(input_ids, attention_mask, keep_indices)


def keep_word_groups(
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    words: Sequence[WordUnit],
    kept_word_indices: Iterable[int],
    required_token_indices: Iterable[int] = (),
) -> MaskedInput:
    """AML-style hard deletion for sufficiency: keep selected word groups and required tokens."""
    keep_indices = _word_token_indices(words, kept_word_indices)
    keep_indices.update(required_token_indices)
    return _select_tokens(input_ids, attention_mask, keep_indices)
