import re
from typing import Iterable, List, Optional, Sequence, Tuple

from src.word_units import AlignmentResult, WordUnit


def _as_list(value) -> List:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _special_token_ids(tokenizer) -> set:
    ids = set()
    for name in ("cls_token_id", "sep_token_id", "pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is not None:
            ids.add(token_id)
    return ids


def _word_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _encoding_word_ids(encoding) -> Optional[List[Optional[int]]]:
    word_ids = getattr(encoding, "word_ids", None)
    if callable(word_ids):
        return list(word_ids())
    if isinstance(encoding, dict) and "_word_ids" in encoding:
        return list(encoding["_word_ids"])
    return None


def align_text_to_tokens(tokenizer, text: str) -> AlignmentResult:
    """Align natural whitespace words to tokenizer subword indices.

    Special tokens and padding are excluded from candidate word nodes. The
    returned subword indices are positions in the tokenizer output, so masking a
    word group can remove every subword for that natural word.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_attention_mask=True,
        add_special_tokens=True,
        truncation=True,
    )
    input_ids = _as_list(encoding["input_ids"])
    attention_mask = _as_list(encoding.get("attention_mask", [1] * len(input_ids)))
    offsets = _as_list(encoding.get("offset_mapping", []))
    spans = _word_spans(text)
    if not spans:
        return AlignmentResult(text, input_ids, attention_mask, [], [], skipped=True, skip_reason="empty_text")

    special_ids = _special_token_ids(tokenizer)
    word_ids = _encoding_word_ids(encoding)
    grouped = {idx: [] for idx in range(len(spans))}

    for token_idx, token_id in enumerate(input_ids):
        if token_id in special_ids:
            continue
        if token_idx < len(attention_mask) and attention_mask[token_idx] == 0:
            continue
        if word_ids is not None and token_idx < len(word_ids) and word_ids[token_idx] is not None:
            word_idx = word_ids[token_idx]
            if 0 <= word_idx < len(spans):
                grouped[word_idx].append(token_idx)
            continue
        if token_idx >= len(offsets):
            continue
        start, end = offsets[token_idx]
        if start == end:
            continue
        for word_idx, (word_start, word_end, _word) in enumerate(spans):
            if _overlap(start, end, word_start, word_end):
                grouped[word_idx].append(token_idx)
                break

    words: List[WordUnit] = []
    for word_idx, (start, end, word_text) in enumerate(spans):
        subword_indices = grouped[word_idx]
        if not subword_indices:
            return AlignmentResult(
                text,
                input_ids,
                attention_mask,
                [],
                [],
                skipped=True,
                skip_reason=f"unaligned_word:{word_text}",
            )
        words.append(WordUnit(word_idx, word_text, subword_indices, start, end))

    candidate_indices = [idx for word in words for idx in word.subword_indices]
    return AlignmentResult(text, input_ids, attention_mask, words, candidate_indices)
