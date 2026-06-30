from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class WordUnit:
    """A natural word and the tokenizer subword indices that realize it."""

    index: int
    text: str
    subword_indices: List[int]
    char_start: Optional[int] = None
    char_end: Optional[int] = None


@dataclass
class AlignmentResult:
    """Tokenization alignment result used by diagnostic masking and ranking."""

    text: str
    input_ids: List[int]
    attention_mask: List[int]
    words: List[WordUnit]
    candidate_token_indices: List[int]
    skipped: bool = False
    skip_reason: Optional[str] = None
