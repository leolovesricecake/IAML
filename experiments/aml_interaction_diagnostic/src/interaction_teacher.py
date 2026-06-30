from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Sequence

from src.candidate_graph import CandidateEdge


@dataclass
class EdgeInteractionScore:
    """Finite-difference interaction scores for one candidate edge."""

    i: int
    j: int
    edge_type: str
    original_score: float
    single_mask_score_i: float
    single_mask_score_j: float
    pair_mask_score_ij: float
    interaction_score: float
    absolute_interaction_score: float
    normalized_interaction_score: float


@dataclass
class InteractionResult:
    """All edge interaction scores and the scorer cache used to compute them."""

    edge_scores: List[EdgeInteractionScore]
    score_cache: Dict[FrozenSet[int], float]


def compute_interactions(
    edges: Sequence[CandidateEdge],
    scorer: Callable[[FrozenSet[int]], float],
    normalizer_eps: float = 1e-12,
) -> InteractionResult:
    """Compute I_ij = s(x)-s(x\\i)-s(x\\j)+s(x\\ij), caching repeated masks."""
    cache: Dict[FrozenSet[int], float] = {}

    def score(masked_words: Iterable[int]) -> float:
        key = frozenset(masked_words)
        if key not in cache:
            cache[key] = float(scorer(key))
        return cache[key]

    original = score(())
    edge_scores = []
    for edge in edges:
        score_i = score((edge.i,))
        score_j = score((edge.j,))
        score_ij = score((edge.i, edge.j))
        interaction = original - score_i - score_j + score_ij
        edge_scores.append(
            EdgeInteractionScore(
                i=edge.i,
                j=edge.j,
                edge_type=edge.edge_type,
                original_score=original,
                single_mask_score_i=score_i,
                single_mask_score_j=score_j,
                pair_mask_score_ij=score_ij,
                interaction_score=interaction,
                absolute_interaction_score=abs(interaction),
                normalized_interaction_score=interaction / (abs(original) + normalizer_eps),
            )
        )
    return InteractionResult(edge_scores=edge_scores, score_cache=cache)
