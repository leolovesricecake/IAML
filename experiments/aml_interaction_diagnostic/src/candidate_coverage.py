from typing import Iterable, List, Sequence, Set, Tuple


Pair = Tuple[int, int]


def normalize_pair(pair: Pair) -> Pair:
    """Return a canonical undirected pair."""
    i, j = pair
    return (min(i, j), max(i, j))


def recall_at_k(all_pairs_ranked: Sequence[Pair], candidate_edges: Iterable[Pair], k: int) -> float:
    """CandidateRecall@K against an all-pairs ranked reference."""
    if k <= 0:
        return 0.0
    topk = {normalize_pair(pair) for pair in all_pairs_ranked[:k]}
    candidates = {normalize_pair(pair) for pair in candidate_edges}
    return len(topk & candidates) / k


def recall_report(all_pairs_ranked: Sequence[Pair], candidate_edges: Iterable[Pair], ks=(1, 3, 5)) -> dict:
    """Return recall metrics for the configured K values."""
    return {f"recall@{k}": recall_at_k(all_pairs_ranked, candidate_edges, k) for k in ks}
