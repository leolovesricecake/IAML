from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.word_units import WordUnit


@dataclass(frozen=True)
class CandidateEdge:
    """A sparse undirected word-pair candidate edge."""

    i: int
    j: int
    edge_type: str


def _merge_edge_type(existing: str, new_type: str) -> str:
    if existing == new_type:
        return existing
    return "adjacent_dependency"


def _add_edge(edges: Dict[Tuple[int, int], str], i: int, j: int, edge_type: str, n_words: int) -> None:
    if i == j or not (0 <= i < n_words) or not (0 <= j < n_words):
        return
    key = (min(i, j), max(i, j))
    edges[key] = _merge_edge_type(edges[key], edge_type) if key in edges else edge_type


def build_candidate_edges(
    words: Sequence[WordUnit],
    dependency_edges: Optional[Iterable[Tuple[int, int]]] = None,
    include_adjacent: bool = True,
    include_dependency: bool = True,
) -> List[CandidateEdge]:
    """Build adjacent and dependency candidate edges over natural word units."""
    edges: Dict[Tuple[int, int], str] = {}
    n_words = len(words)
    if include_adjacent:
        for i in range(max(0, n_words - 1)):
            _add_edge(edges, words[i].index, words[i + 1].index, "adjacent", n_words)
    if include_dependency and dependency_edges:
        for i, j in dependency_edges:
            _add_edge(edges, i, j, "dependency", n_words)
    return [CandidateEdge(i, j, edge_type) for (i, j), edge_type in sorted(edges.items())]


def dependency_edges_from_spacy(text: str, nlp) -> List[Tuple[int, int]]:
    """Create undirected word-index dependency edges from a spaCy pipeline."""
    if nlp is None:
        return []
    doc = nlp(text)
    edges = []
    for token in doc:
        if token.head is token:
            continue
        edges.append((token.i, token.head.i))
    return edges
