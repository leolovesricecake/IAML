from typing import Callable, Dict, Iterable, List, Sequence, Tuple


DEFAULT_BUDGETS = [1, 5, 10, 20, 50]


def comprehensiveness_score(original_score: float, masked_score: float) -> float:
    """Probability drop after deleting top-attribution units; larger is better."""
    return round(float(original_score) - float(masked_score), 12)


def sufficiency_error(original_score: float, kept_score: float) -> float:
    """Probability drop when keeping only top-attribution units; smaller is better."""
    return round(float(original_score) - float(kept_score), 12)


def aopc(step_scores: Sequence[float]) -> float:
    """AML-compatible AOPC aggregation: sum over steps divided by len(steps)+1."""
    return sum(float(score) for score in step_scores) / (len(step_scores) + 1)


def topk_word_indices(word_attributions: Sequence[float], budget_percent: int) -> List[int]:
    """Return top word indices for a percentage budget, matching AML's int truncation."""
    k = int(len(word_attributions) * budget_percent / 100)
    if k <= 0:
        return []
    return sorted(range(len(word_attributions)), key=lambda idx: word_attributions[idx], reverse=True)[:k]


def evaluate_faithfulness_from_scores(
    original_score: float,
    word_attributions: Sequence[float],
    delete_scorer: Callable[[Iterable[int]], float],
    keep_scorer: Callable[[Iterable[int]], float],
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> Dict[str, object]:
    """Evaluate word-level comprehensiveness, sufficiency error, and AOPC curves."""
    comprehensiveness_steps = []
    sufficiency_steps = []
    per_budget = []
    for budget in budgets:
        selected = topk_word_indices(word_attributions, budget)
        if selected:
            masked_score = delete_scorer(selected)
            kept_score = keep_scorer(selected)
            comp = comprehensiveness_score(original_score, masked_score)
            suff = sufficiency_error(original_score, kept_score)
        else:
            comp = 0.0
            suff = 0.0
        comprehensiveness_steps.append(comp)
        sufficiency_steps.append(suff)
        per_budget.append(
            {
                "budget": budget,
                "selected_word_indices": selected,
                "comprehensiveness": comp,
                "sufficiency_error": suff,
            }
        )
    return {
        "budgets": list(budgets),
        "per_budget": per_budget,
        "comprehensiveness_aopc": aopc(comprehensiveness_steps),
        "sufficiency_aopc": aopc(sufficiency_steps),
        "faithfulness_error": aopc(sufficiency_steps) - aopc(comprehensiveness_steps),
    }
