import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence


def compute_interaction_strengths(interaction_scores: Sequence[float], topk: int = 3) -> Dict[str, float]:
    """Aggregate edge-level signed interactions into sample-level strengths."""
    if not interaction_scores:
        return {"max_abs": 0.0, "mean_topk_abs": 0.0, "energy": 0.0}
    abs_scores = sorted((abs(float(score)) for score in interaction_scores), reverse=True)
    k = max(1, min(int(topk), len(abs_scores)))
    return {
        "max_abs": abs_scores[0],
        "mean_topk_abs": sum(abs_scores[:k]) / k,
        "energy": sum(float(score) ** 2 for score in interaction_scores) / len(interaction_scores),
    }


def assign_quantile_buckets(strengths: Sequence[float]) -> List[str]:
    """Assign low/medium/high buckets by rank tertiles over the full run."""
    if not strengths:
        return []
    ordered = sorted(range(len(strengths)), key=lambda idx: strengths[idx])
    buckets = ["medium"] * len(strengths)
    n = len(strengths)
    for rank, idx in enumerate(ordered):
        fraction = rank / n
        if fraction < 1 / 3:
            buckets[idx] = "low"
        elif fraction >= 2 / 3:
            buckets[idx] = "high"
        else:
            buckets[idx] = "medium"
    return buckets


def assign_fixed_threshold_buckets(strengths: Sequence[float], low_threshold: float, high_threshold: float) -> List[str]:
    """Assign buckets using user-provided fixed thresholds."""
    buckets = []
    for strength in strengths:
        if strength < low_threshold:
            buckets.append("low")
        elif strength >= high_threshold:
            buckets.append("high")
        else:
            buckets.append("medium")
    return buckets


def summarize_bucket_metrics(rows: Iterable[Mapping[str, object]], metric_names: Sequence[str]) -> List[Dict[str, object]]:
    """Summarize mean, std, and n for metrics grouped by interaction bucket."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["bucket"]].append(row)

    summaries = []
    for bucket, bucket_rows in sorted(grouped.items()):
        summary = {"bucket": bucket, "n": len(bucket_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in bucket_rows if row.get(metric) is not None]
            if not values:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = math.sqrt(variance)
        summaries.append(summary)
    return summaries
