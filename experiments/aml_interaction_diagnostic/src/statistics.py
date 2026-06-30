import math
import random
from typing import Callable, Dict, List, Sequence, Tuple


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute Pearson correlation without requiring scipy."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return 0.0
    return numerator / (x_den * y_den)


def _ranks(values: Sequence[float]) -> List[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute Spearman correlation via average ranks."""
    return pearson_correlation(_ranks(xs), _ranks(ys))


def cliffs_delta(high: Sequence[float], low: Sequence[float]) -> float:
    """Cliff's delta; positive means high values tend to exceed low values."""
    if not high or not low:
        return 0.0
    wins = 0
    losses = 0
    for h in high:
        for l in low:
            if h > l:
                wins += 1
            elif h < l:
                losses += 1
    return (wins - losses) / (len(high) * len(low))


def standardized_mean_difference(high: Sequence[float], low: Sequence[float]) -> float:
    """Difference in means divided by pooled standard deviation."""
    if not high or not low:
        return 0.0
    high_mean = mean(high)
    low_mean = mean(low)
    high_var = sum((value - high_mean) ** 2 for value in high) / len(high)
    low_var = sum((value - low_mean) ** 2 for value in low) / len(low)
    pooled = math.sqrt((high_var + low_var) / 2)
    return 0.0 if pooled == 0 else (high_mean - low_mean) / pooled


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap percentile confidence interval for one sample statistic."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates = []
    values = list(values)
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistic(sample))
    estimates.sort()
    lo = estimates[int((alpha / 2) * len(estimates))]
    hi = estimates[min(len(estimates) - 1, int((1 - alpha / 2) * len(estimates)))]
    return lo, hi


def correlation_report(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    """Return Pearson and Spearman correlations for per-example analyses."""
    return {
        "pearson": pearson_correlation(xs, ys),
        "spearman": spearman_correlation(xs, ys),
    }
