import csv
from pathlib import Path
from typing import Sequence

from src.io_utils import ensure_dir


def write_binned_trend_csv(path, strengths: Sequence[float], errors: Sequence[float], n_bins: int = 5) -> None:
    """Write a lightweight binned trend table usable for plotting."""
    path = Path(path)
    ensure_dir(path.parent)
    if not strengths:
        path.write_text("", encoding="utf-8")
        return
    rows = []
    order = sorted(range(len(strengths)), key=lambda idx: strengths[idx])
    bin_size = max(1, len(order) // n_bins)
    for bin_idx in range(n_bins):
        indices = order[bin_idx * bin_size : (bin_idx + 1) * bin_size]
        if not indices:
            continue
        rows.append(
            {
                "bin": bin_idx,
                "mean_interaction_strength": sum(strengths[idx] for idx in indices) / len(indices),
                "mean_faithfulness_error": sum(errors[idx] for idx in indices) / len(indices),
                "n": len(indices),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin", "mean_interaction_strength", "mean_faithfulness_error", "n"])
        writer.writeheader()
        writer.writerows(rows)
