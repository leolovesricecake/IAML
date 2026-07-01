import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


def utc_now_iso() -> str:
    """Return an ISO timestamp with an explicit UTC timezone."""
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo_root) -> str:
    """Return the short git commit, or a stable unavailable marker outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unavailable"
    value = result.stdout.strip()
    return value or "unavailable"


def _finite_values(values: Iterable[object]) -> list[float]:
    finite = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite.append(numeric)
    return finite


def summarize_values(values: Iterable[object]) -> dict[str, object]:
    finite = _finite_values(values)
    if not finite:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {
        "n": len(finite),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(finite),
        "max": max(finite),
    }


def write_csv(path, rows: Sequence[Mapping[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_aopc_curve(per_budget_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in per_budget_rows:
        if row.get("budget") in (None, ""):
            continue
        grouped[int(row["budget"])].append(row)

    curve = []
    for budget in sorted(grouped):
        rows = grouped[budget]
        comp = summarize_values(row.get("comprehensiveness") for row in rows)
        suff = summarize_values(row.get("sufficiency_error") for row in rows)
        log_odds = summarize_values(row.get("log_odds") for row in rows)
        curve.append(
            {
                "budget": budget,
                "n": len(rows),
                "comprehensiveness_mean": comp["mean"],
                "comprehensiveness_std": comp["std"],
                "sufficiency_error_mean": suff["mean"],
                "sufficiency_error_std": suff["std"],
                "log_odds_mean": log_odds["mean"],
                "log_odds_std": log_odds["std"],
            }
        )
    return curve


def diagnostic_metric_summary(per_examples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    valid = [row for row in per_examples if "skip_reason" not in row]
    return {
        "interaction_strength": summarize_values(row.get("interaction_strength") for row in valid),
        "faithfulness_error": summarize_values(row.get("faithfulness_error") for row in valid),
        "comprehensiveness_aopc": summarize_values(
            (row.get("deletion_metrics") or {}).get("comprehensiveness_aopc") for row in valid
        ),
        "sufficiency_aopc": summarize_values(
            (row.get("sufficiency_metrics") or {}).get("sufficiency_aopc") for row in valid
        ),
        "log_odds_aopc": summarize_values((row.get("log_odds_metrics") or {}).get("log_odds_aopc") for row in valid),
    }


def write_diagnostic_outputs(
    run_dir,
    per_budget_rows: Sequence[Mapping[str, object]],
    per_examples: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
    high_vs_low: Mapping[str, object],
    repo_root,
    started_at: str,
    ended_at: str,
) -> dict[str, object]:
    """Write summary.json and AOPC curve CSV for a diagnostic run."""
    run_dir = Path(run_dir)
    curve_rows = build_aopc_curve(per_budget_rows)
    write_csv(
        run_dir / "aopc_curve.csv",
        curve_rows,
        fieldnames=[
            "budget",
            "n",
            "comprehensiveness_mean",
            "comprehensiveness_std",
            "sufficiency_error_mean",
            "sufficiency_error_std",
            "log_odds_mean",
            "log_odds_std",
        ],
    )
    valid_examples = [row for row in per_examples if "skip_reason" not in row]
    summary = {
        "num_examples": len(per_examples),
        "num_valid_examples": len(valid_examples),
        "num_skipped_examples": len(per_examples) - len(valid_examples),
        "num_edges": int(metadata.get("num_edges", 0)),
        "run": {
            "started_at": started_at,
            "ended_at": ended_at,
            "git_commit": git_commit(repo_root),
        },
        "config": metadata.get("config", {}),
        "metadata": {key: value for key, value in metadata.items() if key != "config"},
        "metrics": diagnostic_metric_summary(per_examples),
        "high_vs_low": dict(high_vs_low),
    }
    write_json(run_dir / "summary.json", summary)
    return summary
