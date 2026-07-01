import ast
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo_root) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


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
    return {"n": len(finite), "mean": mean, "std": math.sqrt(variance), "min": min(finite), "max": max(finite)}


def _parse_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if not isinstance(value, str):
        return [value]
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def _expand_aopc_rows(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in results_df.iterrows():
        steps = _parse_list(row.get("steps_k"))
        values = _parse_list(row.get("metric_steps_result"))
        for budget, value in zip(steps, values):
            rows.append(
                {
                    "budget": budget,
                    "metric_step_result": value,
                    "evaluation_metric": row.get("evaluation_metric"),
                    "task": row.get("task"),
                    "explained_model_backbone": row.get("explained_model_backbone"),
                    "interpreter_model_backbone": row.get("interpreter_model_backbone"),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "budget",
                "n",
                "metric_step_result_mean",
                "metric_step_result_std",
                "evaluation_metric",
            ]
        )

    expanded = pd.DataFrame(rows)
    grouped_rows = []
    for (budget, metric), group in expanded.groupby(["budget", "evaluation_metric"], dropna=False):
        summary = summarize_values(group["metric_step_result"].tolist())
        grouped_rows.append(
            {
                "budget": budget,
                "n": int(summary["n"]),
                "metric_step_result_mean": summary["mean"],
                "metric_step_result_std": summary["std"],
                "evaluation_metric": metric,
            }
        )
    return pd.DataFrame(grouped_rows).sort_values(["evaluation_metric", "budget"])


def _metric_summaries(results_df: pd.DataFrame) -> dict[str, object]:
    summaries = {}
    grouped = results_df.groupby("evaluation_metric", dropna=False) if "evaluation_metric" in results_df else []
    for metric, group in grouped:
        summaries[str(metric)] = summarize_values(group["metric_result"].tolist())
    if not summaries and "metric_result" in results_df:
        summaries["all"] = summarize_values(results_df["metric_result"].tolist())
    return summaries


def current_exp_config() -> dict[str, object]:
    from config.config import ExpArgs

    return {
        "task": getattr(getattr(ExpArgs, "task", None), "name", None),
        "eval_metric": getattr(ExpArgs, "eval_metric", None),
        "explained_model_backbone": getattr(ExpArgs, "explained_model_backbone", None),
        "interpreter_model_backbone": getattr(ExpArgs, "interpreter_model_backbone", None),
        "explained_model_name_or_path": getattr(ExpArgs, "explained_model_name_or_path", None),
        "interpreter_model_name_or_path": getattr(ExpArgs, "interpreter_model_name_or_path", None),
        "explained_tokenizer_name_or_path": getattr(ExpArgs, "explained_tokenizer_name_or_path", None),
        "interpreter_tokenizer_name_or_path": getattr(ExpArgs, "interpreter_tokenizer_name_or_path", None),
        "llm_adapter_path": getattr(ExpArgs, "llm_adapter_path", None),
        "local_files_only": bool(getattr(ExpArgs, "local_files_only", False)),
        "trust_remote_code": bool(getattr(ExpArgs, "trust_remote_code", False)),
        "seed": getattr(ExpArgs, "seed", None),
    }


def write_summary_from_results(results_dir, experiment_name: str, started_at: str, ended_at: str) -> dict[str, object]:
    results_dir = Path(results_dir)
    results_csv = results_dir / "results.csv"
    if not results_csv.is_file():
        return {}

    results_df = pd.read_csv(results_csv)
    aopc_curve = _expand_aopc_rows(results_df)
    aopc_curve.to_csv(results_dir / "aopc_curve.csv", index=False)

    baseline_root = Path(__file__).resolve().parents[1]
    summary = {
        "experiment_name": experiment_name,
        "num_examples": int(len(results_df)),
        "run": {
            "started_at": started_at,
            "ended_at": ended_at,
            "git_commit": git_commit(baseline_root),
        },
        "config": current_exp_config(),
        "metrics": _metric_summaries(results_df),
        "aopc_curve_path": "aopc_curve.csv",
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
