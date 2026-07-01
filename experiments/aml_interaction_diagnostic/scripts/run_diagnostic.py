import argparse
import csv
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.aml_adapter import BaselineAmlAdapter, MockAmlAdapter
from src.bucket_analysis import assign_quantile_buckets, compute_interaction_strengths, summarize_bucket_metrics
from src.candidate_graph import build_candidate_edges, dependency_edges_from_spacy
from src.casebook import build_casebook
from src.faithfulness_metrics import DEFAULT_BUDGETS, evaluate_faithfulness_from_scores
from src.interaction_teacher import compute_interactions
from src.io_utils import ensure_dir, write_json, write_jsonl
from src.model_adapter import ProbabilityScorer
from src.plotting import write_binned_trend_csv
from src.statistics import cliffs_delta, correlation_report, standardized_mean_difference
from src.summary_utils import utc_now_iso, write_csv, write_diagnostic_outputs
from src.tokenizer_alignment import align_text_to_tokens


class SimpleOffsetTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": [self.cls_token_id] + [1000 + idx for idx, _ in enumerate(matches)] + [self.sep_token_id],
            "attention_mask": [1] * (len(matches) + 2),
            "offset_mapping": [(0, 0)] + [(m.start(), m.end()) for m in matches] + [(0, 0)],
        }


def _default_samples():
    return [
        {"id": "mock-0", "text": "The movie is not good.", "true_label": 0},
        {"id": "mock-1", "text": "A great and warm film.", "true_label": 1},
        {"id": "mock-2", "text": "Never good but not terrible.", "true_label": 0},
    ]


def _load_config(path):
    if path is None:
        return {}
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_simple_yaml_value(value):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_simple_yaml_value(item) for item in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml(path):
    config = {}
    current_section = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            key, _, value = raw_line.strip().partition(":")
            if indent == 0 and value.strip() == "":
                config[key] = {}
                current_section = key
            elif indent == 0:
                config[key] = _parse_simple_yaml_value(value)
                current_section = None
            elif current_section is not None:
                config[current_section][key] = _parse_simple_yaml_value(value)
    return config


def _load_spacy(disable_dependency, model_name):
    if disable_dependency:
        return None
    try:
        import spacy
        return spacy.load(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load spaCy model '{model_name}'. Install it or pass --disable-dependency."
        ) from exc


def _run_dir(args, config):
    if args.output_dir:
        return ensure_dir(args.output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_name = getattr(args, "adapter", "mock")
    task = getattr(args, "task", config.get("dataset", "mock"))
    checkpoint = Path(getattr(args, "interpreter_checkpoint", None) or "mock_checkpoint").name
    explained = Path(getattr(args, "explained_model_name_or_path", None) or "mock_model").name
    return ensure_dir(EXPERIMENT_ROOT / "outputs" / adapter_name / str(task) / explained / checkpoint / run_id)


def _resolve_interaction_topk(args, config):
    if args.interaction_topk is not None:
        return args.interaction_topk
    interaction_config = config.get("interaction_strength", {}) if isinstance(config, dict) else {}
    return int(interaction_config.get("topk", 3))


def _build_adapter_and_samples(args):
    if args.adapter == "mock":
        adapter = MockAmlAdapter()
        return adapter, SimpleOffsetTokenizer(), _default_samples()[: args.max_samples]

    adapter = BaselineAmlAdapter.from_checkpoint(
        baseline_root=args.baseline_root,
        task=args.task,
        explained_model_backbone=args.explained_model_backbone,
        interpreter_model_backbone=args.interpreter_model_backbone,
        metric=args.metric,
        interpreter_checkpoint=args.interpreter_checkpoint,
        explained_model_name_or_path=args.explained_model_name_or_path,
        interpreter_model_name_or_path=args.interpreter_model_name_or_path,
        explained_tokenizer_name_or_path=args.explained_tokenizer_name_or_path,
        interpreter_tokenizer_name_or_path=args.interpreter_tokenizer_name_or_path,
        llm_adapter_path=args.llm_adapter_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        max_samples=args.max_samples,
    )
    return adapter, adapter.alignment_tokenizer, list(adapter.iter_samples(max_samples=args.max_samples))


def _metadata(args, config, topk, num_edges):
    return {
        "adapter": args.adapter,
        "task": args.task,
        "score_type": "target_probability",
        "mask_operator": "aml_hard_deletion_eval_protocol",
        "log_odds_operator": "aml_ref_token_replacement",
        "dependency_parser": None if args.disable_dependency else f"spacy:{args.spacy_model}",
        "word_attribution_aggregation": "max_subword_attribution",
        "interaction_strength": {"primary": "mean_topk_abs", "topk": topk},
        "explained_model_backbone": args.explained_model_backbone,
        "interpreter_model_backbone": args.interpreter_model_backbone,
        "metric": args.metric,
        "interpreter_checkpoint": args.interpreter_checkpoint,
        "explained_model_name_or_path": args.explained_model_name_or_path,
        "interpreter_model_name_or_path": args.interpreter_model_name_or_path,
        "explained_tokenizer_name_or_path": args.explained_tokenizer_name_or_path,
        "interpreter_tokenizer_name_or_path": args.interpreter_tokenizer_name_or_path,
        "llm_adapter_path": args.llm_adapter_path,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "num_edges": num_edges,
        "config": config,
    }


def run(args):
    started_at = utc_now_iso()
    config = _load_config(args.config)
    run_dir = _run_dir(args, config)
    topk = _resolve_interaction_topk(args, config)
    adapter, tokenizer, samples = _build_adapter_and_samples(args)
    nlp = _load_spacy(args.disable_dependency, args.spacy_model)

    per_examples = []
    per_edges = []
    per_budget_rows = []
    for sample in samples:
        alignment = align_text_to_tokens(tokenizer, sample["text"])
        if alignment.skipped:
            per_examples.append({"id": sample["id"], "text": sample["text"], "skip_reason": alignment.skip_reason})
            continue

        dependency_edges = dependency_edges_from_spacy(sample["text"], nlp) if nlp else []
        candidate_edges = build_candidate_edges(alignment.words, dependency_edges=dependency_edges)
        attribution = adapter.explain(sample["text"], alignment.words)
        scorer = ProbabilityScorer(adapter, sample["text"], alignment.words)
        interaction_result = compute_interactions(candidate_edges, scorer)
        strengths = compute_interaction_strengths(
            [edge.interaction_score for edge in interaction_result.edge_scores], topk=topk
        )
        keep_scorer = ProbabilityScorer(adapter, sample["text"], alignment.words, mode="keep")
        replace_scorer = ProbabilityScorer(adapter, sample["text"], alignment.words, mode="replace_ref")
        faithfulness = evaluate_faithfulness_from_scores(
            original_score=attribution.original_target_score,
            word_attributions=attribution.word_attributions,
            delete_scorer=lambda selected: scorer(frozenset(selected)),
            keep_scorer=lambda selected: keep_scorer(frozenset(selected)),
            replace_scorer=lambda selected: replace_scorer(frozenset(selected)),
            budgets=DEFAULT_BUDGETS,
        )
        for budget_row in faithfulness["per_budget"]:
            per_budget_rows.append(
                {
                    "id": sample["id"],
                    "budget": budget_row["budget"],
                    "selected_word_indices": " ".join(str(idx) for idx in budget_row["selected_word_indices"]),
                    "comprehensiveness": budget_row["comprehensiveness"],
                    "sufficiency_error": budget_row["sufficiency_error"],
                    "log_odds": budget_row["log_odds"],
                }
            )

        edge_rows = []
        for edge_score in interaction_result.edge_scores:
            row = {"id": sample["id"], **asdict(edge_score)}
            edge_rows.append(row)
            per_edges.append(row)

        top_abs = sorted(edge_rows, key=lambda row: row["absolute_interaction_score"], reverse=True)
        top_signed = sorted(edge_rows, key=lambda row: row["interaction_score"], reverse=True)
        per_examples.append(
            {
                "id": sample["id"],
                "text": sample["text"],
                "true_label": sample["true_label"],
                "predicted_label": attribution.predicted_label,
                "original_target_score": attribution.original_target_score,
                "words": [word.text for word in alignment.words],
                "word_units": [asdict(word) for word in alignment.words],
                "word_attributions": attribution.word_attributions,
                "top_attributed_words": [
                    {"word": alignment.words[idx].text, "score": attribution.word_attributions[idx]}
                    for idx in sorted(range(len(attribution.word_attributions)), key=lambda i: attribution.word_attributions[i], reverse=True)[:10]
                ],
                "candidate_edges": [asdict(edge) for edge in candidate_edges],
                "top_interactions_signed": top_signed[:10],
                "top_interactions_absolute": top_abs[:10],
                "interaction_strength": strengths["mean_topk_abs"],
                "interaction_strengths": strengths,
                "deletion_metrics": {"comprehensiveness_aopc": faithfulness["comprehensiveness_aopc"]},
                "sufficiency_metrics": {"sufficiency_aopc": faithfulness["sufficiency_aopc"]},
                "log_odds_metrics": {"log_odds_aopc": faithfulness["log_odds_aopc"]},
                "comprehensiveness_metrics": {"per_budget": faithfulness["per_budget"]},
                "faithfulness_error": faithfulness["faithfulness_error"],
            }
        )

    valid_examples = [row for row in per_examples if "skip_reason" not in row]
    buckets = assign_quantile_buckets([row["interaction_strength"] for row in valid_examples])
    for row, bucket in zip(valid_examples, buckets):
        row["bucket"] = bucket
    bucket_by_id = {row["id"]: row.get("bucket") for row in valid_examples}
    for row in per_budget_rows:
        row["bucket"] = bucket_by_id.get(row["id"])

    write_jsonl(run_dir / "per_example.jsonl", per_examples)
    write_jsonl(run_dir / "per_edge.jsonl", per_edges)
    write_csv(
        run_dir / "per_budget_metrics.csv",
        per_budget_rows,
        fieldnames=[
            "id",
            "bucket",
            "budget",
            "selected_word_indices",
            "comprehensiveness",
            "sufficiency_error",
            "log_odds",
        ],
    )
    metadata = _metadata(args, config, topk, len(per_edges))
    write_json(run_dir / "metadata.json", metadata)
    strengths = [row["interaction_strength"] for row in valid_examples]
    errors = [row["faithfulness_error"] for row in valid_examples]
    write_json(run_dir / "correlations.json", correlation_report(strengths, errors))
    high = [row["faithfulness_error"] for row in valid_examples if row.get("bucket") == "high"]
    low = [row["faithfulness_error"] for row in valid_examples if row.get("bucket") == "low"]
    high_vs_low = {
        "cliffs_delta": cliffs_delta(high, low),
        "standardized_mean_difference": standardized_mean_difference(high, low),
    }
    ended_at = utc_now_iso()
    write_diagnostic_outputs(
        run_dir=run_dir,
        per_budget_rows=per_budget_rows,
        per_examples=per_examples,
        metadata=metadata,
        high_vs_low=high_vs_low,
        repo_root=EXPERIMENT_ROOT.parents[1],
        started_at=started_at,
        ended_at=ended_at,
    )
    bucket_rows = summarize_bucket_metrics(valid_examples, ["interaction_strength", "faithfulness_error"])
    with (run_dir / "bucket_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in bucket_rows for key in row.keys()}) if bucket_rows else ["bucket", "n"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bucket_rows)
    write_json(run_dir / "candidate_coverage.json", {"status": "not_run_in_main_diagnostic"})
    ensure_dir(run_dir / "plots")
    write_binned_trend_csv(run_dir / "plots" / "binned_trend.csv", strengths, errors)
    build_casebook(run_dir)
    print(run_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="Run AML interaction diagnostic")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--adapter", choices=["mock", "baseline"], default="mock")
    parser.add_argument("--baseline-root", default=str(EXPERIMENT_ROOT.parents[1] / "baselines" / "aml-main_copy"))
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--explained-model-backbone", default="ROBERTA")
    parser.add_argument("--interpreter-model-backbone", default="ROBERTA")
    parser.add_argument("--metric", default="AOPC_COMPREHENSIVENESS")
    parser.add_argument("--interpreter-checkpoint", default=None)
    parser.add_argument("--explained-model-name-or-path", default=None)
    parser.add_argument("--interpreter-model-name-or-path", default=None)
    parser.add_argument("--explained-tokenizer-name-or-path", default=None)
    parser.add_argument("--interpreter-tokenizer-name-or-path", default=None)
    parser.add_argument("--llm-adapter-path", default=None)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--disable-dependency", action="store_true", default=False)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--interaction-topk", type=int, default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
