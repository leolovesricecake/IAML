import argparse
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.bucket_analysis import summarize_bucket_metrics
from src.io_utils import read_jsonl, write_json
from src.statistics import correlation_report


def aggregate(run_dir):
    run_dir = Path(run_dir)
    examples = [row for row in read_jsonl(run_dir / "per_example.jsonl") if "skip_reason" not in row]
    write_json(run_dir / "aggregate_summary.json", {
        "bucket_metrics": summarize_bucket_metrics(examples, ["interaction_strength", "faithfulness_error"]),
        "correlations": correlation_report(
            [row["interaction_strength"] for row in examples],
            [row["faithfulness_error"] for row in examples],
        ),
    })


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate AML interaction diagnostic outputs")
    parser.add_argument("--run-dir", required=True)
    return parser


if __name__ == "__main__":
    aggregate(build_parser().parse_args().run_dir)
