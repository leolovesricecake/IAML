import argparse
import itertools
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from scripts.run_diagnostic import SimpleOffsetTokenizer, _default_samples
from src.aml_adapter import MockAmlAdapter
from src.candidate_coverage import recall_report
from src.candidate_graph import build_candidate_edges
from src.interaction_teacher import compute_interactions
from src.io_utils import ensure_dir, write_json
from src.model_adapter import ProbabilityScorer
from src.tokenizer_alignment import align_text_to_tokens


def run(args):
    tokenizer = SimpleOffsetTokenizer()
    adapter = MockAmlAdapter()
    reports = []
    for sample in _default_samples()[: args.num_samples]:
        alignment = align_text_to_tokens(tokenizer, sample["text"])
        if alignment.skipped or len(alignment.words) > args.max_words:
            continue
        all_edges = [
            type("Edge", (), {"i": i, "j": j, "edge_type": "all_pairs"})()
            for i, j in itertools.combinations(range(len(alignment.words)), 2)
        ]
        scorer = ProbabilityScorer(adapter, sample["text"], alignment.words)
        all_scores = compute_interactions(all_edges, scorer).edge_scores
        ranked_pairs = [
            (score.i, score.j)
            for score in sorted(all_scores, key=lambda item: item.absolute_interaction_score, reverse=True)
        ]
        candidates = {(edge.i, edge.j) for edge in build_candidate_edges(alignment.words)}
        reports.append({"id": sample["id"], **recall_report(ranked_pairs, candidates)})
    output_dir = ensure_dir(args.output_dir)
    write_json(output_dir / "candidate_coverage.json", {"examples": reports})
    print(output_dir / "candidate_coverage.json")


def build_parser():
    parser = argparse.ArgumentParser(description="Run all-pairs candidate coverage diagnostic")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=str(EXPERIMENT_ROOT / "outputs" / "candidate_coverage_smoke"))
    parser.add_argument("--max-words", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=2)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
