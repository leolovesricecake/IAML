import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


class AnalysisPrimitiveTests(unittest.TestCase):
    def test_config_loader_handles_simple_yaml_without_pyyaml(self):
        from scripts.run_diagnostic import _load_config

        config = _load_config(EXPERIMENT_ROOT / "configs" / "diagnostic_sst2.yaml")

        self.assertEqual(config["dataset"], "sst2")
        self.assertEqual(config["interaction_strength"]["topk"], 3)
        self.assertEqual(config["budgets"], [1, 5, 10, 20, 50])

    def test_interaction_strengths_include_max_mean_topk_and_energy(self):
        from src.bucket_analysis import compute_interaction_strengths

        strengths = compute_interaction_strengths([1.0, -3.0, 2.0], topk=2)

        self.assertEqual(strengths["max_abs"], 3.0)
        self.assertEqual(strengths["mean_topk_abs"], 2.5)
        self.assertAlmostEqual(strengths["energy"], (1.0 + 9.0 + 4.0) / 3.0)

    def test_quantile_bucket_assignments_are_low_medium_high(self):
        from src.bucket_analysis import assign_quantile_buckets

        buckets = assign_quantile_buckets([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        self.assertEqual(buckets[0], "low")
        self.assertEqual(buckets[2], "medium")
        self.assertEqual(buckets[-1], "high")

    def test_candidate_recall_at_k(self):
        from src.candidate_coverage import recall_at_k

        all_pairs_ranked = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        candidate_edges = {(1, 2), (4, 5)}

        self.assertEqual(recall_at_k(all_pairs_ranked, candidate_edges, 1), 0.0)
        self.assertEqual(recall_at_k(all_pairs_ranked, candidate_edges, 3), 1 / 3)
        self.assertEqual(recall_at_k(all_pairs_ranked, candidate_edges, 5), 2 / 5)

    def test_effect_sizes_have_stable_direction(self):
        from src.statistics import cliffs_delta, standardized_mean_difference

        high = [4.0, 5.0, 6.0]
        low = [1.0, 2.0, 3.0]

        self.assertEqual(cliffs_delta(high, low), 1.0)
        self.assertGreater(standardized_mean_difference(high, low), 0.0)

    def test_config_topk_is_used_when_cli_does_not_override(self):
        from scripts.run_diagnostic import _resolve_interaction_topk

        config = {"interaction_strength": {"topk": 5}}

        self.assertEqual(_resolve_interaction_topk(SimpleNamespace(interaction_topk=None), config), 5)
        self.assertEqual(_resolve_interaction_topk(SimpleNamespace(interaction_topk=3), config), 3)

    def test_mock_diagnostic_writes_summary_and_aopc_curve_outputs(self):
        from scripts.run_diagnostic import build_parser, run

        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "--adapter",
                    "mock",
                    "--max-samples",
                    "2",
                    "--disable-dependency",
                    "--output-dir",
                    tmpdir,
                ]
            )
            run(args)
            output_dir = Path(tmpdir)

            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "per_budget_metrics.csv").is_file())
            self.assertTrue((output_dir / "aopc_curve.csv").is_file())

            aopc_header = (output_dir / "aopc_curve.csv").read_text(encoding="utf-8").splitlines()[0]
            per_budget_header = (output_dir / "per_budget_metrics.csv").read_text(encoding="utf-8").splitlines()[0]

            self.assertIn("budget", aopc_header)
            self.assertIn("comprehensiveness_mean", aopc_header)
            self.assertIn("sufficiency_error_mean", aopc_header)
            self.assertIn("log_odds_mean", aopc_header)
            self.assertIn("id", per_budget_header)
            self.assertIn("log_odds", per_budget_header)

    def test_summary_writer_handles_non_git_repositories(self):
        from src.summary_utils import git_commit

        self.assertIsInstance(git_commit(Path("Z:/definitely/not/a/git/repo")), str)


if __name__ == "__main__":
    unittest.main()
