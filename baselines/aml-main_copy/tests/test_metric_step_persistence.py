import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]


class MetricStepPersistenceTests(unittest.TestCase):
    def test_metrics_transform_results_persists_step_values(self):
        source = (BASELINE_ROOT / "evaluations" / "metrics" / "metrics.py").read_text(encoding="utf-8")

        self.assertIn("metric_steps_result = results_steps", source)
        self.assertNotIn("metric_steps_result = None", source)


if __name__ == "__main__":
    unittest.main()
