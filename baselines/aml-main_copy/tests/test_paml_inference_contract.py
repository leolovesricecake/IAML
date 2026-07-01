import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]


class PamlInferenceContractTests(unittest.TestCase):
    def test_paml_inference_duration_is_available_without_evaluation(self):
        source = (BASELINE_ROOT / "models" / "aml_model.py").read_text(encoding="utf-8")

        self.assertIn("duration = time.time() - begin", source)


if __name__ == "__main__":
    unittest.main()
