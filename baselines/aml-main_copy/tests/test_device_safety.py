import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]


class DeviceSafetyTests(unittest.TestCase):
    def test_critical_runtime_files_do_not_hardcode_cuda_calls(self):
        critical_files = [
            BASELINE_ROOT / "evaluations" / "metrics" / "metrics_utils.py",
            BASELINE_ROOT / "models" / "aml_model.py",
            BASELINE_ROOT / "models" / "train_models_utils.py",
            BASELINE_ROOT / "utils" / "utils_functions.py",
        ]

        offenders = []
        for path in critical_files:
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if ".cuda(" in line or ".cuda()" in line:
                    offenders.append(f"{path.name}:{line_number}:{stripped}")

        self.assertEqual(offenders, [])

    def test_llm_label_vocab_indices_follow_logits_device(self):
        source = (BASELINE_ROOT / "utils" / "utils_functions.py").read_text(encoding="utf-8")

        self.assertIn("ExpArgs.label_vocab_tokens.to(logits.device)", source)

    def test_metric_special_tokens_follow_input_device(self):
        source = (BASELINE_ROOT / "evaluations" / "metrics" / "metrics_utils.py").read_text(encoding="utf-8")

        self.assertIn("self.special_tokens.to(input_ids.device)", source)


if __name__ == "__main__":
    unittest.main()
