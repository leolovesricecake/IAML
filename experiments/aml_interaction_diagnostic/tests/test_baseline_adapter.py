import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


class BaselineAdapterTests(unittest.TestCase):
    def test_build_cli_args_includes_trust_remote_code(self):
        from src.aml_adapter import build_cli_args

        args = build_cli_args(
            task="sst2",
            explained_model_backbone="CAUSAL_LM",
            interpreter_model_backbone="ROBERTA",
            metric="AOPC_COMPREHENSIVENESS",
            interpreter_checkpoint="OUT/PRE_TRAIN/CHECKPOINTS/example",
            explained_model_name_or_path="/models/qwen",
            interpreter_model_name_or_path="/models/roberta",
            explained_tokenizer_name_or_path="/tokenizers/qwen",
            interpreter_tokenizer_name_or_path="/tokenizers/roberta",
            llm_adapter_path="/models/qwen-adapter",
            local_files_only=True,
            trust_remote_code=True,
        )

        self.assertEqual(args.task, "sst2")
        self.assertEqual(args.explained_model_backbone, "CAUSAL_LM")
        self.assertEqual(args.interpreter_model_backbone, "ROBERTA")
        self.assertTrue(args.local_files_only)
        self.assertTrue(args.trust_remote_code)


if __name__ == "__main__":
    unittest.main()
