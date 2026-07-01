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

    def test_explain_uses_paml_inference_path_instead_of_training_forward(self):
        from src.aml_adapter import BaselineAmlAdapter
        from src.word_units import WordUnit

        class FakeAttribution:
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [0.1, 0.7, 0.2]

        class FakeProb:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeProbabilities:
            def __getitem__(self, index):
                return FakeProb(0.8 if index == 1 else 0.2)

            def detach(self):
                return self

        class FakeAmlModel:
            def __init__(self):
                self.inference_called = False

            def forward(self, _batch):
                raise AssertionError("diagnostic explain should not call the training forward path")

            def forwad_paml_inference(self, _batch, is_evaluate):
                self.inference_called = True
                self.is_evaluate = is_evaluate
                return FakeAttribution(), None, 0.0

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeTorch:
            no_grad = FakeNoGrad

        fake_model = FakeAmlModel()
        adapter = BaselineAmlAdapter(
            baseline_root=EXPERIMENT_ROOT,
            aml_model=fake_model,
            data_module=object(),
            torch_module=FakeTorch,
        )
        adapter._collate_text = lambda _text: {"batch": True}
        adapter._predict_batch = lambda _batch: (FakeProbabilities(), 1)

        output = adapter.explain(
            "good film",
            [
                WordUnit(index=0, text="good", subword_indices=[0, 1]),
                WordUnit(index=1, text="film", subword_indices=[2]),
            ],
        )

        self.assertTrue(fake_model.inference_called)
        self.assertFalse(fake_model.is_evaluate)
        self.assertEqual(output.predicted_label, 1)
        self.assertEqual(output.original_target_score, 0.8)
        self.assertEqual(output.word_attributions, [0.7, 0.2])


if __name__ == "__main__":
    unittest.main()
