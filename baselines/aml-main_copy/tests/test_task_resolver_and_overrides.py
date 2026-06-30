import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    sys.modules["torch"] = torch_stub


class TaskResolverTests(unittest.TestCase):
    def test_get_task_supports_known_aliases(self):
        from runs.runs_utils import get_task

        self.assertEqual(get_task("sst").name, "sst")
        self.assertEqual(get_task("sst2").name, "sst")
        self.assertEqual(get_task("ag_news").name, "agn")
        self.assertEqual(get_task("rotten_tomatoes").name, "rtn")

    def test_get_task_rejects_unknown_task(self):
        from runs.runs_utils import get_task

        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            get_task("not_a_task")


class CliOverrideTests(unittest.TestCase):
    def tearDown(self):
        from config.config import ExpArgs

        ExpArgs.task = None
        ExpArgs.explained_model_backbone = None
        ExpArgs.interpreter_model_backbone = None
        ExpArgs.explained_model_name_or_path = None
        ExpArgs.interpreter_model_name_or_path = None
        ExpArgs.explained_tokenizer_name_or_path = None
        ExpArgs.interpreter_tokenizer_name_or_path = None
        ExpArgs.llm_adapter_path = None
        ExpArgs.local_files_only = False
        ExpArgs.trust_remote_code = False

    def test_cli_parser_sets_model_and_tokenizer_overrides(self):
        from config.config import ExpArgs
        from runs.run_cli import apply_cli_args, parse_args

        args = parse_args(
            [
                "sst2",
                "BERT",
                "ROBERTA",
                "AOPC_COMPREHENSIVENESS",
                "--explained-model-name-or-path",
                "local/explained",
                "--interpreter-model-name-or-path",
                "local/interpreter",
                "--explained-tokenizer-name-or-path",
                "local/explained-tokenizer",
                "--interpreter-tokenizer-name-or-path",
                "local/interpreter-tokenizer",
                "--llm-adapter-path",
                "local/adapter",
                "--local-files-only",
            ]
        )

        apply_cli_args(args)

        self.assertEqual(ExpArgs.task.name, "sst")
        self.assertEqual(ExpArgs.explained_model_backbone, "BERT")
        self.assertEqual(ExpArgs.interpreter_model_backbone, "ROBERTA")
        self.assertEqual(ExpArgs.eval_metric, "AOPC_COMPREHENSIVENESS")
        self.assertEqual(ExpArgs.explained_model_name_or_path, "local/explained")
        self.assertEqual(ExpArgs.interpreter_model_name_or_path, "local/interpreter")
        self.assertEqual(ExpArgs.explained_tokenizer_name_or_path, "local/explained-tokenizer")
        self.assertEqual(ExpArgs.interpreter_tokenizer_name_or_path, "local/interpreter-tokenizer")
        self.assertEqual(ExpArgs.llm_adapter_path, "local/adapter")
        self.assertTrue(ExpArgs.local_files_only)

    def test_cli_aliases_qwen_and_deepseek_to_generic_causal_lm(self):
        from config.config import ExpArgs
        from runs.run_cli import apply_cli_args, parse_args

        apply_cli_args(
            parse_args(
                [
                    "sst2",
                    "qwen",
                    "ROBERTA",
                    "AOPC_COMPREHENSIVENESS",
                    "--explained-model-name-or-path",
                    "/models/Qwen2.5-7B-Instruct",
                    "--trust-remote-code",
                ]
            )
        )

        self.assertEqual(ExpArgs.explained_model_backbone, "CAUSAL_LM")
        self.assertEqual(ExpArgs.interpreter_model_backbone, "ROBERTA")
        self.assertEqual(ExpArgs.explained_model_name_or_path, "/models/Qwen2.5-7B-Instruct")
        self.assertTrue(ExpArgs.trust_remote_code)

        apply_cli_args(
            parse_args(
                [
                    "sst2",
                    "deepseek",
                    "BERT",
                    "AOPC_COMPREHENSIVENESS",
                    "--explained-model-name-or-path",
                    "/models/deepseek-llm-7b",
                ]
            )
        )

        self.assertEqual(ExpArgs.explained_model_backbone, "CAUSAL_LM")
        self.assertEqual(ExpArgs.interpreter_model_backbone, "BERT")

    def test_causal_lm_prompt_scoring_is_independent_of_task_lora_flag(self):
        from config.config import ExpArgs
        from runs.run_cli import apply_cli_args, parse_args
        from utils.utils_functions import is_use_prompt

        apply_cli_args(
            parse_args(
                [
                    "emotions",
                    "CAUSAL_LM",
                    "ROBERTA",
                    "AOPC_COMPREHENSIVENESS",
                    "--explained-model-name-or-path",
                    "/models/Qwen2.5-7B-Instruct",
                ]
            )
        )

        self.assertTrue(ExpArgs.task.is_llm_use_lora)
        self.assertTrue(is_use_prompt())

    def test_experiment_name_prefix_distinguishes_model_overrides(self):
        from config.config import ExpArgs
        from runs.run_cli import apply_cli_args, build_experiment_name_prefix, parse_args

        apply_cli_args(
            parse_args(
                [
                    "sst2",
                    "BERT",
                    "ROBERTA",
                    "AOPC_COMPREHENSIVENESS",
                    "--explained-model-name-or-path",
                    "/models/qwen-classifier",
                    "--interpreter-model-name-or-path",
                    "/models/roberta-aml",
                ]
            )
        )

        prefix = build_experiment_name_prefix()

        self.assertIn("sst_BERT_ROBERTA_AOPC_COMPREHENSIVENESS", prefix)
        self.assertIn("explained-qwen-classifier", prefix)
        self.assertIn("interpreter-roberta-aml", prefix)
        self.assertIn(ExpArgs.task.name, prefix)

    def test_cli_validation_rejects_llama_as_interpreter_backbone(self):
        from runs.run_cli import apply_cli_args, parse_args

        args = parse_args(["sst2", "ROBERTA", "LLAMA", "AOPC_COMPREHENSIVENESS"])

        with self.assertRaisesRegex(ValueError, "interpreter_model_backbone"):
            apply_cli_args(args)

    def test_model_path_overrides_fall_back_to_task_defaults(self):
        from config.config import ExpArgs
        from models.model_path_overrides import (
            get_explained_model_name_or_path,
            get_interpreter_model_name_or_path,
            get_tokenizer_name_or_path,
        )

        task = SimpleNamespace(
            bert_fine_tuned_model="task/bert-finetuned",
            bert_base_model="task/bert-base",
            roberta_fine_tuned_model="task/roberta-finetuned",
            roberta_base_model="task/roberta-base",
            distilbert_fine_tuned_model="task/distilbert-finetuned",
            distilbert_base_model="task/distilbert-base",
            llama_model="task/llama",
            mistral_model="task/mistral",
        )
        ExpArgs.explained_model_backbone = "BERT"
        ExpArgs.interpreter_model_backbone = "BERT"

        self.assertEqual(get_explained_model_name_or_path(task, "BERT"), "task/bert-finetuned")
        self.assertEqual(get_interpreter_model_name_or_path(task, "BERT"), "task/bert-finetuned")
        self.assertEqual(get_tokenizer_name_or_path(task, "BERT", role="explained"), "task/bert-finetuned")

        ExpArgs.explained_model_name_or_path = "override/explained"
        ExpArgs.interpreter_model_name_or_path = "override/interpreter"
        ExpArgs.explained_tokenizer_name_or_path = "override/explained-tokenizer"

        self.assertEqual(get_explained_model_name_or_path(task, "BERT"), "override/explained")
        self.assertEqual(get_interpreter_model_name_or_path(task, "BERT"), "override/interpreter")
        self.assertEqual(get_tokenizer_name_or_path(task, "BERT", role="explained"), "override/explained-tokenizer")
        self.assertEqual(get_tokenizer_name_or_path(task, "BERT", role="interpreter"), "override/interpreter")

    def test_generic_causal_lm_requires_explicit_model_path_and_reuses_it_for_tokenizer(self):
        from config.config import ExpArgs
        from models.model_path_overrides import get_explained_model_name_or_path, get_tokenizer_name_or_path

        task = SimpleNamespace()

        with self.assertRaisesRegex(ValueError, "CAUSAL_LM.*--explained-model-name-or-path"):
            get_explained_model_name_or_path(task, "CAUSAL_LM")

        ExpArgs.explained_model_name_or_path = "/models/Qwen2.5-7B-Instruct"

        self.assertEqual(get_explained_model_name_or_path(task, "CAUSAL_LM"), "/models/Qwen2.5-7B-Instruct")
        self.assertEqual(
            get_tokenizer_name_or_path(task, "CAUSAL_LM", role="explained"),
            "/models/Qwen2.5-7B-Instruct",
        )


if __name__ == "__main__":
    unittest.main()
