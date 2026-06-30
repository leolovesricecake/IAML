import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

from src.word_units import WordUnit


@dataclass
class AttributionOutput:
    """Per-example AML attribution output used by the diagnostic pipeline."""

    predicted_label: int
    original_target_score: float
    word_attributions: List[float]


class MockAmlAdapter:
    """Small deterministic adapter for CPU smoke tests without model weights."""

    def explain(self, text: str, words: Sequence[WordUnit]) -> AttributionOutput:
        attributions = []
        for word in words:
            normalized = word.text.lower().strip(".,!?;:\"'")
            if normalized in {"not", "never", "no"}:
                attributions.append(0.95)
            elif normalized in {"good", "great", "bad", "terrible"}:
                attributions.append(0.85)
            else:
                attributions.append(0.15)
        return AttributionOutput(predicted_label=1, original_target_score=self.score(text, words, frozenset()), word_attributions=attributions)

    def score(self, text: str, words: Sequence[WordUnit], masked_words) -> float:
        """Return a mock target probability after masking natural-word indices."""
        masked = set(masked_words)
        score = 0.9
        normalized_words = [word.text.lower().strip(".,!?;:\"'") for word in words]
        for idx in masked:
            word = normalized_words[idx]
            if word in {"not", "never", "no"}:
                score -= 0.18
            elif word in {"good", "great", "bad", "terrible"}:
                score -= 0.14
            else:
                score -= 0.03
        for left, right in (("not", "good"), ("not", "bad"), ("never", "good")):
            if left in normalized_words and right in normalized_words:
                i = normalized_words.index(left)
                j = normalized_words.index(right)
                if i in masked and j in masked:
                    score -= 0.18
        return max(0.01, min(0.99, score))


class BaselineAmlAdapter:
    """Lazy adapter for a trained official AML checkpoint."""

    def __init__(self, baseline_root, aml_model=None, data_module=None, torch_module=None):
        self.baseline_root = baseline_root
        self.aml_model = aml_model
        self.data_module = data_module
        self.torch = torch_module

    @classmethod
    def from_checkpoint(
        cls,
        baseline_root,
        task: str,
        explained_model_backbone: str,
        interpreter_model_backbone: str,
        metric: str,
        interpreter_checkpoint: str,
        explained_model_name_or_path: Optional[str] = None,
        interpreter_model_name_or_path: Optional[str] = None,
        explained_tokenizer_name_or_path: Optional[str] = None,
        interpreter_tokenizer_name_or_path: Optional[str] = None,
        llm_adapter_path: Optional[str] = None,
        local_files_only: bool = False,
    ):
        """Load official AML components from a trained interpreter checkpoint."""
        baseline_root = Path(baseline_root).resolve()
        if str(baseline_root) not in sys.path:
            sys.path.insert(0, str(baseline_root))

        import torch
        from config.config import ExpArgs
        from config.types_enums import ValidationType
        from main.data_module import DataModule
        from models.aml_model_fine_tune import AmlModelFineTune
        from models.train_models_utils import (
            get_explained_ref_token_name,
            init_trainable_embeddings,
            load_explained_model,
            load_interpreter_model,
            load_trainable_embeddings,
        )
        from runs.run_cli import apply_cli_args
        from utils.utils_functions import get_device

        apply_cli_args(
            SimpleNamespace(
                task=task,
                explained_model_backbone=explained_model_backbone,
                interpreter_model_backbone=interpreter_model_backbone,
                metric=metric,
                explained_model_name_or_path=explained_model_name_or_path,
                interpreter_model_name_or_path=interpreter_model_name_or_path,
                explained_tokenizer_name_or_path=explained_tokenizer_name_or_path,
                interpreter_tokenizer_name_or_path=interpreter_tokenizer_name_or_path,
                llm_adapter_path=llm_adapter_path,
                local_files_only=local_files_only,
            )
        )
        ExpArgs.fine_tuned_interpreter_model_path = interpreter_checkpoint

        explained_model = load_explained_model()
        trainable_embeddings, label_embedding_index = init_trainable_embeddings()
        load_trainable_embeddings(trainable_embeddings)
        interpreter_model = load_interpreter_model()
        data_module = DataModule(train_sample=1, test_sample=1, val_type=ValidationType.TEST)
        ref_token_id = get_explained_ref_token_name(data_module.explained_tokenizer)

        aml_model = AmlModelFineTune(
            explained_model=explained_model,
            interpreter_model=interpreter_model,
            explained_tokenizer=data_module.explained_tokenizer,
            interpreter_tokenizer=data_module.interpreter_tokenizer,
            total_training_steps=0,
            experiment_path="",
            checkpoints_path="",
            warmup_steps=0,
            trainable_embeddings=trainable_embeddings,
            label_embedding_index=label_embedding_index,
            ref_token_id=ref_token_id,
        ).to(get_device())
        for param in aml_model.parameters():
            param.requires_grad = False
        aml_model.eval()
        return cls(baseline_root=baseline_root, aml_model=aml_model, data_module=data_module, torch_module=torch)

    def _collate_text(self, text: str):
        from config.constants import EXPLAINED_ATTENTION_MASK_NAME, EXPLAINED_INPUT_IDS_NAME, INPUT_TXT
        from config.constants import INTERPRETER_ATTENTION_MASK_NAME, INTERPRETER_INPUT_IDS_NAME

        tokenized = self.data_module.tokenize({self.data_module.dataset_column_text: text})
        for key in [
            EXPLAINED_INPUT_IDS_NAME,
            EXPLAINED_ATTENTION_MASK_NAME,
            INTERPRETER_INPUT_IDS_NAME,
            INTERPRETER_ATTENTION_MASK_NAME,
        ]:
            tokenized[key] = self.torch.tensor(tokenized[key]).long()
        tokenized[INPUT_TXT] = text
        return self.data_module.collate_fn([tokenized])

    def explain(self, text: str, words: Sequence[WordUnit]) -> AttributionOutput:
        """Run official AML inference and aggregate token attributions to words with max pooling."""
        if self.aml_model is None or self.data_module is None or self.torch is None:
            raise RuntimeError("BaselineAmlAdapter must be constructed with from_checkpoint().")
        batch = self._collate_text(text)
        with self.torch.no_grad():
            output = self.aml_model.forward(batch)
            probabilities = self.torch.softmax(output.explained_model_predicted_logits, dim=1)
            predicted_label = int(output.explained_model_predicted_class.squeeze().item())
            original_score = float(probabilities[0, predicted_label].item())
            token_attr = output.tokens_attr[0].detach().cpu().tolist()
        word_attributions = []
        for word in words:
            scores = [token_attr[idx] for idx in word.subword_indices if idx < len(token_attr)]
            word_attributions.append(max(scores) if scores else 0.0)
        return AttributionOutput(
            predicted_label=predicted_label,
            original_target_score=original_score,
            word_attributions=word_attributions,
        )
