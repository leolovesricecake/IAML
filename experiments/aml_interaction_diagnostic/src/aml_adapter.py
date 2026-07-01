import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence

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

    def score(self, text: str, words: Sequence[WordUnit], masked_words, mode: str = "delete") -> float:
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


def build_cli_args(
    task: str,
    explained_model_backbone: str,
    interpreter_model_backbone: str,
    metric: str,
    interpreter_checkpoint: Optional[str] = None,
    explained_model_name_or_path: Optional[str] = None,
    interpreter_model_name_or_path: Optional[str] = None,
    explained_tokenizer_name_or_path: Optional[str] = None,
    interpreter_tokenizer_name_or_path: Optional[str] = None,
    llm_adapter_path: Optional[str] = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
):
    """Build a minimal namespace compatible with baseline runs.run_cli.apply_cli_args."""
    return SimpleNamespace(
        task=task,
        explained_model_backbone=explained_model_backbone,
        interpreter_model_backbone=interpreter_model_backbone,
        metric=metric,
        interpreter_checkpoint=interpreter_checkpoint,
        explained_model_name_or_path=explained_model_name_or_path,
        interpreter_model_name_or_path=interpreter_model_name_or_path,
        explained_tokenizer_name_or_path=explained_tokenizer_name_or_path,
        interpreter_tokenizer_name_or_path=interpreter_tokenizer_name_or_path,
        llm_adapter_path=llm_adapter_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


class _AlignmentTokenizer:
    """Tokenizer wrapper matching AML's explained-side body tokenization."""

    def __init__(self, tokenizer, add_special_tokens: bool):
        self._tokenizer = tokenizer
        self._add_special_tokens = add_special_tokens

    def __getattr__(self, item):
        return getattr(self._tokenizer, item)

    def __call__(self, text, **kwargs):
        kwargs["add_special_tokens"] = self._add_special_tokens
        return self._tokenizer(text, **kwargs)


class BaselineAmlAdapter:
    """Lazy adapter for a trained official AML checkpoint."""

    def __init__(
        self,
        baseline_root,
        aml_model=None,
        data_module=None,
        torch_module=None,
        ref_token_id=None,
        explained_model=None,
    ):
        self.baseline_root = baseline_root
        self.aml_model = aml_model
        self.data_module = data_module
        self.torch = torch_module
        self.ref_token_id = ref_token_id
        self.explained_model = explained_model or getattr(aml_model, "explained_model", None)
        self.explained_tokenizer = getattr(data_module, "explained_tokenizer", None)
        self.interpreter_tokenizer = getattr(data_module, "interpreter_tokenizer", None)
        self.alignment_tokenizer = self.explained_tokenizer
        self._target_label_by_text = {}
        self._original_score_by_text = {}

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
        trust_remote_code: bool = False,
        max_samples: Optional[int] = None,
    ):
        """Load official AML components from a trained interpreter checkpoint."""
        if not interpreter_checkpoint:
            raise ValueError("--interpreter-checkpoint is required when --adapter baseline is used.")
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
            hf_from_pretrained_kwargs,
            init_trainable_embeddings,
            load_explained_model,
            load_interpreter_model,
            load_trainable_embeddings,
        )
        from models.model_path_overrides import get_tokenizer_name_or_path
        from runs.run_cli import apply_cli_args
        from utils.utils_functions import get_device, is_use_prompt
        from transformers import AutoTokenizer

        apply_cli_args(
            build_cli_args(
                task=task,
                explained_model_backbone=explained_model_backbone,
                interpreter_model_backbone=interpreter_model_backbone,
                metric=metric,
                interpreter_checkpoint=interpreter_checkpoint,
                explained_model_name_or_path=explained_model_name_or_path,
                interpreter_model_name_or_path=interpreter_model_name_or_path,
                explained_tokenizer_name_or_path=explained_tokenizer_name_or_path,
                interpreter_tokenizer_name_or_path=interpreter_tokenizer_name_or_path,
                llm_adapter_path=llm_adapter_path,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        )
        ExpArgs.fine_tuned_interpreter_model_path = interpreter_checkpoint

        explained_model = load_explained_model()
        trainable_embeddings, label_embedding_index = init_trainable_embeddings()
        load_trainable_embeddings(trainable_embeddings)
        interpreter_model = load_interpreter_model()
        data_module = DataModule(train_sample=ExpArgs.task.train_sample, test_sample=max_samples or ExpArgs.task.test_sample, val_type=ValidationType.TEST)
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
        adapter = cls(
            baseline_root=baseline_root,
            aml_model=aml_model,
            data_module=data_module,
            torch_module=torch,
            ref_token_id=ref_token_id,
            explained_model=explained_model,
        )
        alignment_tokenizer = AutoTokenizer.from_pretrained(
            get_tokenizer_name_or_path(ExpArgs.task, ExpArgs.explained_model_backbone, role="explained"),
            use_fast=True,
            **hf_from_pretrained_kwargs(),
        )
        if not getattr(alignment_tokenizer, "is_fast", False):
            raise ValueError("Baseline diagnostic requires a Hugging Face fast tokenizer for word alignment.")
        if getattr(alignment_tokenizer, "pad_token_id", None) is None and getattr(data_module.explained_tokenizer, "pad_token_id", None) is not None:
            alignment_tokenizer.pad_token = data_module.explained_tokenizer.pad_token
        adapter.alignment_tokenizer = _AlignmentTokenizer(
            alignment_tokenizer,
            add_special_tokens=not is_use_prompt(),
        )
        return adapter

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

    def iter_samples(self, max_samples: Optional[int] = None, split: str = "test"):
        """Yield real AML task samples from the prepared validation/test split."""
        if split != "test":
            raise ValueError("BaselineAmlAdapter currently supports split='test' only.")
        from config.constants import INPUT_TXT, LABELS_NAME

        for index, item in enumerate(self.data_module.val_dataset):
            if max_samples is not None and index >= max_samples:
                break
            item_id = index
            if "id" in item:
                raw_id = item["id"]
                item_id = int(raw_id.item()) if hasattr(raw_id, "item") else raw_id
            label = None
            if LABELS_NAME in item:
                raw_label = item[LABELS_NAME]
                label = int(raw_label.item()) if hasattr(raw_label, "item") else int(raw_label)
            yield {
                "id": str(item_id),
                "text": item[INPUT_TXT],
                "true_label": label,
            }

    def _batch_with_explained_inputs(self, text: str, input_ids, attention_mask):
        from config.constants import EXPLAINED_ATTENTION_MASK_NAME, EXPLAINED_INPUT_IDS_NAME

        batch = self._collate_text(text)
        if isinstance(batch[EXPLAINED_INPUT_IDS_NAME], list):
            batch[EXPLAINED_INPUT_IDS_NAME] = [input_ids]
            batch[EXPLAINED_ATTENTION_MASK_NAME] = [attention_mask]
        else:
            batch[EXPLAINED_INPUT_IDS_NAME] = input_ids.unsqueeze(0)
            batch[EXPLAINED_ATTENTION_MASK_NAME] = attention_mask.unsqueeze(0)
        return batch

    def _explained_tensors(self, text: str):
        from config.constants import EXPLAINED_ATTENTION_MASK_NAME, EXPLAINED_INPUT_IDS_NAME

        batch = self._collate_text(text)
        input_ids = batch[EXPLAINED_INPUT_IDS_NAME][0]
        attention_mask = batch[EXPLAINED_ATTENTION_MASK_NAME][0]
        return input_ids.detach().clone().long(), attention_mask.detach().clone().long()

    def _predict_batch(self, batch):
        from config.config import ExpArgs
        from config.constants import (
            EXPLAINED_ATTENTION_MASK_NAME,
            EXPLAINED_INPUT_IDS_NAME,
            LABEL_PROMPT_ATTENTION_MASK,
            LABEL_PROMPT_INPUT_IDS,
            TASK_PROMPT_ATTENTION_MASK,
            TASK_PROMPT_INPUT_IDS,
        )
        from utils.utils_functions import get_device, merge_prompts, run_model

        device = get_device()
        input_ids, attention_mask = merge_prompts(
            inputs=batch[EXPLAINED_INPUT_IDS_NAME],
            attention_mask=batch[EXPLAINED_ATTENTION_MASK_NAME],
            task_prompt=batch[TASK_PROMPT_INPUT_IDS],
            label_prompt=batch[LABEL_PROMPT_INPUT_IDS],
            task_prompt_attention_mask=batch[TASK_PROMPT_ATTENTION_MASK],
            label_prompt_attention_mask=batch[LABEL_PROMPT_ATTENTION_MASK],
        )
        with self.torch.no_grad():
            logits = run_model(
                model=self.explained_model,
                model_backbone=ExpArgs.explained_model_backbone,
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                is_return_logits=True,
            )
            logits = logits.squeeze(0) if logits.dim() > 1 else logits
            probabilities = self.torch.softmax(logits, dim=-1)
            predicted_label = int(self.torch.argmax(logits, dim=-1).item())
        return probabilities.detach(), predicted_label

    def _target_label(self, text: str) -> int:
        if text not in self._target_label_by_text:
            batch = self._collate_text(text)
            probabilities, predicted_label = self._predict_batch(batch)
            self._target_label_by_text[text] = predicted_label
            self._original_score_by_text[text] = float(probabilities[predicted_label].item())
        return self._target_label_by_text[text]

    def _word_token_positions(self, words: Sequence[WordUnit], word_indices: Iterable[int]) -> set[int]:
        positions = set()
        for word_index in word_indices:
            if 0 <= int(word_index) < len(words):
                positions.update(int(idx) for idx in words[int(word_index)].subword_indices)
        return positions

    def _special_positions(self, input_ids):
        from config.config import ExpArgs
        from utils.utils_functions import get_model_special_tokens

        special_ids = set(get_model_special_tokens(ExpArgs.explained_model_backbone, self.explained_tokenizer))
        return {idx for idx, token_id in enumerate(input_ids.tolist()) if int(token_id) in special_ids}

    def _perturb_input(self, input_ids, attention_mask, words: Sequence[WordUnit], selected_words, mode: str):
        active_positions = [idx for idx, value in enumerate(attention_mask.tolist()) if int(value) == 1]
        selected_positions = self._word_token_positions(words, selected_words)
        special_positions = self._special_positions(input_ids)

        if mode == "delete":
            keep_positions = [
                idx for idx in active_positions if idx not in selected_positions or idx in special_positions
            ]
            return input_ids[keep_positions], attention_mask[keep_positions]
        if mode == "keep":
            keep_positions = [idx for idx in active_positions if idx in selected_positions or idx in special_positions]
            return input_ids[keep_positions], attention_mask[keep_positions]
        if mode == "replace_ref":
            perturbed_ids = input_ids.detach().clone()
            for idx in selected_positions:
                if 0 <= idx < perturbed_ids.shape[-1] and idx not in special_positions:
                    perturbed_ids[idx] = int(self.ref_token_id)
            return perturbed_ids, attention_mask
        raise ValueError(f"Unsupported scoring mode: {mode}")

    def score(self, text: str, words: Sequence[WordUnit], masked_words, mode: str = "delete") -> float:
        """Score target-class probability under AML evaluation masking semantics."""
        if self.explained_model is None or self.data_module is None or self.torch is None:
            raise RuntimeError("BaselineAmlAdapter must be constructed with from_checkpoint().")

        target_label = self._target_label(text)
        input_ids, attention_mask = self._explained_tensors(text)
        perturbed_ids, perturbed_attention = self._perturb_input(input_ids, attention_mask, words, set(masked_words), mode)
        batch = self._batch_with_explained_inputs(text, perturbed_ids, perturbed_attention)
        probabilities, _ = self._predict_batch(batch)
        return float(probabilities[target_label].item())

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
        self._target_label_by_text[text] = predicted_label
        self._original_score_by_text[text] = original_score
        word_attributions = []
        for word in words:
            scores = [token_attr[idx] for idx in word.subword_indices if idx < len(token_attr)]
            word_attributions.append(max(scores) if scores else 0.0)
        return AttributionOutput(
            predicted_label=predicted_label,
            original_target_score=original_score,
            word_attributions=word_attributions,
        )
