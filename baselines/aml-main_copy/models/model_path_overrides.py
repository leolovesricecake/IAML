from config.config import ExpArgs
from config.constants import HF_CACHE
from config.types_enums import ModelBackboneTypes


def hf_from_pretrained_kwargs() -> dict:
    """Common Hugging Face from_pretrained kwargs controlled by CLI overrides."""
    return {
        "cache_dir": HF_CACHE,
        "local_files_only": bool(getattr(ExpArgs, "local_files_only", False)),
        "trust_remote_code": bool(getattr(ExpArgs, "trust_remote_code", False)),
    }


def _is_encoder_only(backbone: str) -> bool:
    return backbone in {
        ModelBackboneTypes.BERT.value,
        ModelBackboneTypes.ROBERTA.value,
        ModelBackboneTypes.DISTILBERT.value,
    }


def _is_generic_explained_backbone(backbone: str) -> bool:
    return backbone == ModelBackboneTypes.CAUSAL_LM.value


def _fine_tuned_attr(backbone: str) -> str:
    mapping = {
        ModelBackboneTypes.BERT.value: "bert_fine_tuned_model",
        ModelBackboneTypes.ROBERTA.value: "roberta_fine_tuned_model",
        ModelBackboneTypes.DISTILBERT.value: "distilbert_fine_tuned_model",
        ModelBackboneTypes.LLAMA.value: "llama_model",
        ModelBackboneTypes.MISTRAL.value: "mistral_model",
    }
    if backbone not in mapping:
        raise ValueError(
            f"{backbone} requires --explained-model-name-or-path because no task default path is defined"
        )
    return mapping[backbone]


def _base_attr(backbone: str) -> str:
    mapping = {
        ModelBackboneTypes.BERT.value: "bert_base_model",
        ModelBackboneTypes.ROBERTA.value: "roberta_base_model",
        ModelBackboneTypes.DISTILBERT.value: "distilbert_base_model",
    }
    return mapping[backbone]


def _non_empty(value):
    return value if value not in (None, "") else None


def get_explained_model_name_or_path(task, backbone: str) -> str:
    """Resolve the explained model path, preferring explicit CLI override."""
    explicit = _non_empty(getattr(ExpArgs, "explained_model_name_or_path", None))
    if explicit:
        return explicit
    if _is_generic_explained_backbone(backbone):
        raise ValueError(
            f"{backbone} requires --explained-model-name-or-path because no task default path is defined"
        )
    return getattr(task, _fine_tuned_attr(backbone))


def get_interpreter_model_name_or_path(task, backbone: str) -> str:
    """Resolve interpreter model path while preserving AML's original fallback rules."""
    explicit = _non_empty(getattr(ExpArgs, "fine_tuned_interpreter_model_path", None))
    if explicit:
        return explicit

    explicit = _non_empty(getattr(ExpArgs, "interpreter_model_name_or_path", None))
    if explicit:
        return explicit

    if _is_encoder_only(getattr(ExpArgs, "explained_model_backbone", "")):
        return getattr(task, _fine_tuned_attr(backbone))
    return getattr(task, _base_attr(backbone))


def get_tokenizer_name_or_path(task, backbone: str, role: str = "explained") -> str:
    """Resolve tokenizer path for the explained or interpreter tokenizer."""
    if role == "explained":
        explicit = (
            _non_empty(getattr(ExpArgs, "explained_tokenizer_name_or_path", None))
            or _non_empty(getattr(ExpArgs, "explained_model_name_or_path", None))
        )
        if explicit:
            return explicit
        if _is_generic_explained_backbone(backbone):
            raise ValueError(
                f"{backbone} requires --explained-model-name-or-path or --explained-tokenizer-name-or-path"
            )
        return getattr(task, _fine_tuned_attr(backbone))
    if role == "interpreter":
        return (
            _non_empty(getattr(ExpArgs, "interpreter_tokenizer_name_or_path", None))
            or _non_empty(getattr(ExpArgs, "interpreter_model_name_or_path", None))
            or getattr(task, _fine_tuned_attr(backbone))
        )
    raise ValueError(f"Unsupported tokenizer role: {role}")


def get_llm_adapter_path(task, backbone: str):
    """Resolve an optional LLM adapter path, preferring the CLI override."""
    explicit = _non_empty(getattr(ExpArgs, "llm_adapter_path", None))
    if explicit:
        return explicit
    if backbone == ModelBackboneTypes.LLAMA.value:
        return getattr(task, "llama_adapter", None)
    if backbone == ModelBackboneTypes.MISTRAL.value:
        return getattr(task, "mistral_adapter", None)
    if backbone == ModelBackboneTypes.CAUSAL_LM.value:
        return None
    return None
