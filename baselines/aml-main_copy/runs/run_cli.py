import argparse
import hashlib
import re
from pathlib import Path


_SUPPORTED_INTERPRETER_BACKBONES = {"BERT", "ROBERTA", "DISTILBERT"}
_SUPPORTED_EXPLAINED_BACKBONES = {"BERT", "ROBERTA", "DISTILBERT", "LLAMA", "MISTRAL", "CAUSAL_LM"}
_EXPLAINED_BACKBONE_ALIASES = {
    "AUTO_CAUSAL_LM": "CAUSAL_LM",
    "QWEN": "CAUSAL_LM",
    "QWEN2": "CAUSAL_LM",
    "QWEN3": "CAUSAL_LM",
    "DEEPSEEK": "CAUSAL_LM",
    "DEEPSEEK_V2": "CAUSAL_LM",
    "DEEPSEEK_V3": "CAUSAL_LM",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the AML CLI parser while preserving the original positional API."""
    parser = argparse.ArgumentParser(description="Run Attributive Masking Learning")
    parser.add_argument("task", type=str, help="Task alias, e.g. sst2, imdb, emotions")
    parser.add_argument("explained_model_backbone", type=str, help="Explained model backbone")
    parser.add_argument("interpreter_model_backbone", type=str, help="Interpreter model backbone")
    parser.add_argument("metric", type=str, help="AML evaluation metric")
    parser.add_argument("--explained-model-name-or-path", default=None)
    parser.add_argument("--interpreter-model-name-or-path", default=None)
    parser.add_argument("--explained-tokenizer-name-or-path", default=None)
    parser.add_argument("--interpreter-tokenizer-name-or-path", default=None)
    parser.add_argument("--llm-adapter-path", default=None)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse AML command line arguments."""
    return build_parser().parse_args(argv)


def _is_llm_backbone(backbone: str) -> bool:
    return backbone in {"LLAMA", "MISTRAL", "CAUSAL_LM"}


def _canonical_backbone(value: str) -> str:
    return value.upper().replace("-", "_")


def _canonical_explained_backbone(value: str) -> str:
    normalized = _canonical_backbone(value)
    return _EXPLAINED_BACKBONE_ALIASES.get(normalized, normalized)


def _validate_backbones(explained_backbone: str, interpreter_backbone: str) -> None:
    explained_backbone = _canonical_explained_backbone(explained_backbone)
    interpreter_backbone = _canonical_backbone(interpreter_backbone)
    if explained_backbone not in _SUPPORTED_EXPLAINED_BACKBONES:
        supported = ", ".join(sorted(_SUPPORTED_EXPLAINED_BACKBONES))
        raise ValueError(f"Unsupported explained_model_backbone '{explained_backbone}'. Supported: {supported}")
    if interpreter_backbone not in _SUPPORTED_INTERPRETER_BACKBONES:
        supported = ", ".join(sorted(_SUPPORTED_INTERPRETER_BACKBONES))
        raise ValueError(
            f"Unsupported interpreter_model_backbone '{interpreter_backbone}'. "
            f"AML interpreter backbones currently support: {supported}"
        )


def _slug_component(value: str) -> str:
    raw_name = Path(value).name or value
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip("-._")
    if not slug:
        slug = "model"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def _model_component(label: str, value) -> str:
    if value in (None, ""):
        return f"{label}-default"
    return f"{label}-{_slug_component(value)}"


def apply_cli_args(args: argparse.Namespace) -> None:
    """Apply parsed CLI arguments to ExpArgs, including optional model overrides."""
    from config.config import ExpArgs
    from config.types_enums import RefTokenNameTypes
    from runs.runs_utils import get_task

    _validate_backbones(args.explained_model_backbone, args.interpreter_model_backbone)
    explained_backbone = _canonical_explained_backbone(args.explained_model_backbone)
    interpreter_backbone = _canonical_backbone(args.interpreter_model_backbone)
    ExpArgs.task = get_task(args.task)
    ExpArgs.explained_model_backbone = explained_backbone
    ExpArgs.interpreter_model_backbone = interpreter_backbone
    ExpArgs.eval_metric = args.metric
    ExpArgs.explained_model_name_or_path = args.explained_model_name_or_path
    ExpArgs.interpreter_model_name_or_path = args.interpreter_model_name_or_path
    ExpArgs.explained_tokenizer_name_or_path = args.explained_tokenizer_name_or_path
    ExpArgs.interpreter_tokenizer_name_or_path = args.interpreter_tokenizer_name_or_path
    ExpArgs.llm_adapter_path = args.llm_adapter_path
    ExpArgs.local_files_only = args.local_files_only
    ExpArgs.trust_remote_code = args.trust_remote_code

    if _is_llm_backbone(ExpArgs.explained_model_backbone):
        ExpArgs.ref_token_name = RefTokenNameTypes.UNK.value
        ExpArgs.accumulate_grad_batches = 5
        ExpArgs.batch_size = 4


def build_experiment_name_prefix() -> str:
    """Build a result prefix that distinguishes model override paths."""
    from config.config import ExpArgs

    base = (
        f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_"
        f"{ExpArgs.interpreter_model_backbone}_{ExpArgs.eval_metric}"
    )
    explained = _model_component("explained", ExpArgs.explained_model_name_or_path)
    interpreter = _model_component("interpreter", ExpArgs.interpreter_model_name_or_path)
    return f"{base}_{explained}_{interpreter}"
