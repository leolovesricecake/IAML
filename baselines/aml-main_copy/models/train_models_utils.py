import gc
from pathlib import Path
from typing import Tuple

import torch
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from peft import PeftModel
from torch import Tensor, nn
from transformers import (BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AutoTokenizer, AutoConfig,
                          AutoModelForCausalLM)

from config.config import ExpArgs, BackbonesMetaData
from config.constants import HF_CACHE, NEW_ADDED_TRAINABLE_PARAMS
from config.types_enums import ModelBackboneTypes, LabelTokenPosition, RefTokenNameTypes
from models.model_path_overrides import (get_explained_model_name_or_path, get_interpreter_model_name_or_path,
                                         get_llm_adapter_path, get_tokenizer_name_or_path,
                                         hf_from_pretrained_kwargs)
from models.interpreter_models.bert_interpreter import BertInterpreter
from models.interpreter_models.distilbert_interpreter import DistilBertInterpreter
from models.interpreter_models.roberta_interpreter import RobertaInterpreter
from utils.dataclasses import Task
from utils.utils_functions import get_device, is_model_encoder_only


def load_explained_model():
    task = ExpArgs.task
    if ExpArgs.explained_model_backbone == ModelBackboneTypes.BERT.value:
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            get_explained_model_name_or_path(task, ExpArgs.explained_model_backbone),
            **hf_from_pretrained_kwargs())
        model.to(get_device())
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.ROBERTA.value:
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained(
            get_explained_model_name_or_path(task, ExpArgs.explained_model_backbone),
            **hf_from_pretrained_kwargs())
        model.to(get_device())
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        from transformers import DistilBertForSequenceClassification
        model = DistilBertForSequenceClassification.from_pretrained(
            get_explained_model_name_or_path(task, ExpArgs.explained_model_backbone),
            **hf_from_pretrained_kwargs())
        model.to(get_device())
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        from transformers import LlamaForCausalLM, LlamaForSequenceClassification
        model_path = get_explained_model_name_or_path(task, ExpArgs.explained_model_backbone)
        adapter_path = get_llm_adapter_path(task, ExpArgs.explained_model_backbone)
        if task.is_llm_use_lora or adapter_path:
            model = LlamaForSequenceClassification.from_pretrained(model_path, torch_dtype = torch.bfloat16,
                                                                   num_labels = len(task.labels_int_str_maps.keys()),
                                                                   device_map = "auto",
                                                                   **hf_from_pretrained_kwargs())
            model = PeftModel.from_pretrained(model, adapter_path, device_map = "auto",
                                              local_files_only = bool(ExpArgs.local_files_only))
            model = model.merge_and_unload()

        else:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16, device_map = "auto",
                                                     **hf_from_pretrained_kwargs())

        if ExpArgs.ref_token_name == RefTokenNameTypes.UNK.value:
            model.config.pad_token_id = model.config.eos_token_id
        else:
            raise ValueError("support eos_token_id only for LLMs")
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.MISTRAL.value:
        from transformers import MistralForCausalLM, MistralForSequenceClassification
        model_path = get_explained_model_name_or_path(task, ExpArgs.explained_model_backbone)
        adapter_path = get_llm_adapter_path(task, ExpArgs.explained_model_backbone)
        if task.is_llm_use_lora or adapter_path:
            model = MistralForSequenceClassification.from_pretrained(model_path, torch_dtype = torch.bfloat16,
                                                                     num_labels = len(task.labels_int_str_maps.keys()),
                                                                     **hf_from_pretrained_kwargs())
            model = PeftModel.from_pretrained(model, adapter_path,
                                              local_files_only = bool(ExpArgs.local_files_only))
            model = model.merge_and_unload()
        else:
            model = MistralForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16,
                                                       **hf_from_pretrained_kwargs())

        if ExpArgs.ref_token_name == RefTokenNameTypes.UNK.value:
            model.config.pad_token_id = model.config.eos_token_id
        else:
            raise ValueError("support eos_token_id only for LLMs")

    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.CAUSAL_LM.value:
        model_path = get_explained_model_name_or_path(task, ExpArgs.explained_model_backbone)
        adapter_path = get_llm_adapter_path(task, ExpArgs.explained_model_backbone)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16,
                                                     device_map = "auto", **hf_from_pretrained_kwargs())
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, device_map = "auto",
                                              local_files_only = bool(ExpArgs.local_files_only))
            model = model.merge_and_unload()

        if model.config.pad_token_id is None:
            if model.config.eos_token_id is None:
                raise ValueError("CAUSAL_LM explained models require pad_token_id or eos_token_id")
            model.config.pad_token_id = model.config.eos_token_id

    else:
        raise ValueError("unsupported model backbone explained model selected")

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    return model


# For llm model - do not use the fine-tuned interpreter model option
def get_interpreter_model_path(task: Task):
    if ExpArgs.interpreter_model_backbone in [
        ModelBackboneTypes.BERT.value,
        ModelBackboneTypes.ROBERTA.value,
        ModelBackboneTypes.DISTILBERT.value,
    ]:
        return get_interpreter_model_name_or_path(task, ExpArgs.interpreter_model_backbone)
    raise ValueError("unsupported model backbone selected - interpreter model")


def load_interpreter_model():
    task = ExpArgs.task
    model_path: str = get_interpreter_model_path(task)
    interpreter_model_backbone = ExpArgs.interpreter_model_backbone
    if interpreter_model_backbone == ModelBackboneTypes.BERT.value:
        return BertInterpreter.from_pretrained(model_path, **hf_from_pretrained_kwargs())
    elif interpreter_model_backbone == ModelBackboneTypes.ROBERTA.value:
        return RobertaInterpreter.from_pretrained(model_path, **hf_from_pretrained_kwargs())
    elif interpreter_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        return DistilBertInterpreter.from_pretrained(model_path, **hf_from_pretrained_kwargs())
    else:
        raise ValueError("unsupported model backbone selected")


def get_interpreter_config():
    task = ExpArgs.task
    model_path: str = get_interpreter_model_path(task)
    return AutoConfig.from_pretrained(model_path, **hf_from_pretrained_kwargs())


def get_models_tokenizer(model_backbone, role = "explained"):
    task = ExpArgs.task
    if model_backbone == ModelBackboneTypes.BERT.value:
        return BertTokenizer.from_pretrained(
            get_tokenizer_name_or_path(task, model_backbone, role = role), **hf_from_pretrained_kwargs())
    elif model_backbone == ModelBackboneTypes.ROBERTA.value:
        return RobertaTokenizer.from_pretrained(
            get_tokenizer_name_or_path(task, model_backbone, role = role), **hf_from_pretrained_kwargs())
    elif model_backbone == ModelBackboneTypes.DISTILBERT.value:
        return DistilBertTokenizer.from_pretrained(
            get_tokenizer_name_or_path(task, model_backbone, role = role), **hf_from_pretrained_kwargs())
    elif model_backbone == ModelBackboneTypes.LLAMA.value:
        new_tokenizer = AutoTokenizer.from_pretrained(
            get_tokenizer_name_or_path(task, model_backbone, role = role), padding_side = 'left',
            **hf_from_pretrained_kwargs())
        if new_tokenizer.pad_token_id is None:
            if new_tokenizer.eos_token_id is None:
                raise ValueError(f"{model_backbone} tokenizer requires pad_token_id or eos_token_id")
            new_tokenizer.pad_token_id = new_tokenizer.eos_token_id
        if ExpArgs.ref_token_name == RefTokenNameTypes.UNK.value:
            return new_tokenizer
        else:
            raise ValueError("support eos_token_id only for LLMs")
    elif model_backbone in [ModelBackboneTypes.MISTRAL.value, ModelBackboneTypes.CAUSAL_LM.value]:
        new_tokenizer = AutoTokenizer.from_pretrained(
            get_tokenizer_name_or_path(task, model_backbone, role = role), padding_side = 'left',
            **hf_from_pretrained_kwargs())
        if new_tokenizer.pad_token_id is None:
            if new_tokenizer.eos_token_id is None:
                raise ValueError(f"{model_backbone} tokenizer requires pad_token_id or eos_token_id")
            new_tokenizer.pad_token_id = new_tokenizer.eos_token_id
        if ExpArgs.ref_token_name == RefTokenNameTypes.UNK.value:
            return new_tokenizer
        else:
            raise ValueError("support eos_token_id only for LLMs")
    else:
        raise ValueError("unsupported model type selected")


def get_warmup_steps_and_total_training_steps(n_epochs: int, train_samples_length: int, batch_size: int,
                                              warmup_ratio: int, accumulate_grad_batches: int) -> Tuple[int, int]:
    effective_batch_size = batch_size * accumulate_grad_batches
    steps_per_epoch = (train_samples_length // effective_batch_size) + 1
    total_training_steps = int(steps_per_epoch * n_epochs)
    warmup_steps = int(total_training_steps * warmup_ratio)
    return warmup_steps, total_training_steps


def construct_word_embedding(model, model_backbone: ModelBackboneTypes, input_ids: Tensor):
    if is_model_encoder_only(model_backbone):
        backbone_name = BackbonesMetaData.name[model_backbone]
        model = getattr(model, backbone_name)
        return model.embeddings.word_embeddings(input_ids)
    else:
        return model.get_input_embeddings()(input_ids)


def get_explained_ref_token_name(explained_tokenizer):
    if ExpArgs.ref_token_name == RefTokenNameTypes.MASK.value:
        return explained_tokenizer.mask_token_id
    elif ExpArgs.ref_token_name == RefTokenNameTypes.PAD.value:
        return explained_tokenizer.pad_token_id
    elif ExpArgs.ref_token_name == RefTokenNameTypes.EOS.value:
        return explained_tokenizer.eos_token_id
    elif ExpArgs.ref_token_name == RefTokenNameTypes.UNK.value:
        for token_id in [
            explained_tokenizer.unk_token_id,
            explained_tokenizer.pad_token_id,
            explained_tokenizer.eos_token_id,
        ]:
            if token_id is not None:
                return token_id
        raise ValueError("ref token invalid: tokenizer has no unk_token_id, pad_token_id, or eos_token_id")
    else:
        raise ValueError("ref name invalid")


def is_add_label_embedding():
    return ExpArgs.interpreter_label_token_position != LabelTokenPosition.NONE.value


def init_trainable_embeddings():
    task = ExpArgs.task
    interpreter_config = get_interpreter_config()
    label_embedding_index = None
    if is_add_label_embedding():
        num_label_embeddings = len(task.labels_str_int_maps.keys())
        if ExpArgs.is_include_general_label_token:
            num_label_embeddings = num_label_embeddings + 1
            label_embedding_index = torch.tensor(num_label_embeddings - 1)  # last item index
        trainable_embeddings = nn.Embedding(num_label_embeddings, interpreter_config.hidden_size,
                                            padding_idx = interpreter_config.pad_token_id)
        trainable_embeddings.weight.data.normal_(mean = 0.0, std = interpreter_config.initializer_range)

        for p in trainable_embeddings.parameters():
            p.requires_grad = True

        trainable_embeddings.train()

        return trainable_embeddings, label_embedding_index


def load_trainable_embeddings(trainable_embeddings):
    trainable_embeddings.eval()
    file_path = f"{ExpArgs.fine_tuned_interpreter_model_path}/{NEW_ADDED_TRAINABLE_PARAMS}.pth"
    if is_add_label_embedding():
        if Path(file_path).is_file():
            checkpoint = torch.load(file_path)
            trainable_embeddings.load_state_dict(checkpoint[NEW_ADDED_TRAINABLE_PARAMS])

            for p in trainable_embeddings.parameters():
                p.requires_grad = False
        else:
            raise ValueError("can not find saved embeddings")


def custom_teardown(trainer) -> None:
    self = trainer.strategy
    _optimizers_to_device(self.optimizers, torch.device("cpu"))

    if self.lightning_module is not None:
        self.lightning_module.interpreter_model.cpu()
    self.precision_plugin.teardown()
    self.accelerator.teardown()
    self.checkpoint_io.teardown()


def run_trainer(trainer, model, data_module, explained_model):
    gc.collect()
    torch.cuda.empty_cache()
    explained_model.zero_grad()


    trainer.fit(model = model, datamodule = data_module)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    explained_model.zero_grad()
