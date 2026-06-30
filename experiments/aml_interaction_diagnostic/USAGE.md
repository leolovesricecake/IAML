# AML Interaction Diagnostic 使用说明

本文档只写 Linux 风格命令。主要入口有两个：

1. `baselines/aml-main_copy/runs/run.py`：运行 AML baseline，支持从命令行指定 explained/interpreter 模型、tokenizer、LoRA adapter。
2. `experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py`：运行 interaction diagnostic，输出 per-example、per-edge、bucket、correlation 和 casebook。

## 1. 环境、依赖和显卡号

选择单张显卡：

```bash
CUDA_VISIBLE_DEVICES=0 python ...
```

选择第 2 张物理显卡：

```bash
CUDA_VISIBLE_DEVICES=1 python ...
```

在进程内部，被选中的第一张可见显卡会显示为 `cuda:0`，这是 CUDA 的正常行为。

安装 diagnostic 依赖：

```bash
pip install -r experiments/aml_interaction_diagnostic/requirements_diagnostic.txt
python -m spacy download en_core_web_sm
```

如果只是跑 adjacent-only smoke test，可以不安装 spaCy 模型，并在命令里加 `--disable-dependency`。

## 2. AML Baseline 命令格式

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  <task> <explained_backbone> <interpreter_backbone> <metric> \
  [options]
```

支持的数据集别名：

```text
imdb
emotion / emotions
sst / sst2
agn / ag_news
rtn / rotten_tomatoes
```

`explained_backbone` 支持：

```text
BERT
ROBERTA
DISTILBERT
LLAMA
MISTRAL
CAUSAL_LM
```

`qwen`、`qwen2`、`qwen3`、`deepseek`、`deepseek_v2`、`deepseek_v3` 可以作为 explained backbone 的命令行别名使用，内部会归一到 `CAUSAL_LM`。推荐直接写 `CAUSAL_LM`，因为 AML 侧关心的是 CausalLM verbalizer 打分协议，而不是模型家族名。

`interpreter_backbone` 目前只支持：

```text
BERT
ROBERTA
DISTILBERT
```

常用 metric：

```text
SUFFICIENCY
COMPREHENSIVENESS
EVAL_LOG_ODDS
AOPC_SUFFICIENCY
AOPC_COMPREHENSIVENESS
AOPC_COMPREHENSIVENESS_AOPC_SUFFICIENCY
COMPREHENSIVENESS_SUFFICIENCY
```

## 3. 指定模型、Tokenizer 和 Adapter

可用参数：

```text
--explained-model-name-or-path
--interpreter-model-name-or-path
--explained-tokenizer-name-or-path
--interpreter-tokenizer-name-or-path
--llm-adapter-path
--local-files-only
--trust-remote-code
```

只指定模型路径时，tokenizer 默认回退到同一个模型路径；如果 tokenizer 单独存放，再显式传 tokenizer 参数。

使用默认 Hugging Face 配置：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  sst2 BERT BERT AOPC_COMPREHENSIVENESS
```

指定本地 encoder-only 模型：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  sst2 ROBERTA ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/roberta-sst2 \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only
```

模型和 tokenizer 分开指定：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  imdb BERT ROBERTA COMPREHENSIVENESS \
  --explained-model-name-or-path /models/bert-imdb-classifier \
  --explained-tokenizer-name-or-path /tokenizers/bert-imdb-tokenizer \
  --interpreter-model-name-or-path /models/roberta-base \
  --interpreter-tokenizer-name-or-path /tokenizers/roberta-base \
  --local-files-only
```

LLM explained model 加 LoRA adapter，interpreter 仍使用 encoder-only backbone：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  emotions LLAMA ROBERTA SUFFICIENCY \
  --explained-model-name-or-path /models/Llama-2-7b-hf \
  --llm-adapter-path /models/llama-emotion-lora \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only
```

通用 CausalLM explained model，例如 Qwen：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  sst2 CAUSAL_LM ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/Qwen2.5-7B-Instruct \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only
```

等价的 Qwen alias 写法：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  sst2 qwen ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/Qwen2.5-7B-Instruct \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only
```

DeepSeek 或需要远程自定义代码的模型：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  imdb deepseek ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/deepseek-llm-7b \
  --interpreter-model-name-or-path /models/roberta-base \
  --trust-remote-code \
  --local-files-only
```

## 4. Qwen/DeepSeek 和 Backbone 兼容性

这个命令现在会被提前拒绝：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  sst2 ROBERTA LLAMA AOPC_COMPREHENSIVENESS
```

原因是 `LLAMA` 不能作为 AML interpreter backbone。当前 interpreter model 只有 `BERT/ROBERTA/DISTILBERT` 三套实现。

下面这种写法也不能正确加载真正的 Qwen/DeepSeek 模型：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  sst2 ROBERTA ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/Qwen2.5-7B
```

原因是 `explained_backbone=ROBERTA` 时，代码会调用 `RobertaForSequenceClassification.from_pretrained(...)` 和 `RobertaTokenizer.from_pretrained(...)`。真正的 Qwen/DeepSeek checkpoint 不是 RoBERTa 架构，因此不会正确工作。

当前结论：

- Qwen/DeepSeek 不能仅靠 `--explained-model-name-or-path` 接到 `ROBERTA` backbone 下。
- CausalLM 类 explained model 统一使用 `CAUSAL_LM`，底层通过 Hugging Face `AutoModelForCausalLM` 和 `AutoTokenizer` 加载。
- `CAUSAL_LM` 使用 AML 原有的 LLM verbalizer 协议：task prompt + input + label prompt，取最后位置 logits 上的 label token probability。
- interpreter 仍选 `BERT/ROBERTA/DISTILBERT`，因为 AML interpreter 是自定义 token attribution model，不是普通 explained LLM。

## 5. 输出目录如何区分模型

baseline 的输出根目录仍来自 `ExpArgs.default_root_dir`，默认写到 AML 的 `OUT` 目录结构下：

```text
OUT/CONFIG
OUT/PRE_TRAIN
OUT/INFERENCE_PRETRAIN
OUT/FINE_TUNE
```

本次已修改实验名前缀。只要传了模型 override，`HP/PRETRAIN/INFERENCE_PRETRAIN/FINE_TUNE` 的实验名都会带上 explained/interpreter 模型路径摘要：

```text
PRETRAIN_sst_CAUSAL_LM_ROBERTA_AOPC_COMPREHENSIVENESS_explained-Qwen2.5-7B-Instruct-<hash>_interpreter-roberta-base-<hash>_<timestamp>
```

摘要由路径 basename 和完整路径 hash 组成。因此不同的 `--explained-model-name-or-path` 或 `--interpreter-model-name-or-path` 会落到不同结果目录里。

## 6. Interaction Diagnostic 命令格式

CPU smoke test，不依赖真实 AML checkpoint：

```bash
python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py \
  --max-samples 2 \
  --disable-dependency \
  --output-dir experiments/aml_interaction_diagnostic/outputs/smoke_run
```

指定配置文件、输出目录和样本数：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py \
  --config experiments/aml_interaction_diagnostic/configs/diagnostic_sst2.yaml \
  --max-samples 100 \
  --output-dir experiments/aml_interaction_diagnostic/outputs/sst2_run \
  --disable-dependency
```

启用 spaCy dependency edges：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py \
  --config experiments/aml_interaction_diagnostic/configs/diagnostic_sst2.yaml \
  --spacy-model en_core_web_sm \
  --max-samples 100 \
  --output-dir experiments/aml_interaction_diagnostic/outputs/sst2_spacy
```

调整 interaction strength 的 top-k：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py \
  --config experiments/aml_interaction_diagnostic/configs/diagnostic_sst2.yaml \
  --interaction-topk 5 \
  --max-samples 100 \
  --output-dir experiments/aml_interaction_diagnostic/outputs/sst2_top5 \
  --disable-dependency
```

配置文件示例：

```text
experiments/aml_interaction_diagnostic/configs/diagnostic_sst2.yaml
experiments/aml_interaction_diagnostic/configs/diagnostic_imdb.yaml
experiments/aml_interaction_diagnostic/configs/diagnostic_eraser_movie_reviews.yaml
```

## 7. Diagnostic 输出文件

一个 run 目录下会生成：

```text
metadata.json
per_example.jsonl
per_edge.jsonl
summary.json
bucket_metrics.csv
correlations.json
candidate_coverage.json
plots/binned_trend.csv
casebook.md
```

主要阅读顺序：

1. `summary.json`：run 级别摘要。
2. `correlations.json`：interaction strength 和 faithfulness error 的相关性。
3. `bucket_metrics.csv`：low/medium/high interaction 桶之间的差异。
4. `per_example.jsonl`：逐样本 attribution、interaction、faithfulness。
5. `casebook.md`：高 interaction、低 interaction、best/worst faithfulness 案例。

## 8. Candidate Coverage 和结果重算

短文本 all-pairs coverage：

```bash
python experiments/aml_interaction_diagnostic/scripts/run_candidate_coverage.py \
  --config experiments/aml_interaction_diagnostic/configs/diagnostic_sst2.yaml \
  --max-words 20 \
  --num-samples 100 \
  --output-dir experiments/aml_interaction_diagnostic/outputs/sst2_coverage
```

已有 run 目录重新生成 casebook：

```bash
python experiments/aml_interaction_diagnostic/scripts/generate_casebook.py \
  --run-dir experiments/aml_interaction_diagnostic/outputs/smoke_run
```

已有 run 目录重新聚合结果：

```bash
python experiments/aml_interaction_diagnostic/scripts/aggregate_results.py \
  --run-dir experiments/aml_interaction_diagnostic/outputs/smoke_run
```

## 9. Baseline 数据流和 Device 检查结论

baseline 主数据流：

```text
run.py
  -> load_explained_model()
  -> HpSearch
  -> PreTrain
  -> InferencePretrain
  -> FineTune
```

训练和评估中的关键张量路径：

```text
DataModule
  -> tokenize / collate_fn / padding / optional prompt
  -> AmlModel.forward()
  -> interpreter attribution
  -> explained-side token attribution
  -> soft embedding blend training loss
  -> hard deletion evaluation metrics
```

本次已检查并修掉的 device 风险：

- `evaluations/metrics/metrics_utils.py`：evaluation hard deletion 后的输入统一 `.to(get_device())`，不再硬编码 `.cuda()`。
- `models/train_models_utils.py`：BERT/RoBERTa/DistilBERT explained model 加载后用 `model.to(get_device())`，不再直接 `.cuda()`。
- `models/aml_model.py`：临时 tensor、swap index tensor 使用 `self.device` 或 `input_tensor.device`。
- `utils/utils_functions.py`：prompt merge 输出移动到 `get_device()`，不再硬编码 CUDA。
- `utils/utils_functions.py`：LLM prompt 模式下的 `label_vocab_tokens` 索引移动到 `logits.device`。
- `evaluations/metrics/metrics_utils.py`：special-token 判断中的 token id tensor 移动到 `input_ids.device`。
- 新增单测会扫描关键运行文件，防止 `.cuda()` 再进入这些路径。

当前仍需注意：

- LLM 路径仍使用 Hugging Face `device_map="auto"`；建议用 `CUDA_VISIBLE_DEVICES=<id>` 限制可见显卡，先按单卡运行验证。
- 当前 diagnostic 的 smoke path 使用 mock adapter；真实 AML checkpoint 接入点在 `experiments/aml_interaction_diagnostic/src/aml_adapter.py`。
- Qwen/DeepSeek/Llama/Mistral 等 CausalLM explained model 推荐统一走 `CAUSAL_LM`；只有 interpreter backbone 仍需限定在 `BERT/ROBERTA/DISTILBERT`。
