# AML Interaction Diagnostic 使用说明

本文只写 Linux 风格命令。当前开发环境虽然是 Windows，但最终运行时请按 Linux 命令执行。

本项目有两个主要入口：

- `baselines/aml-main_copy/runs/run.py`：运行 AML baseline，包括 pAML pretrain、pAML inference，代码中 fAML 入口仍保留但默认未启用。
- `experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py`：运行 interaction diagnostic，用来回答“AML 是否在具有高局部二阶 interaction 的样本上表现出更差的 attribution faithfulness”。

## 1. 环境与显卡

选择第 0 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python ...
```

选择第 1 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 python ...
```

注意：进程内部看到的第一张可见 GPU 会显示为 `cuda:0`，这是 CUDA 的正常行为。

安装 diagnostic 依赖：

```bash
pip install -r experiments/aml_interaction_diagnostic/requirements_diagnostic.txt
python -m spacy download en_core_web_sm
```

如果只跑 adjacent-only 或 smoke test，可以不安装 spaCy 模型，并传入：

```bash
--disable-dependency
```

## 2. 先跑一个 Mock Smoke Test

这个命令不需要 AML checkpoint，也不需要真实模型权重，只验证 diagnostic 管线和输出文件是否正常：

```bash
python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py \
  --adapter mock \
  --max-samples 2 \
  --disable-dependency \
  --output-dir experiments/aml_interaction_diagnostic/outputs/smoke_mock
```

mock 输出只能用于检查管线，不能用于论文结论。

## 3. 运行 AML Baseline

基本格式：

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

`qwen`、`qwen2`、`qwen3`、`deepseek`、`deepseek_v2`、`deepseek_v3` 可以作为 explained backbone 的命令行别名使用，内部会归一到 `CAUSAL_LM`。

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

可选模型参数：

```text
--explained-model-name-or-path
--interpreter-model-name-or-path
--explained-tokenizer-name-or-path
--interpreter-tokenizer-name-or-path
--llm-adapter-path
--local-files-only
--trust-remote-code
```

如果只指定模型路径，tokenizer 默认回退到对应模型路径；如果 tokenizer 单独存放，再显式传 tokenizer 参数。

## 4. Baseline 示例：RoBERTa explained + RoBERTa interpreter

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  sst2 ROBERTA ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/roberta-sst2 \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only
```

## 5. Baseline 示例：Qwen / DeepSeek / Llama explained model

如果 explained model 是 Qwen、DeepSeek、Llama、Mistral 等 decoder-only CausalLM，推荐统一使用 `CAUSAL_LM`：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  sst2 CAUSAL_LM ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/Qwen2.5-7B-Instruct \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only \
  --trust-remote-code
```

也可以使用 alias：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  sst2 qwen ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/Qwen2.5-7B-Instruct \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only \
  --trust-remote-code
```

DeepSeek 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/aml-main_copy/runs/run.py \
  imdb deepseek ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/deepseek-llm-7b \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only \
  --trust-remote-code
```

不要这样写：

```bash
python baselines/aml-main_copy/runs/run.py sst2 ROBERTA LLAMA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/Qwen2.5-7B-Instruct
```

原因：

- `explained_backbone=ROBERTA` 会走 `RobertaForSequenceClassification`，不能正确加载 Qwen。
- `interpreter_backbone=LLAMA` 不支持；AML interpreter 目前只实现了 `BERT/ROBERTA/DISTILBERT`。
- Qwen/DeepSeek/Llama/Mistral 这类模型应该放在 explained side，写成 `CAUSAL_LM` 或 alias；interpreter side 仍用 encoder-only attribution model。

## 6. Baseline 输出目录

baseline 输出根目录来自 AML 的 `ExpArgs.default_root_dir`，默认是 `OUT` 结构：

```text
OUT/CONFIG
OUT/PRE_TRAIN
OUT/INFERENCE_PRETRAIN
OUT/FINE_TUNE
```

实验名会包含 task、backbone、metric，以及模型 override 的 basename 和路径 hash。例如：

```text
PRETRAIN_sst_CAUSAL_LM_ROBERTA_AOPC_COMPREHENSIVENESS_explained-Qwen2.5-7B-Instruct-<hash>_interpreter-roberta-base-<hash>_<timestamp>
```

因此不同的 `--explained-model-name-or-path` 或 `--interpreter-model-name-or-path` 会进入不同结果目录。

`INFERENCE_PRETRAIN/RESULTS_DF/<experiment>/` 和 `FINE_TUNE/RESULTS_DF/<experiment>/` 会保存：

```text
results.csv
summary.json
aopc_curve.csv
```

其中：

- `results.csv`：每个样本的 metric 结果。
- `summary.json`：配置、模型、指标聚合、时间、git commit 等 run-level 汇总。
- `aopc_curve.csv`：按 budget 聚合的曲线数据，可直接用于画 AOPC 曲线。

## 7. 真实 Interaction Diagnostic：使用 pAML Checkpoint

真实 diagnostic 使用 `--adapter baseline`，需要一个已有 pAML checkpoint，也就是 baseline pretrain 产物：

```text
OUT/PRE_TRAIN/CHECKPOINTS/<pretrain_experiment>
```

它不依赖 fAML per-sample checkpoint。

RoBERTa explained 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py \
  --adapter baseline \
  --baseline-root baselines/aml-main_copy \
  --task sst2 \
  --explained-model-backbone ROBERTA \
  --interpreter-model-backbone ROBERTA \
  --metric AOPC_COMPREHENSIVENESS \
  --interpreter-checkpoint OUT/PRE_TRAIN/CHECKPOINTS/<pretrain_experiment> \
  --explained-model-name-or-path /models/roberta-sst2 \
  --interpreter-model-name-or-path /models/roberta-base \
  --max-samples 100 \
  --spacy-model en_core_web_sm \
  --output-dir experiments/aml_interaction_diagnostic/outputs/sst2_real_roberta_topk3
```

Qwen explained 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py \
  --adapter baseline \
  --baseline-root baselines/aml-main_copy \
  --task sst2 \
  --explained-model-backbone CAUSAL_LM \
  --interpreter-model-backbone ROBERTA \
  --metric AOPC_COMPREHENSIVENESS \
  --interpreter-checkpoint OUT/PRE_TRAIN/CHECKPOINTS/<pretrain_experiment> \
  --explained-model-name-or-path /models/Qwen2.5-7B-Instruct \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only \
  --trust-remote-code \
  --max-samples 100 \
  --disable-dependency \
  --output-dir experiments/aml_interaction_diagnostic/outputs/sst2_real_qwen_topk3
```

如果 dependency parser 可用，把 `--disable-dependency` 去掉并指定：

```bash
--spacy-model en_core_web_sm
```

## 8. Diagnostic 配置与 Top-k

可以通过 YAML 指定默认配置：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py \
  --adapter baseline \
  --config experiments/aml_interaction_diagnostic/configs/diagnostic_sst2.yaml \
  --interpreter-checkpoint OUT/PRE_TRAIN/CHECKPOINTS/<pretrain_experiment> \
  --output-dir experiments/aml_interaction_diagnostic/outputs/sst2_real_from_yaml
```

`interaction_strength.topk` 会从 YAML 生效。命令行可以覆盖：

```bash
--interaction-topk 5
```

例如：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py \
  --adapter baseline \
  --task sst2 \
  --explained-model-backbone ROBERTA \
  --interpreter-model-backbone ROBERTA \
  --metric AOPC_COMPREHENSIVENESS \
  --interpreter-checkpoint OUT/PRE_TRAIN/CHECKPOINTS/<pretrain_experiment> \
  --interaction-topk 5 \
  --max-samples 100 \
  --disable-dependency \
  --output-dir experiments/aml_interaction_diagnostic/outputs/sst2_real_topk5
```

## 9. Diagnostic 输出文件

一个 diagnostic run 目录会包含：

```text
metadata.json
summary.json
per_example.jsonl
per_edge.jsonl
per_budget_metrics.csv
aopc_curve.csv
bucket_metrics.csv
correlations.json
candidate_coverage.json
plots/binned_trend.csv
casebook.md
```

主要阅读顺序：

1. `summary.json`：run-level 汇总，包括配置、模型、checkpoint、指标均值、时间、git commit。
2. `correlations.json`：interaction strength 和 faithfulness error 的 Pearson/Spearman 相关性。
3. `bucket_metrics.csv`：low/medium/high interaction 桶之间的平均差异。
4. `per_budget_metrics.csv`：每个样本、每个 budget 的 comprehensiveness、sufficiency error、log-odds。
5. `aopc_curve.csv`：按 budget 聚合后的曲线数据，论文画 AOPC 曲线主要用这个文件。
6. `per_example.jsonl`：逐样本 attribution、interaction、faithfulness 详情。
7. `per_edge.jsonl`：每条候选边的二阶 finite-difference interaction。
8. `casebook.md`：高/低 interaction、best/worst faithfulness 案例。

## 10. 指标方向与论文结论怎么读

diagnostic 的主协议与 AML official evaluation 对齐：

- interaction teacher score：target class probability。
- comprehensiveness：hard deletion 后 target probability drop，越大表示 attribution 越能删掉关键内容。
- sufficiency error：只保留 top attribution words 后的 target probability drop，越小越好。
- log-odds：使用 AML 的 ref-token replacement 语义，单独作为 LO 曲线输出。
- AOPC：把多个 budget 下的 step 值聚合，`aopc_curve.csv` 保存的是画曲线用的 per-budget 聚合。

回答研究问题时，请只使用 `--adapter baseline` 的真实结果，不要使用 mock 结果。

核心判断路径：

- 看 `correlations.json`：如果 interaction strength 与 faithfulness error 显著正相关，说明高 interaction 样本上 attribution 更差。
- 看 `bucket_metrics.csv`：比较 high interaction 桶和 low interaction 桶的 `faithfulness_error`、`comprehensiveness_aopc`、`sufficiency_aopc`。
- 看 `summary.json` 的 `high_vs_low`：包括 Cliff's delta 和 standardized mean difference。
- 看 `casebook.md`：人工检查高 interaction 且 faithfulness 差的典型样本。

## 11. 结果重算与 Casebook

重新生成 casebook：

```bash
python experiments/aml_interaction_diagnostic/scripts/generate_casebook.py \
  --run-dir experiments/aml_interaction_diagnostic/outputs/sst2_real_topk3
```

已有 run 目录重新聚合：

```bash
python experiments/aml_interaction_diagnostic/scripts/aggregate_results.py \
  --run-dir experiments/aml_interaction_diagnostic/outputs/sst2_real_topk3
```

短文本 all-pairs coverage：

```bash
python experiments/aml_interaction_diagnostic/scripts/run_candidate_coverage.py \
  --config experiments/aml_interaction_diagnostic/configs/diagnostic_sst2.yaml \
  --max-words 20 \
  --num-samples 100 \
  --output-dir experiments/aml_interaction_diagnostic/outputs/sst2_coverage
```

## 12. 常见问题

### run_diagnostic.py 是否依赖 AML finetune 阶段产物？

不依赖。真实 diagnostic 默认加载 pAML checkpoint：

```text
OUT/PRE_TRAIN/CHECKPOINTS/<pretrain_experiment>
```

### 可以把 `explained_backbone=ROBERTA` 但模型路径指向 Qwen 吗？

不可以。这样会用 RoBERTa 类加载 Qwen 权重，架构不匹配。Qwen/DeepSeek/Llama/Mistral 应使用：

```text
explained_backbone=CAUSAL_LM
interpreter_backbone=ROBERTA/BERT/DISTILBERT
```

### `--local-files-only` 什么时候用？

当模型、tokenizer、checkpoint 都已经在本地时使用，避免 Hugging Face 联网下载：

```bash
--local-files-only
```

### 什么时候需要 `--trust-remote-code`？

Qwen、DeepSeek 或其他 Hugging Face repo 需要自定义模型代码时使用：

```bash
--trust-remote-code
```

### spaCy 没装怎么办？

先用：

```bash
--disable-dependency
```

这会跑 adjacent-only candidate graph；装好 `en_core_web_sm` 后再启用 dependency edges。
