# Task 1 Result: AML 代码审计与 Interaction Diagnostic 实施方案

## A. AML 代码审计报告

### 1. 关键文件与功能映射

- `baselines/aml-main_copy/runs/run.py`：原始总入口，串联 HP search、pAML pretraining、pretrain inference、fAML finetuning。
- `baselines/aml-main_copy/main/data_module.py`：数据集加载、tokenization、padding、prompt 拼接、跨 tokenizer 对齐。
- `baselines/aml-main_copy/models/train_models_utils.py`：explained model、interpreter model、tokenizer、reference token 和 trainable label embedding 的加载。
- `baselines/aml-main_copy/models/aml_model.py`：AML 主训练逻辑，包含 attribution 生成、dual mask forward、loss、validation evaluation。
- `baselines/aml-main_copy/models/aml_model_fine_tune.py`：instance-specific finetuning。
- `baselines/aml-main_copy/models/interpreter_models/*.py`：BERT/Roberta/DistilBERT attribution model，输出 per-token sigmoid attribution。
- `baselines/aml-main_copy/evaluations/metrics/metrics_utils.py`：官方 sufficiency、comprehensiveness、log-odds 评估实现。

### 2. AML 的真实代码路径

1. `run.py` 设置 `ExpArgs.task / explained_model_backbone / interpreter_model_backbone / eval_metric`。
2. `load_explained_model()` 加载并冻结 classifier / LLM。
3. `HpSearch` 搜索 loss coefficients。
4. `PreTrain` 训练 pAML attribution model。
5. `InferencePretrain` 在 test split 上推理 pAML。
6. `FineTune` 对每个 test item 复制 interpreter 并做 instance-specific fAML。

### 3. Attribution/mask 语义与张量形状

- `interpreter_model(...) -> tokens_attr`，形状为 `[batch, seq_len]`，经过 sigmoid，范围 `[0, 1]`。
- attribution score 是“保留强度/保留概率”，不是遮挡概率。
- explained-side special tokens 被强制设为 `1`，即永远保留；interpreter-side special tokens 先置为 `0`，用于 regularization/evaluation 排除。
- 默认 mask 施加在 input embedding 层：`embedding = (1-a) * ref_embedding + a * original_embedding`。
- inverse/complement mask 使用 `1-a`，再把 special tokens 重新设为 `1`。

### 4. 当前 loss 的准确公式/伪代码

```text
logits_orig = F(x)
p_orig = softmax(logits_orig)
a = G_theta(x, p_orig)

x_preserve = blend(a, x)
x_inverse = blend(1-a, x)

prediction_loss = CrossEntropyLoss(F(x_preserve).logits, p_orig)
inverse_loss = -log(1 - softmax(F(x_inverse))[argmax(p_orig)])
regularization_loss = BCE(a_interpreter, 0) 或 L1(a_interpreter)

loss = lambda_p * prediction_loss
     + lambda_inv * inverse_loss
     + lambda_s * regularization_loss
```

训练中每个 batch 至少包含 3 次 explained-model forward：原始输入、preserve mask、inverse mask。validation 还会为每个 evaluation budget 额外 forward。

### 5. 训练和评估入口

- 训练入口：`runs/run.py`
- pAML：`main/run_pre_train.py`
- fAML：`main/run_fine_tune.py`
- pAML inference：`main/run_infrence_pre_train.py`
- evaluation：`evaluations/evaluations.py` 与 `evaluations/metrics/metrics_utils.py`

官方 evaluation 协议有一个关键差异：训练使用 embedding-level soft blend；`SUFFICIENCY` 和 `COMPREHENSIVENESS` 使用 hard deletion；`EVAL_LOG_ODDS` 才使用 reference token 替换。

### 6. 后续扩展最适合插入的位置

- 最小侵入诊断实验应放在 `experiments/aml_interaction_diagnostic/`。
- 通过 adapter 复用 AML attribution 和 explained model scoring，不直接改 `aml_model.py`。
- 若未来进入 interaction-aware AML，最自然插入点是 `AmlModel.calculate_tokens_attribution()` 之后、`forward_with_token_attributions()` 之前。

### 7. 论文描述与代码实现需注意的不一致/细节

- 代码中 target preservation 使用 soft target probability distribution，而不是 hard label CE。
- 代码默认 ref token 为 tokenizer mask token，但 evaluation 的 sufficiency/comprehensiveness 不是 ref-token replacement，而是 hard deletion。
- 当前 baseline 副本缺少 `runs/runs_utils.py`，导致原入口无法直接解析 task。
- 当前数据管线没有 natural word 到 subword 的可靠 offset mapping；encoder-only 场景不构建 `MAP_TOKENS`，非 encoder-only 只做 explained/interpreter tokenizer 对齐。

## B. Phase 0 / Phase 1 / Phase 2 实施计划

### Phase 0: 审计 AML、跑通 baseline、最小 smoke test

- 补 `runs_utils.py`，恢复 baseline task alias 解析。
- 扩展 CLI，支持直接传入 explained/interpreter model/tokenizer path、LLM adapter path、`local_files_only`。
- 在不传 override 时保持 AML 原硬编码任务配置不变。
- 用 unit test 验证 resolver、CLI override 和 path fallback。

### Phase 1: 实现 interaction diagnostic experiment

- 新增 `experiments/aml_interaction_diagnostic/`，不污染 baseline 核心训练文件。
- 实现 word-to-subword alignment、candidate graph、interaction teacher、faithfulness metrics、bucket/statistics/casebook。
- 主 interaction teacher score 使用 target probability，与 AML official evaluation 对齐。
- dependency parser 使用 spaCy；缺模型时可 `--disable-dependency` 跑 adjacent-only smoke。

### Phase 2: interaction-aware AML

仅在 Phase 1 诊断结果支持研究假设后再启动；本任务不实现 GNN、edge head、refined mask 或新的 AML 训练目标。

## C-D. 新增/修改文件与职责

- `baselines/aml-main_copy/runs/runs_utils.py`：task alias resolver。
- `baselines/aml-main_copy/runs/run_cli.py`：CLI parser 和 `ExpArgs` override 写入。
- `baselines/aml-main_copy/models/model_path_overrides.py`：模型/tokenizer/adapter path 解析 helper。
- `baselines/aml-main_copy/runs/run.py`：改为 `main()` 入口并接入新 CLI。
- `baselines/aml-main_copy/models/train_models_utils.py`：所有 from_pretrained 路径先查 override。
- `baselines/aml-main_copy/main/data_module.py`：区分 explained/interpreter tokenizer role。
- `experiments/aml_interaction_diagnostic/src/*.py`：诊断实验核心模块。
- `experiments/aml_interaction_diagnostic/scripts/*.py`：主实验、coverage、aggregation、casebook 命令。
- `experiments/aml_interaction_diagnostic/tests/*.py`：baseline compatibility 与 diagnostic primitive 单元测试。

## E. 关键数据结构与张量/列表形状

- `WordUnit(index, text, subword_indices, char_start, char_end)`：一个自然词对应一个或多个 token index。
- `CandidateEdge(i, j, edge_type)`：无向词对边，`edge_type in {adjacent, dependency, adjacent_dependency}`。
- `EdgeInteractionScore`：保存 `s(x), s(x\i), s(x\j), s(x\ij), I_ij, |I_ij|, normalized_I_ij`。
- `AttributionOutput`：保存 predicted label、original target probability、word attribution。
- `per_example.jsonl`：每个样本的 text、words、word attribution、top interactions、bucket、faithfulness。
- `per_edge.jsonl`：每条候选边的 finite-difference teacher signal。

## F. Interaction teacher batch/caching 方案

每个样本使用 cache key `frozenset(masked_word_indices)`：

```text
score_cache[{}]        = s(x)
score_cache[{i}]       = s(x\i)
score_cache[{j}]       = s(x\j)
score_cache[{i, j}]    = s(x\ij)
```

主实验候选边为 adjacent + dependency，边数近似 `O(n)`。同一 word 出现在多条边时 singleton score 只算一次。后续接入真实模型时可把同一批 mask queries 合成 batch forward。

## G. Forward 数量与复杂度

- 主 interaction teacher：每样本约 `1 + |V_E| + |E|` 次 scorer 查询；batch 后为若干批 explained-model forward。
- Faithfulness：默认 budgets `[1,5,10,20,50]`，每样本 deletion 与 sufficiency 各 5 次，即约 10 次 scorer 查询。
- all-pairs reference：仅短文本子集，约 `1 + n + n(n-1)/2`，默认限制 `max_words <= 20`。

## H. 运行命令草案

主诊断 smoke：

```bash
python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py --max-samples 2 --disable-dependency
```

主诊断配置运行：

```bash
python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py --config experiments/aml_interaction_diagnostic/configs/diagnostic_sst2.yaml
```

all-pairs coverage：

```bash
python experiments/aml_interaction_diagnostic/scripts/run_candidate_coverage.py --config experiments/aml_interaction_diagnostic/configs/diagnostic_sst2.yaml --max-words 20 --num-samples 100
```

casebook / aggregation：

```bash
python experiments/aml_interaction_diagnostic/scripts/generate_casebook.py --run-dir <run_dir>
python experiments/aml_interaction_diagnostic/scripts/aggregate_results.py --run-dir <run_dir>
```

baseline override 示例：

```bash
python baselines/aml-main_copy/runs/run.py sst2 BERT BERT AOPC_COMPREHENSIVENESS --explained-model-name-or-path <model> --interpreter-model-name-or-path <model>
```

## I. 单元测试计划

- task resolver alias 与 unknown task。
- CLI override 写入 `ExpArgs`。
- model/tokenizer path fallback。
- word/subword alignment：全部 subword 被归为同一 natural word，special tokens 不入候选。
- masking：mask 一个 word group 时只删除该词所有 subwords。
- candidate graph：去重并合并 `adjacent_dependency`。
- finite difference teacher：公式与 singleton cache。
- metric direction：comprehensiveness 越大越好，sufficiency error 越小越好。
- interaction strength、quantile buckets、CandidateRecall@K、effect size。

## J. 已确认决策与不确定点

已确认：

1. 允许补最小 `runs_utils.py` 或实验侧 task resolver。
2. interaction teacher 主分数跟 AML evaluation 使用 target probability。
3. dependency parser 用 spaCy。
4. baseline 支持命令行直接指定模型/tokenizer/adapter path。

当前限制：

- 当前 workspace 不是 git repo，无法使用 git worktree 或 git diff。
- `rg.exe` 在当前环境被拒绝执行，因此代码搜索使用 PowerShell。
- 本地默认 Python 是 3.8 且无 pytest；验证使用 Codex bundled Python 3.12 与 `unittest`。
- `BaselineAmlAdapter` 当前是真实 AML checkpoint 接入边界；mock smoke 已可运行，完整 checkpoint loader 需在拿到具体 checkpoint/run artifact 后补齐。

## K. 本轮补充：使用说明、Qwen 兼容性与 device 审计

使用说明文档已更新到 `experiments/aml_interaction_diagnostic/USAGE.md`，所有命令均改为 Linux 风格，覆盖模型路径、数据集、输出目录、显卡号、baseline 和 diagnostic 两类入口。

关于命令：

```bash
python baselines/aml-main_copy/runs/run.py sst2 ROBERTA LLAMA AOPC_COMPREHENSIVENESS
```

该组合现在会被 CLI 提前拒绝，因为当前 AML interpreter 只实现了 `BERT/ROBERTA/DISTILBERT`，没有 `LLAMA` interpreter。若把 `--explained-model-name-or-path` 指向真正的 Qwen checkpoint，同时仍声明 `explained_backbone=ROBERTA`，也不能正确工作；代码会走 `RobertaForSequenceClassification.from_pretrained(...)` 和 RoBERTa tokenizer 逻辑，而 Qwen 不是 RoBERTa 架构。

结果目录已能区分不同模型路径：`run.py` 统一构造 `experiment_name_prefix`，并把 sanitized basename 加 full path hash 写入 `HP/PRETRAIN/INFERENCE_PRETRAIN/FINE_TUNE` 的实验名。例如：

```text
PRETRAIN_sst_ROBERTA_ROBERTA_AOPC_COMPREHENSIVENESS_explained-roberta-sst2-<hash>_interpreter-roberta-base-<hash>_<timestamp>
```

Device 数据流已重新检查并修复关键硬编码 CUDA 点：

- `evaluations/metrics/metrics_utils.py`：hard deletion evaluation 输入改为 `.to(get_device())`。
- `models/train_models_utils.py`：encoder-only explained model 加载后改为 `model.to(get_device())`。
- `models/aml_model.py`：临时 tensor 和 `swap()` index tensor 改为跟随 `self.device` 或 `input_tensor.device`。
- `utils/utils_functions.py`：prompt merge 输出改为 `.to(get_device())`。
- `utils/utils_functions.py`：LLM prompt 模式下 `ExpArgs.label_vocab_tokens` 索引改为跟随 `logits.device`。
- `evaluations/metrics/metrics_utils.py`：special-token 判断中的 token id tensor 改为跟随 `input_ids.device`。
- 新增 `test_device_safety.py` 静态扫描关键运行文件，防止非注释 `.cuda()` 再进入 baseline 关键路径。

## L. 本轮补充：通用 CausalLM explained model 支持

已新增 `CAUSAL_LM` explained backbone，用于 Qwen、DeepSeek、Llama、Mistral 等 decoder-only CausalLM explained model。它不新增 Qwen/DeepSeek 专用 AML 逻辑，而是把这些模型统一看作 Hugging Face `AutoModelForCausalLM + AutoTokenizer`，继续复用 AML 原有的 LLM verbalizer scoring 协议。

新增命令示例：

```bash
python baselines/aml-main_copy/runs/run.py sst2 CAUSAL_LM ROBERTA AOPC_COMPREHENSIVENESS \
  --explained-model-name-or-path /models/Qwen2.5-7B-Instruct \
  --interpreter-model-name-or-path /models/roberta-base \
  --local-files-only
```

`qwen`、`qwen2`、`qwen3`、`deepseek`、`deepseek_v2`、`deepseek_v3` 可作为 CLI alias，内部归一为 `CAUSAL_LM`。需要自定义 HF 代码的模型可传 `--trust-remote-code`。

实现边界：

- 不改 `AmlModel.forward()`、soft embedding blend、loss、hard deletion evaluation 等 AML 核心逻辑。
- interpreter backbone 仍保持 `BERT/ROBERTA/DISTILBERT`，因为这是自定义 attribution model。
- `CAUSAL_LM` 必须显式传 `--explained-model-name-or-path`；任务配置中不再硬编码 Qwen/DeepSeek 默认路径。
- `CAUSAL_LM` 使用 prompt verbalizer：构造 task/input/label prompt，取最后 token logits 上的 label token probability。
- tokenizer 没有 pad token 时 fallback 到 eos token；reference token 依次 fallback `unk -> pad -> eos`，避免 Qwen/DeepSeek 类 tokenizer 因缺少 `unk_token_id` 直接失败。
