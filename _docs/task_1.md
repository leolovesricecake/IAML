你现在位于一个研究代码仓库中。AML 官方实现已经下载到：

```text
baselines/aml-main_copy
```

我的研究目标是：在严格继承 AML 核心机制的前提下，研究其一阶 token attribution 是否会在存在显著局部交互关系的文本样本上失效，并为后续“interaction-aware AML”方法建立可靠代码基础。

请先不要直接大规模重构或实现复杂新模型。请严格按照下面的顺序工作。

# 一、先理解并审计 AML 官方实现

首先完整阅读 `baselines/aml-main_copy` 中与以下内容有关的代码：

1. 被解释模型（explained model / classifier）的加载、冻结与前向过程；
2. attribution model 的结构、输入、输出和训练流程；
3. AML attribution map 的定义；
4. soft masking / hard masking 的具体实现；
5. 原论文中的：

   * preservation loss；
   * inverse/complement masking loss；
   * sparsity regularization；
   * pAML pretraining；
   * fAML finetuning；
6. 训练、验证、测试、保存 attribution、评价 explanation quality 的入口脚本；
7. 数据集处理、tokenizer、attention mask、特殊 token、padding、subword-to-word 映射相关代码。

AML 的核心思想是：训练一个 attribution model (G_\theta)，给定输入 (x) 与待解释模型的目标预测 (y^\star)，输出 attribution mask：

[
a = G_\theta(x, y^\star), \qquad a_i \in [0,1].
]

AML 使用该 mask 构造两个输入：

[
x' = M(a, x),
\qquad
x'' = M(1-a, x),
]

其中 (x') 尽量保留高 attribution 单元，(x'') 尽量遮掉高 attribution 单元。

训练目标通常包含：

[
\mathcal L_{\mathrm{AML}}
=========================

\lambda_p \mathcal L_{\mathrm{preserve}}
+
\lambda_{\mathrm{inv}} \mathcal L_{\mathrm{inverse}}
+
\lambda_s \mathcal L_{\mathrm{sparse}}.
]

其中：

* preservation loss：要求 (F(x')) 保持原预测；
* inverse loss：要求 (F(x'')) 显著偏离原预测；
* sparsity loss：要求 attribution 尽量稀疏。

请不要凭论文描述猜测实现细节。必须以仓库代码为准，明确回答：

* attribution score 的语义到底是“保留概率”还是“遮挡概率”；
* mask 是在 input id、embedding、hidden state 还是其他层施加；
* mask token / learned mask embedding 如何构造；
* target score 使用概率、log probability、logit、cross entropy 还是其他形式；
* 训练时 explained model 是否被冻结；
* attribution model 是否与 explained model 共享 tokenizer / encoder；
* token-level attribution 如何处理 CLS、SEP、PAD、BOS、EOS 等特殊 token；
* 是否已有 word-level 或 phrase-level 映射机制；
* 每个样本在训练中需要多少次 explained model forward。

在开始任何修改前，请输出一份简洁但精确的“AML 代码审计报告”，包括：

```text
1. 关键文件与功能映射
2. AML 的真实代码路径
3. attribution/mask 的张量形状
4. 当前 loss 的准确公式或伪代码
5. 训练和评估入口
6. 后续扩展最适合插入的位置
7. 任何论文描述与代码实现不一致之处
```

---

# 二、研究动机与后续方法方向

AML 当前输出的是独立单元的重要性：

[
a_1, a_2, \ldots, a_n.
]

它可以表示“哪些词重要”，但不能显式表示：

> 哪两个词必须共同保留，模型才会维持当前预测。

例如：

```text
The movie is not good.
```

“not” 和 “good” 可能单独都具有一定影响，但模型真正依赖的是它们组成的联合语义。

对于待解释模型的目标类别分数 (s(x))，定义词/单元 (i,j) 的局部二阶删除交互为：

[
I_{ij}(x)
=========

## s(x)

## s(x_{\setminus i})

s(x_{\setminus j})
+
s(x_{\setminus {i,j}}).
]

其中：

* (x_{\setminus i})：遮挡或删除单元 (i)；
* (x_{\setminus j})：遮挡或删除单元 (j)；
* (x_{\setminus {i,j}})：同时遮挡或删除 (i,j)；
* (s(x))：待解释模型对原预测类别 (y^\star) 的一个标量分数，优先使用 logit；若 AML 代码的已有评价协议使用其他定义，则必须说明并保持一致。

解释约定：

* (I_{ij}>0)：二者存在正协同，联合删除的额外影响大于单独删除影响的可加和；
* (I_{ij}<0)：二者更可能是冗余、替代或负交互；
* (I_{ij}\approx 0)：当前局部行为基本可由一阶影响解释。

注意：这不是全局因果关系，也不是唯一的 Shapley interaction，而是**在特定输入、目标类别和 masking operator 下定义的局部条件二阶有限差分**。

后续拟研究的完整方法是：

```text
候选语言结构图
    ↓
只检查少量语言合理的单元对
    ↓
使用有限 mask queries 得到二阶 interaction teacher signal
    ↓
训练 AML attribution model 同时预测 node attribution 和 edge interaction
    ↓
让 edge interaction 反过来参与最终 attribution mask 的生成
    ↓
在推理时通过一次 attribution-model forward 输出结构化解释
```

图的作用只是把全对搜索空间：

[
O(n^2)
]

缩小成近线性候选边集合：

[
O(n).
]

第一版只考虑：

```text
word-level units
+ adjacent-word edges
+ dependency-parse edges
+ second-order interactions
+ text classification
```

不考虑：

```text
GNN
paragraph/sentence hierarchy
generation tasks
RAG
cross-document interactions
third-order interactions
all-pairs exhaustive search
```

后续完整方法的初步形式是：

[
a_i^{(0)} = \sigma(w^\top h_i+b),
]

[
b_{ij}
======

\tanh(
\operatorname{MLP}
[
h_i;h_j;h_i\odot h_j;e_{ij}
]
),
]

其中：

* (a_i^{(0)})：原 AML 风格的初始节点 attribution；
* (b_{ij}\in[-1,1])：候选边 ((i,j)) 的 interaction score；
* (e_{ij})：边类型，例如 adjacent、dependency、adjacent+dependency；
* (h_i,h_j)：attribution backbone 中对应单元的 contextual representation。

然后用 interaction 对节点 mask 做一次 refinement：

[
z_i^{(0)}=\operatorname{logit}(a_i^{(0)}),
]

[
z_i^{(1)}
=========

z_i^{(0)}
+
\eta
\sum_{j:(i,j)\in E}
b_{ij}a_j^{(0)},
]

[
a_i=\sigma(z_i^{(1)}).
]

最终将 refined mask (a) 而非 (a^{(0)}) 输入 AML 原有的 dual-mask objective。

但是，当前阶段不要实现上述完整模型。当前首要任务是实施诊断实验，验证该方向是否有必要。

---

# 三、当前必须实现：AML interaction diagnostic experiment

请实现一个独立、可复现、尽量少侵入原 AML 代码的诊断实验。

核心研究问题：

> AML 是否在具有高局部二阶 interaction 的样本上，表现出更差的 attribution faithfulness？

该实验应尽量复用 AML 原始模型、mask operator、解释单位和评价协议，避免因为换了 intervention 定义而制造假象。

## 3.1 诊断实验的基本要求

对于每个测试样本：

1. 使用 AML 原实现得到 token-level attribution；
2. 构建一组候选词对；
3. 对候选词对计算局部二阶 interaction score (I_{ij})；
4. 将样本按 interaction strength 分桶，例如：

   * low interaction；
   * medium interaction；
   * high interaction；
5. 分别报告 AML explanation 在各桶中的 faithfulness；
6. 分析 high-interaction 样本是否明显更难解释；
7. 导出可人工检查的 case study，包括文本、AML top tokens、top interaction pairs、各类遮挡后的模型分数。

## 3.2 单元定义：必须处理 word 与 subword 的关系

AML 可能基于 subword tokenizer 工作，但诊断实验的可解释单位应优先设为自然词，而不是 BPE wordpiece。

请实现或复用可靠的 word-to-subword mapping：

```text
natural word
    -> one or more tokenizer subwords
```

规则：

* 一个自然词对应的全部 subword token 必须作为一个 group；
* 当遮挡该词时，遮挡其全部 subword；
* 特殊 token、padding token、BOS/EOS/CLS/SEP 不得成为候选词节点；
* 对无法可靠对齐的 tokenization 情况，必须记录并安全跳过或给出可解释回退策略；
* 输出中必须保存 word text、word index、对应 subword indices。

优先使用 Hugging Face tokenizer 的 offset mapping 或 word_ids；不要用基于字符串猜测的脆弱实现，除非仓库现有数据管线确实不支持 offset mapping，并在报告中明确说明限制。

## 3.3 候选边构建

第一版只构建两类词对：

### A. 相邻词边

对于相邻自然词：

[
(w_i,w_{i+1}).
]

### B. dependency parse 边

使用 spaCy 或 Stanza 为原始文本构建 dependency parse。对于每个非 root token，与其 head 构建无向边：

[
(w_i,w_{\mathrm{head}(i)}).
]

要求：

* 记录边类型：`adjacent`、`dependency`、`adjacent_dependency`；
* 去重；
* 排除特殊 token；
* 处理 parser 失败、空文本、单词文本、非英语文本等边界情况；
* 不要把 parser 输出当成真值，它只是 candidate proposal；
* 首先支持 AML 当前实验最常用的英语分类数据集；
* 若数据集文本并非英语，明确报错或在配置中禁用 dependency edges。

候选图应是稀疏的：

[
E = E_{\mathrm{adjacent}}\cup E_{\mathrm{dependency}},
\qquad |E|=O(n).
]

## 3.4 Interaction teacher 的严格定义

请先确认 AML 的 mask operator。

对于诊断主结果，interaction teacher 应优先使用：

> 与 AML fidelity evaluation 中相同的 masking / deletion operator。

例如：

* 若 AML evaluation 使用 mask embedding，则使用同一种 mask embedding；
* 若 AML evaluation 使用 `<MASK>` token，则使用同一 token；
* 若原实现支持 deletion，则将 deletion 作为额外 robustness check，而不是默认替换原协议。

令 (y^\star) 为原模型在未遮挡输入上的预测类别。

令 (s(x)) 为 (F(x)) 对 (y^\star) 的 score。优先级如下：

```text
1. 原始目标类别 logit；
2. 若代码无法稳定获得 logit，则使用目标类别 log probability；
3. 最后才使用目标类别 probability。
```

请在结果 metadata 中明确记录 score type。

对每条候选边 ((i,j))，计算：

[
I_{ij}
======

## s(x)

## s(x_{\setminus i})

s(x_{\setminus j})
+
s(x_{\setminus{i,j}}).
]

缓存：

[
s(x_{\setminus i})
]

因为同一词会出现在多条边中，避免重复 explained-model forward。

对于每个样本，至少保存：

```text
original_score
single_mask_score_i
single_mask_score_j
pair_mask_score_ij
interaction_score
normalized_interaction_score
edge_type
```

不要在 interaction score 上提前取绝对值；同时保存 signed score 与 absolute score。

## 3.5 如何定义一个样本的 interaction strength

请实现至少三种样本级汇聚方式，并在结果中都保存：

[
S_{\max}(x)=\max_{(i,j)\in E}|I_{ij}|,
]

[
S_{\mathrm{mean-topk}}(x)
=========================

\frac{1}{k}
\sum_{(i,j)\in \mathrm{TopK}(|I|)}
|I_{ij}|,
]

[
S_{\mathrm{energy}}(x)
======================

\frac{1}{|E|}
\sum_{(i,j)\in E} I_{ij}^2.
]

默认主分析使用：

[
S_{\mathrm{mean-topk}}, \quad k=3.
]

请允许通过配置修改 `k`。

样本分桶至少实现两种可选方式：

```text
1. quantile：按全测试集 interaction strength 的三分位数划分 low / medium / high；
2. fixed threshold：由用户传入阈值。
```

默认使用 quantile。

## 3.6 AML faithfulness 指标

请先从 AML 原实现中识别已有的 attribution evaluation metrics，并优先复用其代码和定义。

至少需要支持：

```text
1. Deletion / Leave-one-out faithfulness：
   按 AML attribution 从高到低逐步遮挡 top-k 单元，测目标类别分数下降曲线。

2. Sufficiency：
   仅保留 top-k attribution 单元时，原预测保持程度。

3. Comprehensiveness：
   遮挡 top-k attribution 单元时，原预测下降程度。
```

如果 AML 官方代码采用不同命名、不同 definition 或不同 unit，请保持它的原始协议，并在最终报告中解释。

要求：

* 诊断实验的 faithfulness 单位必须使用同一套自然词 group；
* 不允许将 parent sentence 与 child word 同时计入同一 evaluation ranking；
* 不允许把 special tokens 计入 top-k；
* 支持多个预算，例如 top 10%、20%、30%、50%；
* 至少输出每个样本的 AOPC / 曲线面积或等价的 aggregate score；
* 对每个 interaction bucket 分别汇总均值、标准差、样本数；
* 计算 high vs low bucket 的差异、bootstrap 95% CI 与非参数显著性检验，例如 Mann–Whitney U 或 permutation test；
* 不要只报告 p-value；同时报告 effect size，例如 Cliff’s delta 或 standardized mean difference。

如果原 AML 代码已有 LO、Sufficiency、Comprehensiveness 等实现，请优先调用或复制其核心函数，不要重新发明不同定义。

## 3.7 诊断实验的关键分析

请至少实现以下分析。

### 分析 A：interaction strength 与 AML faithfulness 的相关性

对每个样本计算：

```text
interaction_strength
faithfulness_score
```

报告：

```text
Spearman correlation
Pearson correlation
bootstrap confidence intervals
scatter plot / binned trend plot
```

这里要注意 faithfulness 指标的方向统一。例如：

* 分数越大越好；
* 或先转换成“faithfulness error”，使得越大越差。

必须在代码与文档中明确方向。

### 分析 B：high-interaction 与 low-interaction 分桶对比

比较：

```text
high interaction bucket
vs.
low interaction bucket
```

在以下指标上的差异：

```text
deletion faithfulness
sufficiency
comprehensiveness
AOPC
```

目标是验证：

[
\mathrm{Faithfulness}*{\mathrm{high}}
<
\mathrm{Faithfulness}*{\mathrm{low}}
]

或者 equivalently：

[
\mathrm{Error}_{\mathrm{high}}

>

\mathrm{Error}_{\mathrm{low}}.
]

但不要假设该结论必然成立。若结果不支持，也必须如实输出。

### 分析 C：结构边的价值与候选覆盖

比较三种候选边集合：

```text
1. adjacent only
2. dependency only
3. adjacent + dependency
```

并至少在一个小规模子集上加入：

```text
4. all-pairs exhaustive reference
```

all-pairs exhaustive reference 仅用于短文本或限制最大词数的小样本，例如：

```text
max_words <= 20
num_samples <= 100
```

目标是估计候选图对强交互的覆盖能力：

[
\mathrm{CandidateRecall@K}
==========================

\frac{
|\mathrm{TopK}*{\mathrm{all-pairs}}\cap E*{\mathrm{candidate}}|
}{
K
}.
]

至少报告 `Recall@1`、`Recall@3`、`Recall@5`。

这一步非常重要：如果 language graph 无法覆盖真实 top interactions，那么后续 interaction-aware AML 没有可靠基础。

### 分析 D：定性案例

为每个数据集至少导出：

```text
- 10 个 high-interaction 样本；
- 10 个 low-interaction 样本；
- 10 个 AML faithfulness 最差的样本；
- 10 个 AML faithfulness 最好的样本。
```

每个案例保存：

```json
{
  "id": "...",
  "text": "...",
  "true_label": ...,
  "predicted_label": ...,
  "original_target_score": ...,
  "words": [...],
  "word_attributions": [...],
  "top_attributed_words": [...],
  "candidate_edges": [...],
  "top_interactions_signed": [...],
  "top_interactions_absolute": [...],
  "interaction_strength": ...,
  "deletion_metrics": {...},
  "sufficiency_metrics": {...},
  "comprehensiveness_metrics": {...}
}
```

同时生成易读的 Markdown 或 HTML casebook，至少包含：

```text
文本
预测类别
AML attribution 高亮
top positive interaction pairs
top negative interaction pairs
原始分数、单词遮挡分数、pair 遮挡分数
high/medium/low interaction bucket
```

---

# 四、目录与代码组织要求

不要直接污染 AML 官方基线的核心训练文件，除非确实需要增加最小 hook。

请优先采用以下或等价的目录结构：

```text
baselines/
  aml-main_copy/
    ...

experiments/
  aml_interaction_diagnostic/
    README.md
    requirements_diagnostic.txt
    configs/
      diagnostic_sst2.yaml
      diagnostic_imdb.yaml
      diagnostic_eraser_movie_reviews.yaml
    scripts/
      run_diagnostic.py
      run_candidate_coverage.py
      generate_casebook.py
      aggregate_results.py
    src/
      aml_adapter.py
      model_adapter.py
      tokenizer_alignment.py
      word_units.py
      candidate_graph.py
      masking_adapter.py
      interaction_teacher.py
      faithfulness_metrics.py
      bucket_analysis.py
      statistics.py
      plotting.py
      io_utils.py
    tests/
      test_word_subword_alignment.py
      test_candidate_graph.py
      test_masking_equivalence.py
      test_interaction_finite_difference.py
      test_metric_direction.py
    outputs/
      .gitkeep
```

如果现有仓库具有更合理的实验结构，可以适配，但必须做到：

1. 原 AML baseline 可以独立运行；
2. 诊断代码可以独立运行；
3. 新代码不依赖隐式全局变量或 notebook state；
4. 所有实验参数通过 YAML 或 argparse 配置；
5. 随机种子、模型 checkpoint、数据集 split、mask operator、score type、tokenizer 名称都写入结果 metadata；
6. 长时间运行任务支持断点恢复或缓存；
7. all-pairs reference 与主诊断实验的输出严格分开；
8. 不要把大量中间结果只保存在内存中。

建议结果目录形如：

```text
outputs/
  <dataset>/
    <model_name>/
      <checkpoint_id>/
        <mask_operator>/
          <run_id>/
            config.yaml
            metadata.json
            per_example.jsonl
            per_edge.jsonl
            summary.json
            bucket_metrics.csv
            correlations.json
            candidate_coverage.json
            plots/
            casebook.md
```

---

# 五、严格对照 AML 的要求

核心原则：

> 诊断实验必须测量 AML 在其自身解释协议下的表现，而不是通过替换 mask、替换 score、替换单位或替换指标来人为制造 AML 的弱点。

因此：

1. 先识别 AML 代码中的真实 mask operator；
2. 主实验必须使用 AML 相同的 operator；
3. 先识别 AML 原始 attribution score 的方向；
4. 主实验必须保持相同方向；
5. 先识别 AML 原始 faithfulness metric；
6. 主实验必须复用该 metric 或精确复现；
7. word grouping 是为了让 interaction 单位自然可读，但必须明确说明这与 AML 原 token-level score 如何聚合；
8. 如果 AML 原生只有 subword attribution，则默认 word score 使用以下可配置聚合之一：

   * mean；
   * max；
   * sum；
   * first-subword；
     默认优先使用与 mask semantics 最一致的方式；
9. 每一种聚合策略都应记录在 metadata；
10. 请实现单元测试，验证“对一个 word group 进行 mask”在底层 token mask 上确实遮挡了该词对应的全部 subword，且不会影响其他 word group。

---

# 六、代码质量与工程要求

请遵守：

```text
- Python 3.10+；
- 明确类型标注；
- dataclass 或 pydantic 配置对象；
- 不使用隐藏全局状态；
- 每个关键函数都写 docstring，说明输入、输出、score direction、mask semantics；
- 所有模型 forward 使用 torch.no_grad()，除非确实处于 AML attribution model 训练步骤；
- teacher interaction 计算必须缓存原始和单词删除的分数；
- 使用 batch inference 加速可并行的 masked inputs；
- 对 OOM、空文本、parser error、token alignment error 给出可读日志；
- 任何跳过样本必须记录 skip reason；
- 小样本 smoke test 能在 CPU 上运行；
- 完整实验可以使用 GPU；
- 不修改 baseline 的随机行为，除非通过显式配置开关；
- 不要伪造结果，不要硬编码任何预期结论。
```

请优先使用仓库现有依赖；只有确有必要时才新增 `spacy`、`scipy`、`pandas`、`matplotlib` 等依赖，并在 requirements 文件中写明版本范围。

---

# 七、你必须先交付的内容

在写大量代码前，请先输出以下内容并等待我确认：

```text
A. AML 代码审计报告
B. 实现计划，按 Phase 0 / Phase 1 / Phase 2 划分
C. 建议新增和修改的文件清单
D. 每个文件的职责
E. 关键数据结构和张量形状
F. interaction teacher 的 batch/caching 方案
G. 预计 explained-model forward 数量与复杂度分析
H. 主实验与 all-pairs reference 的运行命令草案
I. 单元测试计划
J. 你识别到的任何不确定点或需要我决策的地方
```

其中 Phase 应定义为：

```text
Phase 0: 审计 AML、跑通原始 baseline、实现最小 smoke test
Phase 1: 实现 interaction diagnostic experiment
Phase 2: 仅在诊断结果支持研究假设后，再实现 interaction-aware AML
```

请不要在未完成 Phase 1 诊断、未输出结果前自行进入 Phase 2。

如果你认为某项设计与 AML 现有代码不兼容，请明确说明冲突位置，并提出“最小偏离 AML”的实现方案，而不是私自替换算法。
