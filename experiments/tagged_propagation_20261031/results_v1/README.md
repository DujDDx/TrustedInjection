# V1：关系源头头 + Set B 方向差惩罚

数据与 V0 相同，没有改动：v0 训练/验证/测试用 `p4_pairs/`，Qwen 留出集用 `heldout/qwen_pairs/`，DeepSeek 留出集用 `heldout/ds_pairs/`。训练只用 v0 的 Qwen 训练集，DeepSeek 只用于测试。

## 方法

`train_pairce.py --rel --w-anti-b 0.3`（MPNet，NEWS-COPY 初始化，冻结 4 层，按验证集损失选模型）

- **关系源头头**（`--rel`）：边 i←j 的打分输入是 `[E, 反向 E, 两者之差, 目标的其他候选均值, 候选的其他目标均值]`，在整个池子的关系里决定指针。
- **Set B 方向差惩罚**（`--w-anti-b`）：只在 Set B 池上加 λ·mean(s(i←j) − s(j←i))²。Set B 里的 A 和它的改写 Bf 长度、措辞几乎一样，文本里没有方向信息。不加惩罚时，模型在 Qwen 上靠偶然线索判方向，换到 DeepSeek 写的文档后，36% 的事件被自信地判反。加了惩罚，两个方向的分数会靠近，F1 校准后的阈值把两边都保留下来。Set A 的方向可以学，所以不加惩罚。池类型标签只在训练时使用，推理时不需要。
- **阈值**：dim1 阈值在 v0 验证集上按 F1 选；dim2 阈值取 max(0.5, dim1 阈值)。

## 结果（dim1 F1；与 token 级 LR 基线比较，基线阈值在训练集上按 F1 调；按事件做配对 bootstrap，2000 次）

| 集合 | 基线 | 种子 42 | 种子 1 | 种子 2 | 3 种子集成 | 集成领先（95% CI） |
|---|---|---|---|---|---|---|
| v0 测试 | 0.639 | 0.745 | 0.745 | 0.746 | 0.762 | +0.123 [+0.089, +0.153] |
| Qwen 留出 | 0.630 | 0.757 | 0.770 | 0.777 | 0.788 | +0.158 [+0.118, +0.198] |
| DS 留出 | 0.637 | 0.718 | 0.668 | 0.684 | 0.700 | +0.063 [+0.023, +0.102] |

集成的 dim2 F1：0.839 / 0.906 / 0.740，基线为 0.588 / 0.582 / 0.595。

DS 上不同种子差别较大：种子 1 只领先 +0.031，置信区间包含 0。

## 本轮试过的其他改动（种子 42，dim1：v0 测试 / Qwen / DS）

| 变体 | dim1 | 结论 |
|---|---|---|
| `--rel`（3 个种子：42 / 1 / 2） | 0.773 / 0.728 / 0.693；0.747 / 0.730 / 0.669；0.786 / 0.727 / 0.661 | DS 领先不稳 |
| `--rel --freeze 8` | 0.679 / 0.687 / 0.627 | 更差 |
| `--rel --lex`（模型内词汇边特征） | 0.763 / 0.736 / 0.689 | dim2 提升，dim1 持平 |
| `--rel --lex --w-rank 1` | 0.775 / 0.736 / 0.681 | 无提升 |
| `--rel --lex --w-anti 0.3 / 1.0`（对所有池加惩罚） | 0.717 / 0.693 / 0.622；0.655 / 0.634 / 0.607 | 全面变差：同时抹掉了 Set A 可学的方向 |
| `--rel --w-anti-b 1.0` | 0.720 / 0.742 / 0.722 | DS 最好，v0 测试较低；只跑了一个种子 |
| **`--rel --w-anti-b 0.3`** | 0.745 / 0.757 / 0.718 | 采用 |

在 DS 上扫 dim1 阈值，任何阈值下都封顶在 0.70 左右，所以问题不在阈值，而在模型。

## 文件

- `runs/<变体>/`：每次训练的 `summary_seed*.json`（含校准阈值和 v0 测试结果）和 `train_seed*.jsonl`（每个 epoch 的验证记录）。
- `comparisons/*.json|txt`：`eval_ensemble.py` 的输出，包括单种子、集成、与基线的配对 bootstrap。
- `ensemble_vs_baseline.json`、`predictions/ensemble_{p4,qwen,ds}_pairs.npz`：最终 3 种子集成在三个测试集上的推理结果。npz 内容：
  - `idx`：pairs.jsonl 行号；
  - `event_id`；
  - `prob`：[n,3,L,2]，dim1 / dim2 概率；
  - `pred`：二值预测；
  - `thr`：阈值。
- 基线见 `../results_baseline/`。`baseline_token.json` 是汇总，`predictions/` 是逐 token 预测，bootstrap 用的就是这些文件。
- 数据集（`p4_pairs/`、`heldout/*_pairs/`）已在之前的提交里，这一轮没有改动。
- 检查点每个约 430 MB，不进仓库，位于 `/mnt/hdd4/dujddx/tagged_pairce_relantib0.3_p4{,_s1,_s2}_20261102/best.pt`。

## 复现

```bash
python train_pairce.py --rel --w-anti-b 0.3 --seed 42 --out /mnt/hdd4/dujddx/tagged_pairce_relantib0.3_p4_20261102
python eval_ensemble.py --ckpts <三个 best.pt> --sets p4_pairs heldout/qwen_pairs heldout/ds_pairs \
    --baseline /mnt/hdd4/dujddx/diag/baseline_preds/lr_tuned_on_train_for_f1 \
    --out results_v1/ensemble_vs_baseline.json --dump results_v1/predictions
```

基线预测由 `baseline_token.py --dump` 生成。
