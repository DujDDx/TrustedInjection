# 提示词池训练与留出测试

## 数据

| 数据 | 事件 | 提示词 | 写作模型 | 位置 |
|---|---|---|---|---|
| v0 | 400（训练 280 / 验证 60 / 测试 60） | 四个角色同一提示词（p00） | Qwen | `docs_p4_raw.jsonl`、`p4_pairs/` |
| 提示词池训练 | 训练 + 验证 340 | 每个（事件, 分支）从训练提示词 p00–p11 中给 A、C、B、Bf 各分一个，互不相同 | Qwen | `pool_train/` |
| Qwen 留出 | 测试 50（缺最后 10 个，生成时截断报错中止） | 留出提示词 p12–p15，四个角色互不相同 | Qwen | `heldout/qwen_*` |
| DS 留出 | 测试 56 | 同上 | DeepSeek（只用于测试） | `heldout/ds_*` |

材料、真值和审核规则都与 v0 相同，另加泄漏审核（`gen_heldout.py` 里的 LEAK 规则）。
提示词在 `prompt_pool.py`，生成用 `gen_heldout.py`，评测用 `eval_heldout.py`。

## 结果（seed 42 单次运行）

| 测试集 | 训练数据 | dim1 F1 | dim2 F1 | B 排在 C 前 | 词面规则 2-gram / LCS |
|---|---|---|---|---|---|
| v0 测试 | v0 | 0.651 | 0.728 | **0.858** | 0.725 / 0.762 |
| v0 测试 | 提示词池 | **0.678** | **0.765** | 0.825 | 0.725 / 0.762 |
| Qwen 留出 | v0 | 0.654 | **0.711** | **0.847** | 0.837 / 0.832 |
| Qwen 留出 | 提示词池 | **0.665** | 0.690 | 0.837 | 0.837 / 0.832 |
| DS 留出 | v0 | 0.629 | 0.683 | **0.740** | 0.748 / 0.743 |
| DS 留出 | 提示词池 | **0.649** | **0.737** | 0.721 | 0.748 / 0.743 |

模型结果：`heldout/eval_pairce_match.json`（v0 训练）、`heldout/eval_pairce_pool.json`（提示词池训练）。
词面规则：`measure_signal.py` 的 T1，只在测试事件上计算。

## 结论

- 提示词池训练让逐词标签略好，但 B 排在 C 前在三个测试集上都降了 0.01–0.03；在 Qwen 留出上与词面规则打平，在 DS 留出上仍低于规则。
- 留出集上 A 和 C 用不同提示词，A–C 的 2-gram 重叠从 0.278 降到 0.235，而 A–B 基本不变（0.404 / 0.409），所以只看词重叠就能分对大部分 B/C。
- 主要失分在 Set B 的来源判断（0.54–0.68，Set A 为 0.76–0.87）；来源找对时判断值是否被改的正确率为 0.85–0.97。

`summary_seed42.json` 和 `train_seed42.jsonl` 是提示词池训练的参数和逐轮验证记录（验证损失最低在第 5 轮）。
检查点不入库，在 `/mnt/hdd4/dujddx/tagged_pairce_pool_p4_20261102/best.pt`。

```bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python train_pairce.py \
    --data pool_train/pairs --out /mnt/hdd4/dujddx/tagged_pairce_pool_p4_20261102
python eval_heldout.py --ckpt /mnt/hdd4/dujddx/tagged_pairce_pool_p4_20261102/best.pt \
    --sets p4_pairs heldout/qwen_pairs heldout/ds_pairs --out heldout/eval_pairce_pool.json
```
