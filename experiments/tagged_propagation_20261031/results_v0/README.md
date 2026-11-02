# Milestone V0：v0 数据训练、v0 测试

数据：`events_p4v2.jsonl` + `docs_p4_raw.jsonl`（`gen_docs.py` 单一提示词，四个角色同一提示词），
`build_pairs.py` 组装为 `p4_pairs/`，按事件切分 280 / 60 / 60（训练 / 验证 / 测试）。
所有数字为 seed 42 单次运行，测试集为 60 个测试事件。

| 模型 | 脚本 | dim1 F1 | dim2 F1 | B 排在 C 前 |
|---|---|---|---|---|
| 三通道 MPNet + 第 9、10 层后跨文档注意力 | `train_crossdoc.py` | 0.327 | 0.000 | — |
| 来源指针头 | `train_srcptr.py` | 0.456 | 0.046 | — |
| 成对交叉编码器 | `train_pairce.py` | 0.585 | 0.000 | 0.634 |
| 成对交叉编码器 + 精确匹配嵌入 + 类别加权 | `train_pairce.py`（默认参数） | **0.651** | **0.728** | **0.858** |

词面基线（`measure_signal.py` 的 T1，输入只保留 `p4_pairs/docs.jsonl` 中 split 为 test 的文档）：
B 排在 C 前 0.725（2-gram）/ 0.762（LCS）。

每个目录含 `summary_seed42.json`（测试指标、参数、耗时）和 `train_seed42.jsonl`（逐轮验证）。
检查点（约 450 MB）不入库，在 `/mnt/hdd4/dujddx/tagged_{crossdoc,srcptr,pairce,pairce_match}_p4_2026110{1,2}/best.pt`。

复现最佳模型：

```bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python train_pairce.py \
    --data p4_pairs --out /mnt/hdd4/dujddx/tagged_pairce_match_p4_20261102
```
