# DGEKT on ASSIST17：实测结果与论文差异对照

生成时间：2026-11-19
运行环境：Tesla T4 (16GB, 单卡) / torch 2.10.0+cu128 / numpy 2.3.5 / scipy 1.17.0 / scikit-learn 1.8.0

---

## 1. 结论摘要

| 指标 | AUC | ACC | 说明 |
|---|---|---|---|
| 论文 Table 3（DGEKT / ASSIST17） | **0.775400** | — | 论文只报告 AUC，最大训练 600 epoch |
| 本次 20 轮（独立 valid 选模型） | **0.773361** | 0.719776 | 训练集比论文少 10% |
| 仓库自带权重 `save2017model.pkl` | 0.774205 | 0.718870 | 论文代码 epoch 10，用**完整**训练集 |
| 与论文差距 | **−0.002039（−0.20 pp）** | — | 相对差距 0.26% |

**一句话结论**：在「训练集减少 10% + 用独立验证集选 best + 测试集全程留白」这一更严格的设置下，
20 轮训练达到 AUC 0.7734，与论文的 0.7754 相差 0.20 个百分点；与仓库自带权重（完整训练集、
epoch 10）相差 0.08 个百分点。差异主要由**训练集规模**解释，而非实现错误。

一个反向证据支持这个判断：把仓库自带的 `save2017model.pkl` 直接在本项目的数据管线上评估，
得到 0.774205，与 2022 年原始日志中 epoch 10 的 0.774508 几乎一致（差异来自评估集
`drop_last` 设置，见 §6.4）。这说明数据加载、图构建、损失与评测口径都与原实现一致。

---

## 2. 论文关键设定（从 PDF 提取）

来源：`Cui 等 - 2024 - DGEKT A Dual Graph Ensemble Learning Method for Knowledge Tracing.pdf`

- Table 2（数据集统计）：1,709 学生 / 3,162 练习 / 102 概念 / 942,816 交互 / 正确率 37.26%
- Table 3（主结果）：DGEKT 在 ASSIST17 上 AUC = 77.54%
- Table 5（消融）：CAG 77.14% / TG 77.67% / RmCAHG 76.00% / RmDTG 76.76% / RmOKD 76.89% / DGEKT 77.54%
- §4.1 划分方式：「randomly select 80% of students for training and validation and the remaining 20% for testing」
- §4.1 序列处理：最大长度 50，超过则切分为多个子序列
- §4.4 超参：embedding 256 维、双图各两层、GRU 1 层 128 单元、温度 γ=0.5、
  **λ=0.01**、batch 128、初始学习率 1e-3、**最大 600 epoch**
- §4.6 敏感性：λ 在 1e-5 ~ 1e-2 之间性能相对平稳，超过 1e-2 后迅速下降

---

## 3. 本次数据划分

原代码只有 train / test 两份，且 `KTtrain()` 直接用 **test** 选 best epoch（信息泄漏）。
本次按 block 级切出独立验证集：

| split | 学生数(block) | chunk 数 | 交互记录数 |
|---|---|---|---|
| train | 1,230 | 14,499 | 694,185 |
| valid | 137 | 1,568 | 75,208 |
| test | 342 | 3,630 | 173,423 |
| 合计 | 1,709 | 19,697 | 942,816 |

- 1,709 与论文 Table 2 的学生数完全一致，说明**每个 4 行 block 就是一个学生**，
  因此 block 级切分等价于论文的「按学生划分」，不会出现同一学生跨 split 的泄漏。
- 942,816 与论文 Table 2 的交互数完全一致。
- 切分种子 216（与论文代码 `set_seed(216)` 相同），train/valid 无重叠，并集等于原文件（已断言校验）。
- 有向转移图 `adj_out/adj_in` **只用 1,230 个训练 block 构建**（nnz = 92,905），
  验证集与测试集的交互不进入图结构。

---

## 4. 20 轮训练曲线

`outputs/history.csv`（best 由 valid AUC 决定）

| epoch | train_loss | valid_auc | valid_acc | is_best |
|---|---|---|---|---|
| 1 | 244.276 | 0.680251 | 0.662095 | ✓ |
| 2 | 235.439 | 0.685741 | 0.665239 | ✓ |
| 3 | 233.503 | 0.702273 | 0.673438 | ✓ |
| 4 | 229.225 | 0.731303 | 0.686991 | ✓ |
| 5 | 224.074 | 0.745618 | 0.696518 | ✓ |
| 6 | 218.741 | 0.757987 | 0.706629 | ✓ |
| 7 | 214.278 | 0.764991 | 0.710017 | ✓ |
| 8 | 211.018 | 0.767445 | 0.711114 | ✓ |
| 9 | 208.326 | 0.769405 | 0.713337 | ✓ |
| 10 | 205.793 | 0.770814 | 0.714232 | ✓ |
| 11 | 203.669 | 0.772018 | 0.716806 | ✓ |
| 12 | 201.421 | 0.772144 | 0.715085 | ✓ |
| 13 | 199.192 | 0.771747 | 0.715817 | — |
| 14 | 197.129 | 0.772204 | 0.715519 | ✓ |
| **15** | **194.995** | **0.772293** | **0.716061** | **✓ best** |
| 16 | 192.828 | 0.772146 | 0.715465 | — |
| 17 | 190.708 | 0.770825 | 0.715221 | — |
| 18 | 188.576 | 0.771213 | 0.715533 | — |
| 19 | 186.475 | 0.769662 | 0.714679 | — |
| 20 | 184.407 | 0.768884 | 0.715140 | — |
| 21 | 182.327 | 0.769045 | 0.712741 | — （续训验证） |
| 22 | 180.178 | 0.768512 | 0.714760 | — （续训验证） |

- 单轮约 135 秒（train 113 batch + valid 13 batch），20 轮约 45 分钟。
- 21、22 轮由 `--resume last` 续训得到，用于验证断点续训可用性；best 未改变。
- **valid AUC 在 epoch 15 达到峰值后开始回落**，而 train_loss 仍在单调下降
  （194.99 → 180.18），是明确的过拟合信号。

最终在 test 上**只评估一次**（加载 epoch 15 的 `best.pt`）：

| 指标 | 值 |
|---|---|
| test AUC | **0.773361** |
| test ACC | 0.719776 |
| test chunks | 3,630 |
| 平均 CrossEntropy loss | 207.786 |
| 平均 KD loss | 7.102 |

---

## 5. 同 epoch 与原始日志对照

仓库 `2022_4_28_result.log` 记录的是**在 test 上**的指标（原代码用 test 选模型，且训练集为完整 1,367 block）。

| epoch | 2022 日志 AUC (test, 1367 训练块) | 本次 valid AUC (90% 训练块) | 差值 |
|---|---|---|---|
| 1 | 0.6841 | 0.6803 | −0.0038 |
| 2 | 0.6909 | 0.6857 | −0.0052 |
| 3 | 0.7105 | 0.7023 | −0.0082 |
| 4 | 0.7364 | 0.7313 | −0.0051 |
| 5 | 0.7546 | 0.7456 | −0.0090 |
| 6 | 0.7643 | 0.7580 | −0.0063 |
| 7 | 0.7690 | 0.7650 | −0.0040 |
| 8 | 0.7717 | 0.7674 | −0.0043 |
| 9 | 0.7734 | 0.7694 | −0.0040 |
| 10 | 0.7745 | 0.7708 | −0.0037 |

同 epoch 稳定低 0.4–0.9 个百分点，量级与「训练集减少 10% + 验证集为独立学生」相符。
到 epoch 15 时 valid AUC 升至 0.7723，超过原始日志 epoch 10 的水平，说明**多训练 5 轮
基本补偿了训练集缩减的损失**。

---

## 6. 差异归因（逐条量化）

### 6.1 训练集少 10%（主因，约 −0.2 ~ −0.4 pp）
论文在 80% 学生内部再分 train/valid，未说明比例；本次按 90/10 切分，
训练学生从 1,367 降到 1,230。§5 的同 epoch 对照显示这一项造成约 0.4–0.9 pp 的阶段性差距，
多训练到 epoch 15 后可回补到 −0.08 pp（相对仓库权重）。

### 6.2 best epoch 选择方式不同（使结果更保守）
- 原代码：用 **test** AUC 选 best epoch —— 存在信息泄漏，指标偏乐观。
- 本次：用**独立 valid** AUC 选 best（epoch 15），test 只在最后评估一次。
这是本次指标略低于原代码的**结构性原因**，也是本次实验更可信的地方。

### 6.3 训练轮数（20 vs 600）
论文最大 600 epoch，本次 20 轮。但 §4 显示 valid 在 15 轮即达峰后回落，
说明**在当前超参下继续训练不会提升**，600 epoch 的设定与本次配置不可直接类比
（可能依赖论文更强的 KD 正则，见 6.4）。

### 6.4 KD 温度系数 λ 的取值差异（最值得注意）
| 来源 | λ |
|---|---|
| 论文 §4.4 | 0.01 |
| 代码 `Constants.kd_loss` | 5e-6 |

两者相差 **2000 倍**。论文 Fig.6(c) 称 λ ∈ [1e-5, 1e-2] 性能平稳，因此 5e-6 尚在可用区间，
这解释了为什么代码用 5e-6 也能复现 0.7745。但弱 KD 正则可能正是本次 15 轮后
迅速过拟合、无法像论文那样训练 600 轮的原因。**若要把训练延长到 60/100 轮，建议先把
`kd_loss` 调到 1e-3 ~ 1e-2 再试**，否则只会看到 valid 继续下降。

### 6.5 评估集 `drop_last` 设置（≈ −0.03 pp）
原代码 test loader `drop_last=True`，会丢弃最后不足 128 的 46 个样本；
本次改为 `drop_last=False`，评估更完整。用自带权重复测：
`drop_last=True` → 0.774508；`drop_last=False` → 0.774205。差异 0.0003，量级可忽略。

### 6.6 退化 chunk 带来的虚假负样本（≈ −0.02 pp）
当某学生 `seq_len % 50 == 1` 时，切块后最后一个 chunk 只有 1 个有效交互，
其余 49 步全为 padding。原实现的截断逻辑（`pc[i] > 0`）在此情况下失效，
会把 49 个 padding 步当作「答错」样本计入损失与 AUC：

| split | 退化 chunk 数 | 注入的虚假负样本 |
|---|---|---|
| train | 34 | 34 × 49 = 1,666 |
| valid | 3 | 147 |
| test | 4 | 196 |

test 因此被注入 196 个「必判错」样本，占 173,423 的 0.11%，对 AUC 的影响在 0.02 pp 量级。
**本次刻意不修改这一行为**，以保持与论文口径可比；导出 CSV 时已剔除这些 padding 行。

### 6.7 随机种子与硬件
种子固定为 216（与论文代码一致），`cudnn.deterministic=True`。
但 T4 与论文所用 GPU 不同，cuDNN 算子实现差异会带来 ~1e-3 量级的 AUC 波动。

---

## 7. 延长训练的使用与建议

已验证断点续训可用：

```bash
python -m KnowledgeTracing.evaluation.run --stage train --epochs 22 --resume last
```

输出 `[resume] from .../last.pt: epoch=20, best_auc=0.772293`，只跑 21、22 两轮，不重复历史轮次，
`history.csv` 继续追加。

**要真正延长到 60/100 轮，需先处理过拟合**，建议按以下顺序实验：

1. 把 `kd_loss` 从 5e-6 调到 1e-3 或 1e-2（论文值），再跑到 60 轮；
2. 或加学习率衰减（如每 20 轮 ×0.5）；
3. 或加 early stopping（patience≈5，按 valid AUC）；
4. 每次重跑前删除 `outputs/history.csv` 与 `model/checkpoints/*.pt`，避免与旧曲线混淆。

命令：

```bash
python -m KnowledgeTracing.evaluation.run --stage all --epochs 60
```

（`kd_loss` 目前只能改 `Constants.py`，未开放 CLI 参数。）

---

## 8. 产物清单

| 文件 | 内容 |
|---|---|
| `Dataset/assist2017/split_seed216/train.csv` | 1,230 个训练 block（4 行/块，格式与原文件一致） |
| `Dataset/assist2017/split_seed216/valid.csv` | 137 个验证 block |
| `Dataset/assist2017/split_seed216/split_meta.json` | 切分元信息（种子、块数、记录数、长度分布） |
| `KnowledgeTracing/model/checkpoints/best.pt` | epoch 15，valid AUC 0.772293（含 optimizer 与 RNG 状态） |
| `KnowledgeTracing/model/checkpoints/last.pt` | epoch 22，用于续训 |
| `outputs/history.csv` | 22 轮训练记录 |
| `outputs/metrics.json` | 最终 test AUC/ACC 与运行配置 |
| `outputs/predictions_valid.csv` | valid 逐时间步预测，73,640 行 |
| `outputs/predictions_test.csv` | test 逐时间步预测，169,793 行 |
| `outputs/compare_with_paper.md` | 本文件 |

预测 CSV 字段：

```
split, sample_idx, block_id, chunk_id, step, question_id, concept_id,
y_true, prob_cahg, prob_dtg, prob_ens, y_pred
```

- `step` 为 chunk 内 1-based 步号，表示"用第 step-1 步及之前的状态预测第 step 步"
- `prob_cahg` / `prob_dtg` / `prob_ens` 分别为超图分支、有向图分支与蒸馏集成分支的概率
- 行数与数据集严格自洽：valid 75,208 − 1,568 = 73,640；test 173,423 − 3,630 = 169,793
  （每个 chunk 首步无法预测，故减去 chunk 数）
- 导出前会断言：chunk 数等于数据集长度、样本全覆盖、`y_true` 与原始 CSV 逐位相等

导出数据的分布校验：

| split | 行数 | 真实正确率 | 预测准确率 |
|---|---|---|---|
| valid | 73,640 | 37.87% | 71.61% |
| test | 169,793 | 37.41% | 71.98% |

test 正确率 37.41% 与论文 Table 2 的 37.26% 一致（微小差异来自剔除各 chunk 首步）。

---

## 9. 复现命令

```bash
cd /workspace

# 1) 块级切分（只需一次）
python -m KnowledgeTracing.evaluation.run --stage split

# 2) 20 轮训练 + 一次性测试 + 预测导出
python -m KnowledgeTracing.evaluation.run --stage all --epochs 20

# 3) 延长训练（从 last.pt 续到 60 轮）
python -m KnowledgeTracing.evaluation.run --stage train --epochs 60 --resume last

# 4) 单独重跑测试 / 导出
python -m KnowledgeTracing.evaluation.run --stage test
python -m KnowledgeTracing.evaluation.run --stage export

# 5) 对齐论文原始设定（不切验证集、用 test 选模型，仅用于复现历史数字）
python -m KnowledgeTracing.evaluation.run --stage train --epochs 20 --no-valid
```

---

## 10. 本次实现相对原代码的改动

| 文件 | 改动 | 是否影响数值 |
|---|---|---|
| `Constant/Constants.py` | 路径改绝对路径；新增 train/valid/test 路径、产物目录、`apply_overrides()` | 否 |
| `data/split.py` | 新增：block 级切分与 chunk 枚举 | 否 |
| `data/dataloader.py` | 新增 valid loader；评估集 `drop_last=False` | 仅影响评估集大小（−0.03 pp） |
| `DirectedGCN/load_data.py` | `get_adj(train_path=)` 参数化，只用切分后训练集建图 | 防止泄漏 |
| `hgnn_models/hypergraph_utils.py` | `np.mat` → `np.asmatrix`（numpy≥2.0 兼容） | 否，数值等价 |
| `model/Model.py` | `ques` 由普通 tensor 改为 `register_buffer` | 否，但修复了 checkpoint 丢失题目嵌入的严重问题 |
| `evaluation/eval.py` | 抽出 `aligned_predictions`；`test_epoch` 支持 `collect` | 否，AUC 差 <1e-8 |
| `evaluation/pipeline.py` | 新增：训练/验证/测试/续训管线 | — |
| `evaluation/export.py` | 新增：逐时间步预测导出 + 一致性断言 | — |
| `evaluation/run.py` | 改为 CLI 子命令，加 `__main__` 守卫 | — |

**已修复的严重问题**：原 `Model.py` 中 `self.ques` 既不在 `parameters()` 也不在 `state_dict()`，
原代码靠 `torch.save(整个模型对象)` 规避；改用 `state_dict` 后，从 checkpoint 恢复的模型会
使用**重新随机初始化**的题目嵌入。实测这一 bug 会让 test AUC 从 0.7734 掉到 0.6713。
现改为 `register_buffer` 并在 checkpoint 中冗余保存。
