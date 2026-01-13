# NeurIPS 投稿计划（ESM / steer_memory_4.5.3）

本计划以“当前仓库已有的实现与结果”为基础，目标是在 NeurIPS 投稿前把研究叙事、实验矩阵、代码可复现性与写作材料一次性做完。

> 说明：NeurIPS 每年具体 deadline 会变动，请以官网为准；下文用“倒排周计划”表达节奏。

---

## 1. 论文叙事（要讲清楚的 3 件事）

1) **问题**：在固定 token 预算下，模型的“试错/探索”会直接挤占最终输出预算；同时，单次 decoding 很难判断“此刻注入哪一个 steering 向量有用”。  
2) **方法**：离线挖掘记忆条目（Stage I–II），Stage III 构建检索索引（memory 存储 `<state key, delta>` 以及 ``right-state→null''）；在线用“检索 + 轻量 probing”在控制点选择注入，并用固定 probe 长度控制开销。  
3) **贡献点**（建议形成 2–3 个可被 reviewer 复述的 bullet）：  
   - 统一的“离线挖掘 → 选择 → 记忆 → 在线控制”的 pipeline（可复现、可扩展到多数据集）  
   - 一个明确的“固定开销 probing”策略（`probe_tokens` + `L` + gating：`min_sim`/`min_entries`/`tau_null`），把退化机制显式约束住  
   - 经验结论：在“离线产物稀疏/噪声”阶段，门控能避免退化；在“离线产物足够强”阶段才会出现稳定收益（需要用实验支撑）

---

## 2. 当前结果暴露的关键风险（必须在投稿前解决）

基于 `docs/实验结果与配置汇总.md` 的结论：

- **普通数据集在 T=256/512 下截断率偏高**，导致结果对“probe 开销”极其敏感；容易出现“ESM 变差”的 reviewer 直觉结论。主结果建议至少 `T>=1024`（最终更建议 `2048`）；数学类任务通常要 `4096`。  
- **memory 优势信号稀疏（p90=0）**：Ahat 校准差，轻易触发“低相似度乱用工具→退化”。  
- **部分离线 run 不完整**（缺 `memory/entries.jsonl`），会导致 eval-only 卡住/失败；投稿前必须把“可复现脚本链”做干净。  

投稿前的底线目标：

1) 在主任务（至少 GSM8K / SVAMP）上，ESM **不退化**（与 greedy 持平或更好），并且在足够大的 `T_max`（例如 `decode.max_new_tokens >= 512`）下稳定。  
2) 能解释“何时有收益/何时退化”的机制（用 diagnostics + case studies 支撑）。  

---

## 3. 实验路线图（建议的矩阵）

### 3.1 主结果（必须）

- 数据集：`gsm8k`（主） + `svamp`（副）  
- 预算：普通数据集建议 `decode.max_new_tokens: 1024/2048`；数学类建议 `4096` 起步  
- 方法：`greedy` vs `esm`  
- 产物：`tables/main_results_single.csv`、`tables/diagnostics_T*.csv`、`cases/cases_T*.md`

建议起点配置：

- 离线+在线一体：`configs/gsm8k_recommended_paper.yaml`

### 3.2 消融（必须）

在固定 `T_max`（建议 1024）跑：

- `online.variant`: `esm` / `no_memory` / `no_probing`
- Memory 形态：是否保留/选择 ``right-state→null'' 条目（保证“正确态不注入”）
- `min_sim`：例如 `0.15 / 0.20 / 0.25`（验证“门控阈值-收益/退化”的相变）

### 3.3 关键超参扫（建议做，但要控制规模）

优先扫这三个（最影响退化/开销）：

- `probe_tokens`：`4 / 8`
- `L`：`1 / 2 / 3`
- `k_retrieve × k_scale`：`k_retrieve=8/16/32` × `k_scale=0.5/1/2`

### 3.4 额外对照（加分项）

- 更强基线：budget 更大下的 greedy（排除“ESM 只是减少截断”的假收益）  
- 不同模型规模：例如 7B vs 14B（如果算力允许）  

---

## 4. 工程与可复现性（投稿前 checklist）

- 每个主实验/消融都要有：
  - 一条可复制的命令（含 `--config`、`--run-id`）
  - 一个明确的输出目录（`outputs/<run_name>/<run_id>/`）
  - `config_resolved.json` + `logs/run.log` + `tables/*.csv`
- 避免“同一 run_id 用不同 YAML 重跑不同 stage”导致产物混杂；如必须复用 run_id，请把当时的 YAML 另存到 `outputs/.../configs/`（手动或脚本化）。

---

## 5. 写作材料清单（建议先把产物路径固定下来）

建议论文里至少包含：

- 主表：固定 `T_max` 下的准确率（含 greedy / esm / ablations）
- 诊断表：截断率、probe overhead、tool/null 使用率（来自 `tables/diagnostics_T*.csv`）
- Case study：ESM 改善与退化各若干例（来自 `cases/cases_T*.md`）

输出路径约定已由代码生成：

- `outputs/<run>/tables/main_results_single.csv`
- `outputs/<run>/tables/diagnostics_T*.csv`
- `outputs/<run>/cases/cases_T*.md`

---

## 6. 倒排周计划（示例）

你可以把“NeurIPS 截止日”当作第 0 周：

- T-10 ~ T-8 周：稳定主 pipeline（普通数据集 @1024/2048 不退化），确定推荐配置；补齐 SVAMP
- T-8 ~ T-6 周：做 3×3 的关键超参扫（probe_tokens/L/min_sim 或 k_retrieve×k_scale×min_sim），锁定默认 online 策略
- T-6 ~ T-4 周：做消融（no_memory/no_probing + dense/pos-only memory），补齐诊断与案例
- T-4 ~ T-2 周：跑全量数据（主结论）、整理所有图表、写初稿并迭代
- T-2 ~ T-0 周：写作润色、补相关工作、补限界/失败案例、清理代码与复现说明

---

## 7. 风险与备选方案

- 若 ESM 在某任务上始终不稳定：把贡献重心转向“预算受限 probing 的退化机制与门控”+“什么时候能安全启用工具”的分析结论，并用更多诊断/案例支撑。
- 若 memory 信号始终稀疏：考虑增强 key 表征/层选择（例如更合理的 layer、或多层聚合的 key），并将其作为方法改进点与消融重点。
