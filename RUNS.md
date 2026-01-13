# 实验记录（含 run_id）与恢复指南

## 环境

- Conda 环境（已配置好）：`easysteer`
- 数据根目录：`datasets`
- 仓库路径：`steer_memory_4.5.3/`（本目录）

## 当前实现要点（v4.5.3）

- 离线 Stage I（`mine`）：对每个 `(control_point_m, layer)` 计算均值隐层 `h_wrong_avg / h_right_avg`，并存两条条目：
  - `<h_wrong_avg, v>`，其中 `v = h_right_avg - h_wrong_avg`（raw，不做归一化）
  - `<h_right_avg, 0/null>`（用于“命中正确态时不注入”，在线合并到 null 分支）
- Stage II（`select`）：质量 `Q = \\bar{R}_P - \\bar{R}_N`；多样性用 key 隐层的 Gram 矩阵（实现里对 key 做 L2 normalize 后点积，相当于 cosine）。
- Stage III（`memory`）：纯索引构建，输出 `memory/keys.npy` + `memory/entries.jsonl`。
- Online（`eval`/`case`）：
  - 每步固定 `online.probe_tokens` 做 probing（每个候选 + null），不再有 `probe_budget_frac`/全局 probe budget。
  - `Score_i = beta*Ahat_i + rho*(logprob_i - logprob_null)`，最终注入 `k_scale * Score_i * v_i`。
- 4.5.2 旧产物不兼容：需要用当前代码重跑 `mine → select → memory` 再 `eval`。

## 常用复现/继续

- 全流程（建议顺序）：
  - `conda run -n easysteer python run.py --config <cfg.yaml> mine`
  - `conda run -n easysteer python run.py --config <cfg.yaml> --run-id latest select`
  - `conda run -n easysteer python run.py --config <cfg.yaml> --run-id latest memory`
  - `conda run -n easysteer python run.py --config <cfg.yaml> --run-id latest eval`
- 只跑 eval 且复用离线产物：在 eval-only 配置里设置 `eval.artifact_run_dir: "outputs/<离线run_name>/latest"`。

## 历史记录（4.5.2，已过时）

- 旧版包含 `calibrate`/`tau_probe`/投影索引等实现与产物；当前版本已删除对应代码与配置字段。
