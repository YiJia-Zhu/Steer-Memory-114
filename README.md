# Episodic Steering Memory (ESM) — 论文复现实验仓库

本目录用于复现论文草稿 `idea4.5.2.tex` 中提出的 **Episodic Steering Memory (ESM)**，并自动产出可直接填入论文的表格/图/案例数据。

## 快速导航

- 手动跑实验 / 配参 / 查结果：`docs/实验运行指南.md`
- 已有实验结果与配置汇总（含当前问题与结论）：`docs/实验结果与配置汇总.md`
- NeurIPS 投稿计划（倒排周计划 + 实验矩阵）：`docs/NeurIPS投稿计划.md`
- 配置索引（挑哪个 YAML 开跑）：`configs/README.md`
- 最近跑过的实验记录（含 run_id）：`RUNS.md`

## 目录结构

- `esm/`：ESM 代码实现（数据、离线三阶段、在线控制、评测与分析）
- `run.py`：统一入口（一个 CLI 覆盖全流程）
- `configs/`：YAML 配置（debug / main / ablation / suite）
- `outputs/`：所有输出集中在这里（logs/tables/figures/cases/eval）

## 分段/控制点规则

- 默认按空行（`\n\n`）切分段落；离线与在线都以 `control_points.segment_delimiter` 作为 stop 分隔符逐段生成。
- 控制点出现在每个“命中分隔符”的段落之后；离线与在线流程共享同一规则。
- 若某段在预算内未生成分隔符，则该样本会提前结束分段（后续控制点不会出现）。

## ESM 记忆条目格式（v4.5.3）

- **只使用 EasySteer/vLLM**：隐层提取与注入统一走 EasySteer（不再走 GGUF 路径）。
- **key/value**：检索 key 用隐层状态 `h`（不降维）；value 用原始向量差 `v = h_right - h_wrong`（不做归一化）。
- **双条目存储**：每个 `(m, layer)` 存两条：`<h_wrong_avg, v>` 与 `<h_right_avg, 0/null>`；在线检索到 `0/null` 会并入 null 分支。
- **注入强度**：在线 `Score_i = beta*Ahat_i + rho*(logprob_i - logprob_null)`，最终注入 `k_scale * Score_i * v_i`（`v_i` 保持 raw）。

## 环境说明（默认）

- Conda 环境：`easysteer`
- 常用写法：

```bash
conda run -n easysteer python run.py -h
conda run -n easysteer python run.py --config configs/debug.yaml eval
```

- 如果 `conda run` 太慢/不稳定，也可以直接用环境里的 python：

```bash
PY=${PY:-python}
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PY run.py --config configs/debug.yaml eval
```

GPU 使用约定：

- 单个进程默认只用一张卡（`model.tensor_parallel_size: 1`）
- 两张 GPU 的用法是“并行跑两个进程”（例如 A/B 两个配置，或 `run.py suite --gpus "0,1"` 一卡一个 dataset）

## 输出对齐（论文用）

每次运行都会写到 `outputs/<run_name>/<run_id>/`，常用产物包括：

- `tables/main_results_single.csv`、`tables/ablation.csv`
- `tables/diagnostics_T*.csv`、`cases/cases_T*.md`
