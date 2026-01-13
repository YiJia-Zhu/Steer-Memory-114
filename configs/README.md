# Config Index

本目录包含两类配置：
- **推荐/当前在跑**：优先使用下方列出的 YAML（字段与 `esm/config.py` 保持同步）。
- **历史/调参**：其余 YAML 主要用于记录旧实验或调参轨迹，可能包含已弃用字段；当前 loader 会忽略未知字段（见 `esm/config.py:_dc`），但建议复制后再按 `docs/实验运行指南.md` 的 schema 精简一遍。

## 常用入口

- 快速 smoke：`debug.yaml`
- GSM8K 小规模快速迭代：`gsm8k_recommended_small.yaml`
- GSM8K 论文/投稿起点：`gsm8k_recommended_paper.yaml`
- GSM8K eval-only（复用 step2 离线产物，1024/2048）：`gsm8k_eval_step2.yaml`
- 主实验起点（全量数据/默认较慢）：`main.yaml`
- 多数据集套件（每张 GPU 一个数据集进程）：`suite_tiny.yaml` / `suite_small.yaml`

说明：
- `decode.max_new_tokens` 太小常出现 `finish_reason=length` 截断，导致 Greedy/ESM 评测偏低；如需更真实精度，请把 `decode.max_new_tokens` 提到 `1024/2048`（数学类至少 `4096`）或使用更紧凑 prompt（如 `gsm8k_0shot_compact`）。
- 若你跑的是旧版配置（出现 `online.kappa` / `online.alpha_mult` / `offline_calibrate` 等），请以 `docs/实验运行指南.md` 为准更新。
