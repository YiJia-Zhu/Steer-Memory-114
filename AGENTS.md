# 仓库工作指南（给开发/助手）

## 项目结构与模块组织

- `esm/`：核心 Python 包
  - `esm/data/`：数据集加载 + prompt 模板
  - `esm/offline/`：离线 Stage I–III（mine/select/memory）
  - `esm/online/`：在线 ESM 解码/控制器
  - `esm/eval/`：抽取器、指标与预算工具
  - `esm/analysis/`：从运行产物生成表/图/案例
  - `esm/utils/`：I/O 与日志等工具
  - `esm/entrypoints/`：`run.py` 子命令背后的实现
- `run.py`：统一 CLI 入口
- `configs/`：YAML 配置（从 `configs/debug.yaml` 开始）
- `outputs/`：按 `run_name/run_id` 存放运行产物（不要手改）

## 构建、测试与开发命令

- CLI 帮助：`python run.py -h`
- 运行单个 stage：`python run.py --config configs/main.yaml mine|select|memory|eval`
- 快速 smoke（小限制）：`python run.py --config configs/debug.yaml eval`
- 恢复最新一次：`python run.py --config configs/main.yaml --run-id latest eval`
- 多数据集 suite（一卡一个数据集进程）：`python run.py --config configs/suite_small.yaml suite --gpus "0,1"`
- 语法检查（不需要 GPU）：`python -m compileall esm run.py`

建议在 conda 环境 `easysteer` 中运行（EasySteer/vLLM 依赖在其中），例如：`conda run -n easysteer python run.py ...`。多数 stage 需要 GPU；设置 `CUDA_VISIBLE_DEVICES` 并保持 `TOKENIZERS_PARALLELISM=false`。

## 代码风格与命名

- Python：4 空格缩进，鼓励 type hints，函数/模块用 `snake_case`，类用 `CamelCase`。
- 优先配置驱动：新增超参优先加到 `esm/config.py` 并通过 YAML 串起来，避免硬编码路径或魔法数。
- 保持可复现：尊重 `cfg.seed`，并将产物写到 `outputs/<run_name>/<run_id>/`。

## 配置与可复现性

- 配置 schema 在 `esm/config.py`；每次运行会写 `outputs/<run_name>/<run_id>/config_resolved.json` 便于精确复现。

## 测试建议

本仓库没有单独的单测套件。建议用以下方式做最小验证：

- `python -m compileall esm run.py`
- `python run.py --config configs/debug.yaml eval`，并检查 `outputs/.../config_resolved.json`、`outputs/.../logs/run.log`、`outputs/.../tables/`。

如需新增测试：使用 `pytest`，放在 `tests/` 下命名为 `test_*.py`，尽量保持 CPU-only。

## 提交与 PR 约定

此 checkout 不包含 Git 历史，建议使用一致的提交格式，例如 `type(scope): summary`（如：`fix(eval): handle empty budget rows`）。

PR 建议包含：运行命令/配置、预期输出路径，以及（如相关）链接到 `outputs/tables/*.csv` 或 `outputs/figures/*.pdf`。除非明确需要，否则避免提交大体量 `outputs/` 目录。
