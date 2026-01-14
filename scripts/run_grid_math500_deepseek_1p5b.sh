# 先跑提取记忆的参数搜索
bash scripts/sweep_stage1_math500_deepseek_1p5b.sh

# 然后生成csv
python scripts/summarize_sweep_results.py --run-name gs_stage1_math500_deepseek_1p5b
# 人工选择最好的 stage1 run_id，并在 sweep_online 里设置 ARTIFACT_RUN_DIR（作为 mine/candidates.jsonl 的来源）
# sweep_online 会跑 select+memory+eval，并可同时扫 offline_select 与 online 的超参
bash scripts/sweep_online_math500_deepseek_1p5b.sh
