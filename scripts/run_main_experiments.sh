#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-editable (TOP)
# =========================

# GPUs to use for parallel runs (one job per GPU). Edit as needed.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# Stages to run (comma-separated). Default: full pipeline.
# Examples:
#   STAGES="eval"                              # eval-only
#   STAGES="mine,select,memory,eval"           # full pipeline
STAGES="${STAGES:-mine,select,memory,eval}"

# Which methods to run in eval.
# - RUN_GREEDY=0 can save time during ESM-heavy experiments.
# - RUN_ESM=0 makes it greedy-only (then STAGES="eval" is usually enough).
RUN_GREEDY="${RUN_GREEDY:-1}"
RUN_ESM="${RUN_ESM:-1}"

# Base config template. Per-job configs are generated from this file.
BASE_CFG="${BASE_CFG:-configs/main_experiments_base.yaml}"

# Outputs: each job uses run_name = <RUN_NAME_PREFIX>_<model_key>_<dataset_key>
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-main}"

# Shared run_id for this batch (helps grouping).
EXP_ID="${EXP_ID:-$(date +%Y%m%d_%H%M%S)}"

# If DRY_RUN=1, only generate configs and print the plan.
DRY_RUN="${DRY_RUN:-0}"

# Use current python by default (assumes you're already in the easysteer env).
# If you prefer forcing conda-run, set USE_CONDA_RUN=1.
USE_CONDA_RUN="${USE_CONDA_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-easysteer}"

# Global optional overrides (leave empty to keep per-dataset / base config values).
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-}"

# -------------------------
# Models to run
# Format: <model_key>|<name_or_path>|<tensor_parallel_size>
# -------------------------
MODEL_SPECS=(
  "ds_r1_qwen_1p5b|huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B|1"
  "qwen2p5_7b|huggingface_models/Qwen2.5-7B-Instruct|1"
  # "llama3p2_3b|huggingface_models/Llama-3.2-3B-Instruct|1"
)

# -------------------------
# Datasets to run (must be supported by esm/data/loaders.py and available locally)
# Format:
#   <dataset>|<prompt_template>|<train_split>|<eval_split>|<max_train>|<max_eval>|<T_max>|<max_model_len>
#
# Use "__KEEP__" to keep the base-config value for that field.
# -------------------------
DATASET_SPECS=(
  "gsm8k|gsm8k_0shot|train|test|__KEEP__|__KEEP__|2048|4096"
  "math500|math_0shot|test|test|__KEEP__|__KEEP__|4096|8192"
  "aime_2024|gsm8k_0shot|train|train|__KEEP__|__KEEP__|4096|8192"
  "arc-c|arc_0shot|train|validation|__KEEP__|__KEEP__|1024|4096"
  "openbookqa|arc_0shot|train|validation|__KEEP__|__KEEP__|1024|4096"
  "commonsense_qa|arc_0shot|train|validation|__KEEP__|__KEEP__|1024|4096"
)

# =========================

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=(python)
if [[ "${USE_CONDA_RUN}" == "1" ]]; then
  PY=(conda run -n "${CONDA_ENV}" python)
fi

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "[error] BASE_CFG not found: ${BASE_CFG}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -le 0 ]]; then
  echo "[error] Empty GPUS list: ${GPUS}" >&2
  exit 2
fi

IFS=',' read -r -a STAGE_LIST <<< "${STAGES}"
if [[ "${#STAGE_LIST[@]}" -le 0 ]]; then
  echo "[error] Empty STAGES: ${STAGES}" >&2
  exit 2
fi

OUT_CFG_DIR="${OUT_CFG_DIR:-configs/_main_generated/${EXP_ID}}"
LOG_DIR="${LOG_DIR:-logs/main_${EXP_ID}}"
mkdir -p "${OUT_CFG_DIR}" "${LOG_DIR}"

sanitize() {
  local s="$1"
  s="$(echo "${s}" | tr '[:upper:]' '[:lower:]')"
  s="${s//[^a-z0-9]/_}"
  s="${s//__/_}"
  s="${s#_}"
  s="${s%_}"
  echo "${s}"
}

write_cfg() {
  local out_cfg="$1"
  local run_name="$2"
  local model_path="$3"
  local tp="$4"
  local dataset="$5"
  local prompt_template="$6"
  local train_split="$7"
  local eval_split="$8"
  local max_train="$9"
  local max_eval="${10}"
  local tmax="${11}"
  local max_model_len="${12}"

  env BASE_CFG="${BASE_CFG}" OUT_CFG="${out_cfg}" RUN_NAME="${run_name}" \
    MODEL_PATH="${model_path}" MODEL_TP="${tp}" \
    DATASET="${dataset}" PROMPT_TEMPLATE="${prompt_template}" TRAIN_SPLIT="${train_split}" EVAL_SPLIT="${eval_split}" \
    MAX_TRAIN="${max_train}" MAX_EVAL="${max_eval}" TMAX="${tmax}" MAX_MODEL_LEN="${max_model_len}" \
    RUN_GREEDY="${RUN_GREEDY}" RUN_ESM="${RUN_ESM}" \
    GLOBAL_MAX_TRAIN="${MAX_TRAIN_EXAMPLES}" GLOBAL_MAX_EVAL="${MAX_EVAL_EXAMPLES}" \
    "${PY[@]}" - <<'PY'
import os
from pathlib import Path

import yaml

base_cfg = Path(os.environ["BASE_CFG"])
out_cfg = Path(os.environ["OUT_CFG"])

with base_cfg.open("r", encoding="utf-8") as f:
    raw = yaml.safe_load(f) or {}
if not isinstance(raw, dict):
    raise TypeError(f"Config root must be a dict, got {type(raw)}")

raw.setdefault("outputs", {})
raw["outputs"]["run_name"] = os.environ["RUN_NAME"]

raw.setdefault("model", {})
raw["model"]["name_or_path"] = os.environ["MODEL_PATH"]
tp_s = str(os.environ.get("MODEL_TP", "__KEEP__")).strip()
if tp_s and tp_s != "__KEEP__":
    raw["model"]["tensor_parallel_size"] = int(tp_s)

raw.setdefault("task", {})
raw["task"]["dataset"] = os.environ["DATASET"]
raw["task"]["train_split"] = os.environ["TRAIN_SPLIT"]
raw["task"]["eval_split"] = os.environ["EVAL_SPLIT"]

def _maybe_int(s: str):
    s = (s or "").strip()
    if s == "" or s == "__KEEP__":
        return "__KEEP__"
    if s.lower() in {"none", "null"}:
        return None
    return int(s)

# Global override (env) > per-dataset > base config
g_train = _maybe_int(os.environ.get("GLOBAL_MAX_TRAIN", ""))
g_eval = _maybe_int(os.environ.get("GLOBAL_MAX_EVAL", ""))
ds_train = _maybe_int(os.environ.get("MAX_TRAIN", "__KEEP__"))
ds_eval = _maybe_int(os.environ.get("MAX_EVAL", "__KEEP__"))

if g_train != "__KEEP__":
    raw["task"]["max_train_examples"] = g_train
elif ds_train != "__KEEP__":
    raw["task"]["max_train_examples"] = ds_train

if g_eval != "__KEEP__":
    raw["task"]["max_eval_examples"] = g_eval
elif ds_eval != "__KEEP__":
    raw["task"]["max_eval_examples"] = ds_eval

raw.setdefault("prompt", {})
raw["prompt"]["template"] = os.environ["PROMPT_TEMPLATE"]

raw.setdefault("decode", {})
tmax = _maybe_int(os.environ.get("TMAX", "__KEEP__"))
if tmax != "__KEEP__":
    raw["decode"]["max_new_tokens"] = int(tmax) if tmax is not None else None
    # By default, align mining rollout budget with eval budget for better segment/CP coverage.
    raw.setdefault("offline_mine", {})
    raw["offline_mine"]["max_new_tokens"] = int(tmax) if tmax is not None else None

max_model_len = _maybe_int(os.environ.get("MAX_MODEL_LEN", "__KEEP__"))
if max_model_len != "__KEEP__":
    raw.setdefault("model", {})
    raw["model"]["max_model_len"] = int(max_model_len) if max_model_len is not None else None

# Eval methods toggle.
methods = []
if str(os.environ.get("RUN_GREEDY", "1")).strip() not in {"0", "false", "False"}:
    methods.append("greedy")
if str(os.environ.get("RUN_ESM", "1")).strip() not in {"0", "false", "False"}:
    methods.append("esm")
if not methods:
    raise ValueError("Both RUN_GREEDY and RUN_ESM are disabled; nothing to run.")

raw.setdefault("eval", {})
raw["eval"]["methods"] = methods
raw["eval"]["ablations"] = []

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
PY
}

jobs=()  # "RUN_NAME|CFG_PATH|CMD"
for mspec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model_key model_path model_tp <<< "${mspec}"
  model_key="$(sanitize "${model_key}")"

  for dspec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r dataset prompt_template train_split eval_split max_train max_eval tmax max_model_len <<< "${dspec}"
    dataset_key="$(sanitize "${dataset}")"
    run_name="$(sanitize "${RUN_NAME_PREFIX}")_${model_key}_${dataset_key}"
    cfg_path="${OUT_CFG_DIR}/${run_name}.yaml"

    write_cfg \
      "${cfg_path}" "${run_name}" "${model_path}" "${model_tp}" \
      "${dataset}" "${prompt_template}" "${train_split}" "${eval_split}" \
      "${max_train}" "${max_eval}" "${tmax}" "${max_model_len}"

    cmd="cd \"${REPO_ROOT}\""
    for st in "${STAGE_LIST[@]}"; do
      st="$(echo "${st}" | xargs)"
      if [[ -z "${st}" ]]; then
        continue
      fi
      cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${EXP_ID}\" ${st}"
    done

    jobs+=( "${run_name}|${cfg_path}|${cmd}" )
  done
done

echo "[mode] main experiments"
echo "[base_cfg] ${BASE_CFG}"
echo "[run_name_prefix] ${RUN_NAME_PREFIX}"
echo "[exp_id] ${EXP_ID}"
echo "[stages] ${STAGES}"
echo "[methods] RUN_GREEDY=${RUN_GREEDY} RUN_ESM=${RUN_ESM}"
echo "[gpus] ${GPUS}"
echo "[n_models] ${#MODEL_SPECS[@]}"
echo "[n_datasets] ${#DATASET_SPECS[@]}"
echo "[n_jobs] ${#jobs[@]}"
echo "[out_cfg_dir] ${OUT_CFG_DIR}"
echo "[log_dir] ${LOG_DIR}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry_run] 1 (configs generated; jobs not started)"
  exit 0
fi

available_gpus=( "${GPU_LIST[@]}" )
declare -A pid_to_gpu=()
declare -A pid_to_name=()

start_job() {
  local gpu="$1"
  local name="$2"
  local cfg="$3"
  local cmd="$4"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[start][gpu=${gpu}] ${name}"
  echo "  cfg: ${cfg}"
  echo "  log: ${log_path}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec bash -lc "${cmd}"
  ) >"${log_path}" 2>&1 &

  local pid="$!"
  pid_to_gpu["${pid}"]="${gpu}"
  pid_to_name["${pid}"]="${name}"
}

queue=( "${jobs[@]}" )
while [[ "${#queue[@]}" -gt 0 || "${#pid_to_gpu[@]}" -gt 0 ]]; do
  while [[ "${#queue[@]}" -gt 0 && "${#available_gpus[@]}" -gt 0 ]]; do
    job="${queue[0]}"
    queue=( "${queue[@]:1}" )

    gpu="${available_gpus[0]}"
    available_gpus=( "${available_gpus[@]:1}" )

    IFS='|' read -r name cfg cmd <<< "${job}"
    start_job "${gpu}" "${name}" "${cfg}" "${cmd}"
  done

  if [[ "${#pid_to_gpu[@]}" -gt 0 ]]; then
    if ! wait -n; then
      echo "[error] a job failed; killing remaining jobs..." >&2
      for pid in "${!pid_to_gpu[@]}"; do
        kill "${pid}" 2>/dev/null || true
      done
      exit 1
    fi

    for pid in "${!pid_to_gpu[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        gpu="${pid_to_gpu[${pid}]}"
        name="${pid_to_name[${pid}]}"
        unset pid_to_gpu["${pid}"]
        unset pid_to_name["${pid}"]
        available_gpus+=( "${gpu}" )
        echo "[done][gpu=${gpu}] ${name}"
      fi
    done
  fi
done

echo
echo "[all done] exp_id=${EXP_ID}"
