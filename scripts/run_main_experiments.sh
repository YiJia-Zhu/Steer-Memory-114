#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-editable (TOP)
# =========================
# GPUs to use for parallel runs (one job per GPU). Edit as needed.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
# GPUS="${GPUS:-0,1}"

# -------------------------
# Models to run
# Format: <model_key>|<name_or_path>|<tensor_parallel_size>|<max_num_seqs>
# -------------------------
MODEL_SPECS=(
  "ds_r1_qwen_1p5b|huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B|1|256"
  "qwen2p5_3b|huggingface_models/Qwen2.5-3B-Instruct|1|256"
  "ds_r1_qwen_7b|huggingface_models/DeepSeek-R1-Distill-Qwen-7B|1|128"
  "qwen2p5_7b|huggingface_models/Qwen2.5-7B-Instruct|1|128"
)

# -------------------------
# Datasets to run (must be supported by esm/data/loaders.py and available locally)
# Format:
#   <dataset>|<prompt_template>|<train_split>|<eval_split>|<max_train>|<max_eval>|<T_max/max_new_token>|<max_model_len>
#
# Notes:
# - T_max is a shorthand in this script for generation budget: decode.max_new_tokens (and offline_mine.max_new_tokens).
# - max_model_len is the vLLM context window cap: prompt tokens + generated tokens (affects KV cache/memory).
# - For long math reasoning datasets (e.g., aime/amc/aime25), set both to "__KEEP__" to follow configs/default.
#
# Use "__KEEP__" to keep the base-config value for that field.
# -------------------------
DATASET_SPECS=(
  "math500|math_0shot|test|test|100|400|16384|16384"
  "aime_2024|math_0shot|train|train|10|20|16384|16384"
  "amc23|math_0shot|test|test|10|30|16384|16384"
  "aime25|math_0shot|test|test|10|20|16384|16384"
  "gsm8k|gsm8k_0shot|train|test|100|null|2048|4096"
  "arc-c|arc_0shot|train|validation|100|null|1024|4096" # 1.12k 299
  "openbookqa|arc_0shot|train|validation|100|null|1024|4096" # 4k 500 
  "commonsense_qa|arc_0shot|train|validation|100|null|1024|4096" # 9k 1k
)
# DATASET_SPECS=(
#   "math500|math_0shot|test|test|10|10|16384|16384"
#   "aime_2024|math_0shot|train|train|10|10|16384|16384"
#   "amc23|math_0shot|test|test|10|10|16384|16384"
#   "aime25|math_0shot|test|test|10|10|16384|16384"
#   "gsm8k|gsm8k_0shot|train|test|10|10|2048|4096"
#   "arc-c|arc_0shot|train|validation|10|10|1024|4096" # 1.12k 299
#   "openbookqa|arc_0shot|train|validation|10|10|1024|4096" # 4k 500 
#   "commonsense_qa|arc_0shot|train|validation|10|10|1024|4096" # 9k 1k
# )

# -------------------------
# Runtime controls
# -------------------------

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
BASE_CFG="${BASE_CFG:-configs/default}"

# Outputs: each job uses run_name = <RUN_NAME_PREFIX>_<model_key>_<dataset_key>
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-main}"

# Shared run_id for this batch (helps grouping).
EXP_ID="${EXP_ID:-$(date +%Y%m%d_%H%M%S)}"

# If DRY_RUN=1, only generate configs and print the plan.
DRY_RUN="${DRY_RUN:-0}"

# Resume / listing controls.
# - RESUME_FROM: 0-based job index to start from (skips jobs < RESUME_FROM).
# - LIST_JOBS=1: print idx -> (model,dataset,run_name,run_id) and exit.
#
# Ordering is the nested-loop order in this script:
#   model -> dataset
# Index formula (0-based):
#   idx = m*n_datasets + d
# where (m,d) are 0-based indices into MODEL_SPECS and DATASET_SPECS.
RESUME_FROM="${RESUME_FROM:-${RESUME:-0}}"
LIST_JOBS="${LIST_JOBS:-0}"

# Use current python by default (assumes you're already in the easysteer env).
# If you prefer forcing conda-run, set USE_CONDA_RUN=1.
USE_CONDA_RUN="${USE_CONDA_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-easysteer}"

# Global optional overrides (leave empty to keep per-dataset / base config values).
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-}"

# =========================

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=(python)
if [[ "${USE_CONDA_RUN}" == "1" ]]; then
  PY=(conda run -n "${CONDA_ENV}" python)
fi

if [[ ! -f "${BASE_CFG}" ]]; then
  # Allow BASE_CFG=configs/default (no extension).
  if [[ -f "${BASE_CFG}.yaml" ]]; then
    BASE_CFG="${BASE_CFG}.yaml"
  elif [[ -f "${BASE_CFG}.yml" ]]; then
    BASE_CFG="${BASE_CFG}.yml"
  fi
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

n_models="${#MODEL_SPECS[@]}"
n_datasets="${#DATASET_SPECS[@]}"
total_jobs=$(( n_models * n_datasets ))
if [[ ! "${RESUME_FROM}" =~ ^[0-9]+$ ]]; then
  echo "[error] RESUME_FROM must be a non-negative integer, got: ${RESUME_FROM}" >&2
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
  local max_num_seqs="$5"
  local dataset="$6"
  local prompt_template="$7"
  local train_split="$8"
  local eval_split="$9"
  local max_train="${10}"
  local max_eval="${11}"
  local tmax="${12}"
  local max_model_len="${13}"

  env BASE_CFG="${BASE_CFG}" OUT_CFG="${out_cfg}" RUN_NAME="${run_name}" \
    MODEL_PATH="${model_path}" MODEL_TP="${tp}" \
    MAX_NUM_SEQS="${max_num_seqs}" \
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

max_num_seqs = _maybe_int(os.environ.get("MAX_NUM_SEQS", "__KEEP__"))
if max_num_seqs != "__KEEP__":
    if max_num_seqs is None:
        raise ValueError("MAX_NUM_SEQS cannot be null/none")
    raw.setdefault("model", {})
    raw["model"]["max_num_seqs"] = int(max_num_seqs)

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

jobs=()  # "IDX|RUN_NAME|CFG_PATH|CMD"
job_idx=0
if [[ "${LIST_JOBS}" == "1" ]]; then
  echo "[list_jobs] 1"
  echo "[ordering] model -> dataset"
  echo "[note] set EXP_ID to match previous run_id if needed"
  echo
fi
for mspec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model_key model_path model_tp model_max_num_seqs <<< "${mspec}"
  model_key="$(sanitize "${model_key}")"
  model_max_num_seqs="${model_max_num_seqs:-__KEEP__}"

  for dspec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r dataset prompt_template train_split eval_split max_train max_eval tmax max_model_len <<< "${dspec}"
    dataset_key="$(sanitize "${dataset}")"
    run_name="$(sanitize "${RUN_NAME_PREFIX}")_${model_key}_${dataset_key}"
    cfg_path="${OUT_CFG_DIR}/${run_name}.yaml"

    if [[ "${LIST_JOBS}" == "1" ]]; then
      printf "%5d | model=%s | dataset=%s | run_name=%s | run_id=%s\n" \
        "${job_idx}" "${model_key}" "${dataset_key}" "${run_name}" "${EXP_ID}"
      job_idx=$((job_idx + 1))
      continue
    fi

    if (( job_idx < RESUME_FROM )); then
      job_idx=$((job_idx + 1))
      continue
    fi

    write_cfg \
      "${cfg_path}" "${run_name}" "${model_path}" "${model_tp}" "${model_max_num_seqs}" \
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

    jobs+=( "${job_idx}|${run_name}|${cfg_path}|${cmd}" )
    job_idx=$((job_idx + 1))
  done
done

if [[ "${LIST_JOBS}" == "1" ]]; then
  echo
  echo "[n_jobs_total] ${total_jobs}"
  exit 0
fi

echo "[mode] main experiments"
echo "[base_cfg] ${BASE_CFG}"
echo "[run_name_prefix] ${RUN_NAME_PREFIX}"
echo "[exp_id] ${EXP_ID}"
echo "[stages] ${STAGES}"
echo "[methods] RUN_GREEDY=${RUN_GREEDY} RUN_ESM=${RUN_ESM}"
echo "[resume_from] ${RESUME_FROM} (0-based index)"
echo "[gpus] ${GPUS}"
echo "[n_models] ${n_models}"
echo "[n_datasets] ${n_datasets}"
echo "[n_jobs_total] ${total_jobs}"
echo "[n_jobs] ${#jobs[@]}"
echo "[out_cfg_dir] ${OUT_CFG_DIR}"
echo "[log_dir] ${LOG_DIR}"
echo

if (( ${#jobs[@]} == 0 )); then
  echo "[no jobs] RESUME_FROM=${RESUME_FROM} (n_jobs_total=${total_jobs})"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry_run] 1 (configs generated; jobs not started)"
  exit 0
fi

available_gpus=( "${GPU_LIST[@]}" )
declare -A pid_to_gpu=()
declare -A pid_to_name=()
failed=0
failed_jobs=()

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

    IFS='|' read -r idx name cfg cmd <<< "${job}"
    safe_name="$(printf "%05d__%s" "${idx}" "${name}")"
    start_job "${gpu}" "${safe_name}" "${cfg}" "${cmd}"
  done

  if [[ "${#pid_to_gpu[@]}" -gt 0 ]]; then
    done_pid=""
    rc=0
    if wait -n -p done_pid; then
      rc=0
    else
      rc=$?
    fi

    if [[ -n "${done_pid}" ]]; then
      gpu="${pid_to_gpu[${done_pid}]}"
      name="${pid_to_name[${done_pid}]}"
      unset pid_to_gpu["${done_pid}"]
      unset pid_to_name["${done_pid}"]
      available_gpus+=( "${gpu}" )
      if (( rc != 0 )); then
        failed=$((failed + 1))
        failed_jobs+=( "${name}|gpu=${gpu}|rc=${rc}" )
        echo "[fail][gpu=${gpu}][rc=${rc}] ${name}" >&2
      else
        echo "[done][gpu=${gpu}] ${name}"
      fi
    fi
  fi
done

echo
if (( failed > 0 )); then
  echo "[all done] exp_id=${EXP_ID} (failed=${failed}/${#jobs[@]})" >&2
  for j in "${failed_jobs[@]}"; do
    echo "  - ${j}" >&2
  done
  exit 1
fi
echo "[all done] exp_id=${EXP_ID}"
