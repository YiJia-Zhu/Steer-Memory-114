#!/usr/bin/env bash
set -euo pipefail
set -m

# =========================
# User-editable (TOP)
# =========================
# GPUs to use for parallel runs. One job will reserve tensor_parallel_size GPUs.
GPUS="${GPUS:-5}"
# GPUS="${GPUS:-0,1}"

# -------------------------
# Models to sweep
# Format: <model_key>|<name_or_path>|<tensor_parallel_size>|<max_num_seqs>
# -------------------------
MODEL_SPECS=(
  # "ds_r1_qwen_1p5b|huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B|1|512"
  # "qwen2p5_3b|huggingface_models/Qwen2.5-3B-Instruct|1|512"
  "ds_r1_qwen_7b|huggingface_models/DeepSeek-R1-Distill-Qwen-7B|1|256"
  "qwen2p5_7b|huggingface_models/Qwen2.5-7B-Instruct|1|256"
)

# -------------------------
# Datasets to sweep (must be supported by esm/data/loaders.py and available locally)
# Format:
#   <dataset>|<prompt_template>|<train_split>|<eval_split>|<max_train>|<max_eval>|<T_max>|<max_model_len>
#
# Notes:
# - T_max is a shorthand for generation budget: decode.max_new_tokens (and offline_mine.max_new_tokens).
# - max_model_len is the vLLM context window cap: prompt tokens + generated tokens.
# - Use "__KEEP__" to keep BASE_CFG values for that field.
# -------------------------
DATASET_SPECS=(
  "math500|math_0shot|test|test|100|400|16384|16384"
  "aime_2024|math_0shot|train|train|10|20|16384|16384"
  "amc23|math_0shot|test|test|10|30|16384|16384"
  "aime25|math_0shot|test|test|10|20|16384|16384"
  "arc-c|arc_0shot|train|validation|100|null|1024|4096" # 1.12k 299
  "openbookqa|arc_0shot|train|validation|100|null|1024|4096" # 4k 500 
  # "gsm8k|gsm8k_0shot|train|test|100|null|2048|4096"
  # "commonsense_qa|arc_0shot|train|validation|100|null|1024|4096" # 9k 1k
)

# DATASET_SPECS=(
#   # "math500|math_0shot|test|test|10|10|16384|16384"
#   # "aime_2024|math_0shot|train|train|10|10|16384|16384"
#   # "amc23|math_0shot|test|test|10|10|16384|16384"
#   # "aime25|math_0shot|test|test|10|10|16384|16384"
#   # "gsm8k|gsm8k_0shot|train|test|10|10|2048|4096"
#   # "arc-c|arc_0shot|train|validation|10|10|1024|4096" # 1.12k 299
#   # "openbookqa|arc_0shot|train|validation|10|10|1024|4096" # 4k 500 
#   "commonsense_qa|arc_0shot|train|validation|10|10|1024|4096" # 9k 1k
# )
# -------------------------
# Grid to sweep (3 params)
# Total points = |MODEL_SPECS| * |DATASET_SPECS| * |LAYER_LIST| * |ETA0_LIST| * |KSCALE_LIST|
# -------------------------

# 提取/注入的层数：支持比例（适配不同模型层数），如 1/5 或 0.6（注意：整数如 1 会被当作绝对层号 1）
LAYER_LIST=(0.6)
# Reward 函数中回答长度权重
ETA0_LIST=(0.01)
# 注入强度
KSCALE_LIST=(0.2 0.4)

# LAYER_LIST=(0.8)
# # Reward 函数中回答长度权重
# ETA0_LIST=(0.1)
# # 注入强度
# KSCALE_LIST=(1.0)

# -------------------------
# Runtime controls
# -------------------------

# Which methods to run in eval.
# - Default for sweeps: only ESM (greedy + ablations are expensive and usually unnecessary for grid search).
RUN_GREEDY="${RUN_GREEDY:-0}"
RUN_ESM="${RUN_ESM:-1}"
# If RUN_ABLATIONS=1, keep BASE_CFG's cfg.eval.ablations; otherwise force no ablations.
RUN_ABLATIONS="${RUN_ABLATIONS:-0}"

# Safety cap: refuse to run if grid size exceeds this.
MAX_GRID="${MAX_GRID:-4096}"

# Base config (all other params follow this file).
BASE_CFG="${BASE_CFG:-configs/default}"

# Outputs: each job uses run_name = <RUN_NAME_PREFIX>_<model_key>_<dataset_key>
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-grid}"

# Sweep id used to prefix per-run run_id.
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"

# If DRY_RUN=1, only generate per-run configs and print the plan.
DRY_RUN="${DRY_RUN:-0}"

# Resume / listing controls.
# - RESUME_FROM: 0-based job index to start from (skips jobs < RESUME_FROM).
# - LIST_JOBS=1: print idx -> (model,dataset,layer,eta0,k_scale,run_name,run_id) and exit.
#
# Ordering is the nested-loop order in this script:
#   model -> dataset -> layer -> eta0 -> k_scale
# Index formula (0-based):
#   idx = ((((m*n_datasets + d)*n_layers + l)*n_eta0 + e)*n_k + k)
# where (m,d,l,e,k) are 0-based indices into:
#   MODEL_SPECS, DATASET_SPECS, LAYER_LIST, ETA0_LIST, KSCALE_LIST.
RESUME_FROM="${RESUME_FROM:-${RESUME:-0}}"
LIST_JOBS="${LIST_JOBS:-0}"

# Use current python by default (assumes you're already in the easysteer env).
# If you prefer forcing conda-run, set USE_CONDA_RUN=1.
USE_CONDA_RUN="${USE_CONDA_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-easysteer}"

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

n_models="${#MODEL_SPECS[@]}"
n_datasets="${#DATASET_SPECS[@]}"
if (( n_models <= 0 )); then
  echo "[error] Empty MODEL_SPECS" >&2
  exit 2
fi
if (( n_datasets <= 0 )); then
  echo "[error] Empty DATASET_SPECS" >&2
  exit 2
fi
if (( ${#LAYER_LIST[@]} <= 0 )); then
  echo "[error] Empty LAYER_LIST" >&2
  exit 2
fi
if (( ${#ETA0_LIST[@]} <= 0 )); then
  echo "[error] Empty ETA0_LIST" >&2
  exit 2
fi
if (( ${#KSCALE_LIST[@]} <= 0 )); then
  echo "[error] Empty KSCALE_LIST" >&2
  exit 2
fi

grid_size=$(( n_models * n_datasets * ${#LAYER_LIST[@]} * ${#ETA0_LIST[@]} * ${#KSCALE_LIST[@]} ))
if (( grid_size > MAX_GRID )); then
  echo "[error] grid too large: ${grid_size} > MAX_GRID=${MAX_GRID}" >&2
  exit 2
fi

if [[ ! "${RESUME_FROM}" =~ ^[0-9]+$ ]]; then
  echo "[error] RESUME_FROM must be a non-negative integer, got: ${RESUME_FROM}" >&2
  exit 2
fi

OUT_CFG_DIR="${OUT_CFG_DIR:-configs/_grid_full_generated/${SWEEP_ID}}"
LOG_DIR="${LOG_DIR:-}"

OUTPUTS_ROOT="$(
  BASE_CFG="${BASE_CFG}" "${PY[@]}" - <<'PY'
import os
from pathlib import Path

import yaml

base_cfg = Path(os.environ["BASE_CFG"])
raw = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
root = (raw.get("outputs") or {}).get("root_dir") or "outputs"
print(Path(root).expanduser())
PY
)"
if [[ -z "${OUTPUTS_ROOT}" ]]; then
  OUTPUTS_ROOT="outputs"
fi
if [[ "${OUTPUTS_ROOT}" != /* ]]; then
  OUTPUTS_ROOT="${REPO_ROOT}/${OUTPUTS_ROOT}"
fi

mkdir -p "${OUT_CFG_DIR}"
if [[ -n "${LOG_DIR}" ]]; then
  mkdir -p "${LOG_DIR}"
fi

sanitize() {
  local s="$1"
  s="$(echo "${s}" | tr '[:upper:]' '[:lower:]')"
  s="${s//[^a-z0-9]/_}"
  s="${s//__/_}"
  s="${s#_}"
  s="${s%_}"
  echo "${s}"
}

f_tag() {
  # token -> tag-friendly string (safe for filenames/run_ids)
  local s="$1"
  s="${s//-/m}"
  s="${s//./p}"
  s="${s//\//d}"
  echo "${s}"
}

write_cfg() {
  local out_cfg="$1"
  local run_name="$2"
  local model_path="$3"
  local model_tp="$4"
  local max_num_seqs="$5"
  local dataset="$6"
  local prompt_template="$7"
  local train_split="$8"
  local eval_split="$9"
  local max_train="${10}"
  local max_eval="${11}"
  local tmax="${12}"
  local max_model_len="${13}"
  local LAYER="${14}"
  local ETA0="${15}"
  local KS="${16}"

  env BASE_CFG="${BASE_CFG}" OUT_CFG="${out_cfg}" RUN_NAME="${run_name}" \
    MODEL_PATH="${model_path}" MODEL_TP="${model_tp}" \
    MAX_NUM_SEQS="${max_num_seqs}" \
    DATASET="${dataset}" PROMPT_TEMPLATE="${prompt_template}" TRAIN_SPLIT="${train_split}" EVAL_SPLIT="${eval_split}" \
    MAX_TRAIN="${max_train}" MAX_EVAL="${max_eval}" TMAX="${tmax}" MAX_MODEL_LEN="${max_model_len}" \
    OFFLINE_ETA0="${ETA0}" OFFLINE_LAYER="${LAYER}" ONLINE_K_SCALE="${KS}" \
    RUN_GREEDY="${RUN_GREEDY}" RUN_ESM="${RUN_ESM}" RUN_ABLATIONS="${RUN_ABLATIONS}" \
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

run_name = str(os.environ.get("RUN_NAME", "")).strip()
if run_name != "":
    raw.setdefault("outputs", {})
    raw["outputs"]["run_name"] = run_name

raw.setdefault("model", {})
raw["model"]["name_or_path"] = os.environ["MODEL_PATH"]
tp_s = str(os.environ.get("MODEL_TP", "__KEEP__")).strip()
if tp_s and tp_s != "__KEEP__":
    raw["model"]["tensor_parallel_size"] = int(tp_s)
mns_s = str(os.environ.get("MAX_NUM_SEQS", "__KEEP__")).strip()
if mns_s and mns_s != "__KEEP__":
    if mns_s.lower() in {"none", "null"}:
        raise ValueError("MAX_NUM_SEQS cannot be null/none")
    raw["model"]["max_num_seqs"] = int(mns_s)

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

max_train = _maybe_int(os.environ.get("MAX_TRAIN", "__KEEP__"))
max_eval = _maybe_int(os.environ.get("MAX_EVAL", "__KEEP__"))
if max_train != "__KEEP__":
    raw["task"]["max_train_examples"] = max_train
if max_eval != "__KEEP__":
    raw["task"]["max_eval_examples"] = max_eval

raw.setdefault("prompt", {})
raw["prompt"]["template"] = os.environ["PROMPT_TEMPLATE"]

raw.setdefault("decode", {})
tmax = _maybe_int(os.environ.get("TMAX", "__KEEP__"))
if tmax != "__KEEP__" and tmax is not None:
    raw["decode"]["max_new_tokens"] = int(tmax)
    raw.setdefault("offline_mine", {})
    raw["offline_mine"]["max_new_tokens"] = int(tmax)

max_model_len = _maybe_int(os.environ.get("MAX_MODEL_LEN", "__KEEP__"))
if max_model_len != "__KEEP__" and max_model_len is not None:
    raw.setdefault("model", {})
    raw["model"]["max_model_len"] = int(max_model_len)

raw.setdefault("offline_mine", {})
eta0 = str(os.environ.get("OFFLINE_ETA0", "")).strip()
layer = str(os.environ.get("OFFLINE_LAYER", "")).strip()
if eta0 != "":
    raw["offline_mine"]["eta0"] = float(eta0)
if layer != "":
    # layer spec can be an int (0-based) or a ratio string like "1/5" / "0.6"
    raw["offline_mine"]["candidate_layers"] = [layer]

raw.setdefault("online", {})
ks = str(os.environ.get("ONLINE_K_SCALE", "")).strip()
if ks != "":
    raw["online"]["k_scale"] = float(ks)

# Eval methods toggle.
methods = []
if str(os.environ.get("RUN_GREEDY", "0")).strip() not in {"0", "false", "False"}:
    methods.append("greedy")
if str(os.environ.get("RUN_ESM", "1")).strip() not in {"0", "false", "False"}:
    methods.append("esm")
if not methods:
    raise ValueError("Both RUN_GREEDY and RUN_ESM are disabled; nothing to run.")
raw.setdefault("eval", {})
raw["eval"]["methods"] = methods

# Ablations toggle.
if str(os.environ.get("RUN_ABLATIONS", "0")).strip() in {"0", "false", "False"}:
    raw.setdefault("eval", {})
    raw["eval"]["ablations"] = []

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
PY
}

jobs=()  # "IDX|RUN_NAME|RID|TP|CFG_PATH|CMD"
job_idx=0
if [[ "${LIST_JOBS}" == "1" ]]; then
  echo "[list_jobs] 1"
  echo "[ordering] model -> dataset -> layer -> eta0 -> k_scale"
  echo "[note] set SWEEP_ID to match previous run_ids if needed"
  echo
fi
for mspec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model_key model_path model_tp model_max_num_seqs <<< "${mspec}"
  model_key="$(sanitize "${model_key}")"
  model_tp="${model_tp:-1}"
  model_max_num_seqs="${model_max_num_seqs:-__KEEP__}"

  for dspec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r dataset prompt_template train_split eval_split max_train max_eval tmax max_model_len <<< "${dspec}"
    dataset_key="$(sanitize "${dataset}")"
    run_name="$(sanitize "${RUN_NAME_PREFIX}")_${model_key}_${dataset_key}"

    for LAYER in "${LAYER_LIST[@]}"; do
      for ETA0 in "${ETA0_LIST[@]}"; do
        for KS in "${KSCALE_LIST[@]}"; do
          rid="${SWEEP_ID}__L$(f_tag "${LAYER}")_eta$(f_tag "${ETA0}")_ks$(f_tag "${KS}")"
          cfg_path="${OUT_CFG_DIR}/${run_name}__${rid}.yaml"
          if [[ "${LIST_JOBS}" == "1" ]]; then
            printf "%5d | model=%s | dataset=%s | layer=%s | eta0=%s | k_scale=%s | run_name=%s | run_id=%s\n" \
              "${job_idx}" "${model_key}" "${dataset_key}" "${LAYER}" "${ETA0}" "${KS}" "${run_name}" "${rid}"
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
            "${max_train}" "${max_eval}" "${tmax}" "${max_model_len}" \
            "${LAYER}" "${ETA0}" "${KS}"

          cmd="cd \"${REPO_ROOT}\""
          cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" mine"
          cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" select"
          cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" memory"
          cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" eval"

          jobs+=( "${job_idx}|${run_name}|${rid}|${model_tp}|${cfg_path}|${cmd}" )
          job_idx=$((job_idx + 1))
        done
      done
    done
  done
done

if [[ "${LIST_JOBS}" == "1" ]]; then
  echo
  echo "[n_jobs_total] ${grid_size}"
  exit 0
fi

echo "[mode] full grid (mine+select+memory+eval)"
echo "[base_cfg] ${BASE_CFG}"
echo "[run_name_prefix] ${RUN_NAME_PREFIX}"
echo "[sweep_id] ${SWEEP_ID}"
echo "[resume_from] ${RESUME_FROM} (0-based index)"
echo "[gpus] ${GPUS}"
echo "[n_models] ${n_models}"
echo "[n_datasets] ${n_datasets}"
echo "[grid_size] ${grid_size}"
echo "[n_jobs] ${#jobs[@]}"
echo "[out_cfg_dir] ${OUT_CFG_DIR}"
echo "[outputs_root] ${OUTPUTS_ROOT}"
if [[ -n "${LOG_DIR}" ]]; then
  echo "[log_dir] ${LOG_DIR}"
else
  echo "[log_dir] per-run (outputs/<run_name>/<run_id>/logs/stdout.log)"
fi
echo

if (( ${#jobs[@]} == 0 )); then
  echo "[no jobs] RESUME_FROM=${RESUME_FROM} (grid_size=${grid_size})"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry_run] 1 (configs generated; jobs not started)"
  exit 0
fi

available_gpus=( "${GPU_LIST[@]}" )
declare -A pid_to_gpus=()
declare -A pid_to_job=()
failed=0
failed_jobs=()

# Completion event queue (bash 4.x compatible replacement for `wait -n -p`).
EVENT_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/grid_sweep.${SWEEP_ID}.XXXXXX")"
EVENT_FIFO="${EVENT_TMP_DIR}/events.fifo"
mkfifo "${EVENT_FIFO}"
cleanup_events() {
  rm -rf "${EVENT_TMP_DIR}"
}

terminate_requested=0
terminate_jobs() {
  local pids=("${!pid_to_gpus[@]}")
  if (( ${#pids[@]} == 0 )); then
    return
  fi
  echo "[signal] terminating ${#pids[@]} running job(s)..." >&2
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}
on_signal() {
  local sig="$1"
  if [[ "${terminate_requested}" == "1" ]]; then
    exit 1
  fi
  terminate_requested=1
  echo "[signal] ${sig} received; terminating running jobs..." >&2
  terminate_jobs
  if [[ "${sig}" == "INT" ]]; then
    exit 130
  fi
  if [[ "${sig}" == "TERM" ]]; then
    exit 143
  fi
  exit 1
}
trap cleanup_events EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

start_job() {
  local gpu_ids="$1"
  local job="$2"
  local cfg="$3"
  local cmd="$4"
  local run_name="$5"
  local run_id="$6"
  local log_path=""

  if [[ -n "${LOG_DIR}" ]]; then
    log_path="${LOG_DIR}/${job}.log"
    mkdir -p "${LOG_DIR}"
  else
    local run_dir="${OUTPUTS_ROOT}/${run_name}/${run_id}"
    local run_logs="${run_dir}/logs"
    mkdir -p "${run_logs}"
    log_path="${run_logs}/stdout.log"
  fi

  echo "[start][gpu=${gpu_ids}] ${job}"
  echo "  cfg: ${cfg}"
  echo "  log: ${log_path}"

  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    bash -lc "${cmd}"
    rc=$?
    # Use BASHPID (not $$) so the parent can match `$!`.
    printf '%s %s\n' "${BASHPID}" "${rc}" >"${EVENT_FIFO}"
    exit "${rc}"
  ) >"${log_path}" 2>&1 &

  local pid="$!"
  pid_to_gpus["${pid}"]="${gpu_ids}"
  pid_to_job["${pid}"]="${job}"
}

queue=( "${jobs[@]}" )
while [[ "${#queue[@]}" -gt 0 || "${#pid_to_gpus[@]}" -gt 0 ]]; do
  started_any=0
  while [[ "${#queue[@]}" -gt 0 ]]; do
    found=0
    for i in "${!queue[@]}"; do
      job="${queue[$i]}"
      IFS='|' read -r idx run_name rid tp cfg cmd <<< "${job}"
      tp="${tp:-1}"
      need="${tp}"
      if (( need <= 0 )); then
        need=1
      fi
      if (( ${#available_gpus[@]} >= need )); then
        gsel=( "${available_gpus[@]:0:${need}}" )
        available_gpus=( "${available_gpus[@]:${need}}" )
        gpu_ids="$(IFS=','; echo "${gsel[*]}")"
        job_name="${run_name}__${rid}"
        safe_job_name="$(sanitize "${job_name}")"
        safe_job_name="$(printf "%05d__%s" "${idx}" "${safe_job_name}")"

        # Remove this job from queue.
        queue=( "${queue[@]:0:${i}}" "${queue[@]:$((i + 1))}" )
        start_job "${gpu_ids}" "${safe_job_name}" "${cfg}" "${cmd}" "${run_name}" "${rid}"
        found=1
        started_any=1
        break
      fi
    done
    if [[ "${found}" == "0" ]]; then
      break
    fi
  done

  if [[ "${#pid_to_gpus[@]}" -gt 0 ]]; then
    done_pid=""
    rc=0
    # Block until any job finishes and reports "<pid> <rc>".
    if IFS=' ' read -r done_pid rc <"${EVENT_FIFO}"; then
      :
    else
      echo "[error] failed to read job completion event" >&2
      exit 2
    fi
    # Reap to avoid zombies (ignore wait rc; we already captured rc).
    wait "${done_pid}" >/dev/null 2>&1 || true

    if [[ -n "${done_pid}" ]]; then
      gpu_ids="${pid_to_gpus[${done_pid}]}"
      job_name="${pid_to_job[${done_pid}]}"
      unset pid_to_gpus["${done_pid}"]
      unset pid_to_job["${done_pid}"]
      IFS=',' read -r -a freed <<< "${gpu_ids}"
      available_gpus+=( "${freed[@]}" )
      if (( rc != 0 )); then
        failed=$((failed + 1))
        failed_jobs+=( "${job_name}|gpu=${gpu_ids}|rc=${rc}" )
        echo "[fail][gpu=${gpu_ids}][rc=${rc}] ${job_name}" >&2
      else
        echo "[done][gpu=${gpu_ids}] ${job_name}"
      fi
    fi
  else
    if [[ "${#queue[@]}" -gt 0 && "${started_any}" == "0" ]]; then
      echo "[error] cannot schedule remaining jobs; check GPUS list and MODEL_SPECS tensor_parallel_size" >&2
      exit 2
    fi
  fi
done

echo
if (( failed > 0 )); then
  echo "[all done] full sweep_id=${SWEEP_ID} (failed=${failed}/${#jobs[@]})" >&2
  for j in "${failed_jobs[@]}"; do
    echo "  - ${j}" >&2
  done
  exit 1
fi
echo "[all done] full sweep_id=${SWEEP_ID}"
