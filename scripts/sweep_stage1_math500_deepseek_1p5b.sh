#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-editable (TOP)
# =========================

# GPUs to use for parallel runs (one job per GPU). Edit as needed.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# Grid (Stage I / offline_mine)
# Total points = |K_LIST| * |ETA0_LIST| * |LAYER_LIST|
# (must be <= MAX_GRID)
# K_LIST=(4 6 8)
# ETA0_LIST=(0.0 0.001 0.002)
LAYER_LIST=(16 18)
# NOTE: offline_select.B / offline_select.min_per_control_point 已挪到 online sweep 中搜索（select/memory 很快）。

# Safety cap: refuse to run if grid size exceeds this.
MAX_GRID="${MAX_GRID:-128}"

# Base config (derived from smoke, with train=100 / eval=400 / esm-only).
BASE_CFG="${BASE_CFG:-configs/grid_math500_deepseek_1p5b.yaml}"

# Output run name (folder under outputs/).
RUN_NAME="${RUN_NAME:-gs_stage1_math500_deepseek_1p5b}"

# Sweep id used to prefix per-run run_id.
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"

# If DRY_RUN=1, only generate per-run configs and print the plan.
DRY_RUN="${DRY_RUN:-0}"

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
  echo "[error] BASE_CFG not found: ${BASE_CFG}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -le 0 ]]; then
  echo "[error] Empty GPUS list: ${GPUS}" >&2
  exit 2
fi

K_EFF=(__KEEP__)
ETA0_EFF=(__KEEP__)
LAYER_EFF=(__KEEP__)
if declare -p K_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#K_LIST[@]}" -gt 0 ]]; then
    K_EFF=( "${K_LIST[@]}" )
  fi
fi
if declare -p ETA0_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#ETA0_LIST[@]}" -gt 0 ]]; then
    ETA0_EFF=( "${ETA0_LIST[@]}" )
  fi
fi
if declare -p LAYER_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#LAYER_LIST[@]}" -gt 0 ]]; then
    LAYER_EFF=( "${LAYER_LIST[@]}" )
  fi
fi

grid_size=$(( ${#K_EFF[@]} * ${#ETA0_EFF[@]} * ${#LAYER_EFF[@]} ))
if (( grid_size > MAX_GRID )); then
  echo "[error] grid too large: ${grid_size} > MAX_GRID=${MAX_GRID}" >&2
  exit 2
fi

OUT_CFG_DIR="${OUT_CFG_DIR:-configs/_grid_stage1_generated}"
LOG_DIR="${LOG_DIR:-logs/grid_stage1_${SWEEP_ID}}"
mkdir -p "${OUT_CFG_DIR}" "${LOG_DIR}"

f_tag() {
  # float -> tag-friendly token, e.g. 0.001 -> 0p001
  local s="$1"
  s="${s//-/m}"
  s="${s//./p}"
  echo "${s}"
}

write_cfg() {
  local out_cfg="$1"
  local K="$2"
  local ETA0="$3"
  local LAYER="$4"

  env BASE_CFG="${BASE_CFG}" OUT_CFG="${out_cfg}" RUN_NAME="${RUN_NAME}" \
    OFFLINE_K="${K}" OFFLINE_ETA0="${ETA0}" OFFLINE_LAYER="${LAYER}" \
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

# Enforce sweep defaults: fixed sample counts, esm-only, no ablations.
raw.setdefault("task", {})
raw["task"]["max_train_examples"] = 100
raw["task"]["max_eval_examples"] = 400

raw.setdefault("eval", {})
raw["eval"]["methods"] = ["esm"]
raw["eval"]["ablations"] = []
raw["eval"].pop("artifact_run_dir", None)

raw.setdefault("outputs", {})
raw["outputs"]["run_name"] = os.environ["RUN_NAME"]

raw.setdefault("offline_mine", {})
K = str(os.environ.get("OFFLINE_K", "__KEEP__")).strip()
ETA0 = str(os.environ.get("OFFLINE_ETA0", "__KEEP__")).strip()
LAYER = str(os.environ.get("OFFLINE_LAYER", "__KEEP__")).strip()
if K != "" and K != "__KEEP__":
    raw["offline_mine"]["K"] = int(K)
if ETA0 != "" and ETA0 != "__KEEP__":
    raw["offline_mine"]["eta0"] = float(ETA0)
if LAYER != "" and LAYER != "__KEEP__":
    raw["offline_mine"]["candidate_layers"] = [int(LAYER)]

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
PY
}

jobs=()  # "RID|CFG_PATH|CMD"
for K in "${K_EFF[@]}"; do
  for ETA0 in "${ETA0_EFF[@]}"; do
    for LAYER in "${LAYER_EFF[@]}"; do
      rid="${SWEEP_ID}__s1"
      has_tag=0
      if [[ "${K}" != "__KEEP__" ]]; then
        rid+="_K${K}"
        has_tag=1
      fi
      if [[ "${ETA0}" != "__KEEP__" ]]; then
        rid+="_eta$(f_tag "${ETA0}")"
        has_tag=1
      fi
      if [[ "${LAYER}" != "__KEEP__" ]]; then
        rid+="_L${LAYER}"
        has_tag=1
      fi
      if [[ "${has_tag}" == "0" ]]; then
        rid+="_base"
      fi

      cfg_path="${OUT_CFG_DIR}/${rid}.yaml"
      write_cfg "${cfg_path}" "${K}" "${ETA0}" "${LAYER}"

      cmd="cd \"${REPO_ROOT}\""
      cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" mine"
      cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" select"
      cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" memory"
      cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" eval"

      jobs+=( "${rid}|${cfg_path}|${cmd}" )
    done
  done
done

echo "[mode] stage1 (full pipeline)"
echo "[base_cfg] ${BASE_CFG}"
echo "[run_name] ${RUN_NAME}"
echo "[sweep_id] ${SWEEP_ID}"
echo "[gpus] ${GPUS}"
echo "[grid_size] ${grid_size}"
echo "[out_cfg_dir] ${OUT_CFG_DIR}"
echo "[log_dir] ${LOG_DIR}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry_run] 1 (configs generated; jobs not started)"
  exit 0
fi

available_gpus=( "${GPU_LIST[@]}" )
declare -A pid_to_gpu=()
declare -A pid_to_rid=()

start_job() {
  local gpu="$1"
  local rid="$2"
  local cfg="$3"
  local cmd="$4"
  local log_path="${LOG_DIR}/${rid}.log"

  echo "[start][gpu=${gpu}] ${rid}"
  echo "  cfg: ${cfg}"
  echo "  log: ${log_path}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec bash -lc "${cmd}"
  ) >"${log_path}" 2>&1 &

  local pid="$!"
  pid_to_gpu["${pid}"]="${gpu}"
  pid_to_rid["${pid}"]="${rid}"
}

queue=( "${jobs[@]}" )
while [[ "${#queue[@]}" -gt 0 || "${#pid_to_gpu[@]}" -gt 0 ]]; do
  while [[ "${#queue[@]}" -gt 0 && "${#available_gpus[@]}" -gt 0 ]]; do
    job="${queue[0]}"
    queue=( "${queue[@]:1}" )

    gpu="${available_gpus[0]}"
    available_gpus=( "${available_gpus[@]:1}" )

    IFS='|' read -r rid cfg cmd <<< "${job}"
    start_job "${gpu}" "${rid}" "${cfg}" "${cmd}"
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
        rid="${pid_to_rid[${pid}]}"
        unset pid_to_gpu["${pid}"]
        unset pid_to_rid["${pid}"]
        available_gpus+=( "${gpu}" )
        echo "[done][gpu=${gpu}] ${rid}"
      fi
    done
  fi
done

echo
echo "[all done] stage1 sweep_id=${SWEEP_ID}"
