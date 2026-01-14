#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-editable (TOP)
# =========================

# GPUs to use for parallel runs (one job per GPU). Edit as needed.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# Grid (Online / select+memory+eval; reuse Stage-I mine artifacts)
# Total points = |(optional)KSCALE_LIST| * |(optional)TAU_LIST| * |(optional)MINSIM_LIST| * |(optional)RHO_LIST|
#             * |(optional)ONLINE_L_LIST| * |(optional)ONLINE_PROBE_TOKENS_LIST|
#             * |(optional)OFFLINE_SELECT_B_LIST| * |(optional)OFFLINE_SELECT_MIN_PER_M_LIST|
# (must be <= MAX_GRID)
# 注入强度
KSCALE_LIST=(0.25 0.5 1.0)
# 注入steer的阈值，低于则不注入
TAU_LIST=(0.0 0.05 0.1)
# top-1相似阈值，低于则判为无记忆，相当于null
# MINSIM_LIST=(0.0 0.1 0.2)
# probe 的 logprob gain 权重；决定“探测信号”在 Score 里占比（rho=0 等价不使用 probe 信号，但仍会花 probe tokens，想省算力可再配合 probe_tokens=0/variant=no_probing）
# RHO_LIST=(0.0 0.1 0.2)

ONLINE_L_LIST=(1 2 3)                    # probe candidates (L)

# ONLINE_PROBE_TOKENS_LIST=(4 8 16)        # probe length (per cand)

# offline_select（select/memory 很快；放到 online sweep 中一起扫）
# OFFLINE_SELECT_B_LIST=(16 24 32)              # library size B
# OFFLINE_SELECT_MIN_PER_M_LIST=(0 1 2)        # min_per_control_point

# Safety cap: refuse to run if grid size exceeds this.
MAX_GRID="${MAX_GRID:-128}"

# Base config (derived from smoke, with train=100 / eval=400 / esm-only).
BASE_CFG="${BASE_CFG:-configs/grid_math500_deepseek_1p5b.yaml}"

# Reuse Stage-I artifacts (must contain mine/candidates.jsonl). Point this to the BEST stage1 run you picked.
# Examples:
#   ARTIFACT_RUN_DIR="outputs/gs_stage1_math500_deepseek_1p5b/<run_id>"
#   ARTIFACT_RUN_DIR="outputs/gs_stage1_math500_deepseek_1p5b/latest"
ARTIFACT_RUN_DIR="${ARTIFACT_RUN_DIR:-outputs/gs_stage1_math500_deepseek_1p5b/latest}"

# Output run name (folder under outputs/).
RUN_NAME="${RUN_NAME:-gs_online_math500_deepseek_1p5b}"

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

resolve_artifact_dir() {
  local p="$1"
  if [[ -d "${p}" ]]; then
    echo "${p}"
    return 0
  fi
  local base
  base="$(basename "${p}")"
  shopt -s nocasematch
  if [[ "${base}" == "latest" ]]; then
    local parent
    parent="$(dirname "${p}")"
    local latest_path="${parent}/LATEST"
    if [[ -f "${latest_path}" ]]; then
      local rid
      rid="$(cat "${latest_path}" | tr -d '\r' | xargs || true)"
      if [[ -n "${rid}" && -d "${parent}/${rid}" ]]; then
        echo "${parent}/${rid}"
        shopt -u nocasematch
        return 0
      fi
    fi
  fi
  shopt -u nocasematch
  echo "${p}"
  return 0
}

ARTIFACT_ROOT="$(resolve_artifact_dir "${ARTIFACT_RUN_DIR}")"
ART_CAND="${ARTIFACT_ROOT}/mine/candidates.jsonl"
if [[ ! -f "${ART_CAND}" ]]; then
  echo "[error] artifact mine candidates not found: ${ART_CAND}" >&2
  exit 2
fi

KSCALE_EFF=(__KEEP__)
TAU_EFF=(__KEEP__)
MINSIM_EFF=(__KEEP__)
RHO_EFF=(__KEEP__)
if declare -p KSCALE_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#KSCALE_LIST[@]}" -gt 0 ]]; then
    KSCALE_EFF=( "${KSCALE_LIST[@]}" )
  fi
fi
if declare -p TAU_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#TAU_LIST[@]}" -gt 0 ]]; then
    TAU_EFF=( "${TAU_LIST[@]}" )
  fi
fi
if declare -p MINSIM_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#MINSIM_LIST[@]}" -gt 0 ]]; then
    MINSIM_EFF=( "${MINSIM_LIST[@]}" )
  fi
fi
if declare -p RHO_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#RHO_LIST[@]}" -gt 0 ]]; then
    RHO_EFF=( "${RHO_LIST[@]}" )
  fi
fi

ONLINE_L_EFF=(__KEEP__)
ONLINE_PROBE_TOKENS_EFF=(__KEEP__)
if declare -p ONLINE_L_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#ONLINE_L_LIST[@]}" -gt 0 ]]; then
    ONLINE_L_EFF=( "${ONLINE_L_LIST[@]}" )
  fi
fi
if declare -p ONLINE_PROBE_TOKENS_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#ONLINE_PROBE_TOKENS_LIST[@]}" -gt 0 ]]; then
    ONLINE_PROBE_TOKENS_EFF=( "${ONLINE_PROBE_TOKENS_LIST[@]}" )
  fi
fi

SELECT_B_EFF=(__KEEP__)
SELECT_MINM_EFF=(__KEEP__)
if declare -p OFFLINE_SELECT_B_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#OFFLINE_SELECT_B_LIST[@]}" -gt 0 ]]; then
    SELECT_B_EFF=( "${OFFLINE_SELECT_B_LIST[@]}" )
  fi
fi
if declare -p OFFLINE_SELECT_MIN_PER_M_LIST >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [[ "${#OFFLINE_SELECT_MIN_PER_M_LIST[@]}" -gt 0 ]]; then
    SELECT_MINM_EFF=( "${OFFLINE_SELECT_MIN_PER_M_LIST[@]}" )
  fi
fi

grid_size=$(( ${#KSCALE_EFF[@]} * ${#TAU_EFF[@]} * ${#MINSIM_EFF[@]} * ${#RHO_EFF[@]} * ${#ONLINE_L_EFF[@]} * ${#ONLINE_PROBE_TOKENS_EFF[@]} * ${#SELECT_B_EFF[@]} * ${#SELECT_MINM_EFF[@]} ))
if (( grid_size > MAX_GRID )); then
  echo "[error] grid too large: ${grid_size} > MAX_GRID=${MAX_GRID}" >&2
  exit 2
fi

OUT_CFG_DIR="${OUT_CFG_DIR:-configs/_grid_online_generated}"
LOG_DIR="${LOG_DIR:-logs/grid_online_${SWEEP_ID}}"
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
  local KS="$2"
  local TAU="$3"
  local MS="$4"
  local RHO="$5"
  local L="$6"
  local PT="$7"
  local SEL_B="$8"
  local SEL_MINM="$9"

  env BASE_CFG="${BASE_CFG}" OUT_CFG="${out_cfg}" RUN_NAME="${RUN_NAME}" ARTIFACT_RUN_DIR="${ARTIFACT_RUN_DIR}" \
    ONLINE_K_SCALE="${KS}" ONLINE_TAU_NULL="${TAU}" ONLINE_MIN_SIM="${MS}" ONLINE_RHO="${RHO}" \
    ONLINE_L="${L}" ONLINE_PROBE_TOKENS="${PT}" \
    OFFLINE_SELECT_B="${SEL_B}" OFFLINE_SELECT_MIN_PER_M="${SEL_MINM}" \
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

raw.setdefault("online", {})
ks = str(os.environ.get("ONLINE_K_SCALE", "__KEEP__")).strip()
tau = str(os.environ.get("ONLINE_TAU_NULL", "__KEEP__")).strip()
ms = str(os.environ.get("ONLINE_MIN_SIM", "__KEEP__")).strip()
rho = str(os.environ.get("ONLINE_RHO", "__KEEP__")).strip()
if ks != "" and ks != "__KEEP__":
    raw["online"]["k_scale"] = float(ks)
if tau != "" and tau != "__KEEP__":
    raw["online"]["tau_null"] = float(tau)
if ms != "" and ms != "__KEEP__":
    raw["online"]["min_sim"] = float(ms)
if rho != "" and rho != "__KEEP__":
    raw["online"]["rho"] = float(rho)

online_l = str(os.environ.get("ONLINE_L", "__KEEP__")).strip()
online_pt = str(os.environ.get("ONLINE_PROBE_TOKENS", "__KEEP__")).strip()
if online_l != "" and online_l != "__KEEP__":
    raw["online"]["L"] = int(online_l)
if online_pt != "" and online_pt != "__KEEP__":
    raw["online"]["probe_tokens"] = int(online_pt)

sel_b = str(os.environ.get("OFFLINE_SELECT_B", "__KEEP__")).strip()
sel_minm = str(os.environ.get("OFFLINE_SELECT_MIN_PER_M", "__KEEP__")).strip()
raw.setdefault("offline_select", {})
if sel_b != "" and sel_b != "__KEEP__":
    raw["offline_select"]["B"] = int(sel_b)
if sel_minm != "" and sel_minm != "__KEEP__":
    raw["offline_select"]["min_per_control_point"] = int(sel_minm)

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
PY
}

jobs=()  # "RID|CFG_PATH|CMD"
for KS in "${KSCALE_EFF[@]}"; do
  for TAU in "${TAU_EFF[@]}"; do
    for MS in "${MINSIM_EFF[@]}"; do
      for RHO in "${RHO_EFF[@]}"; do
        for L in "${ONLINE_L_EFF[@]}"; do
          for PT in "${ONLINE_PROBE_TOKENS_EFF[@]}"; do
            for SEL_B in "${SELECT_B_EFF[@]}"; do
              for SEL_MINM in "${SELECT_MINM_EFF[@]}"; do
                rid="${SWEEP_ID}__on"
                has_tag=0
                if [[ "${KS}" != "__KEEP__" ]]; then
                  rid+="_ks$(f_tag "${KS}")"
                  has_tag=1
                fi
                if [[ "${TAU}" != "__KEEP__" ]]; then
                  rid+="_tau$(f_tag "${TAU}")"
                  has_tag=1
                fi
                if [[ "${MS}" != "__KEEP__" ]]; then
                  rid+="_sim$(f_tag "${MS}")"
                  has_tag=1
                fi
                if [[ "${RHO}" != "__KEEP__" ]]; then
                  rid+="_rho$(f_tag "${RHO}")"
                  has_tag=1
                fi
                if [[ "${L}" != "__KEEP__" ]]; then
                  rid+="_L${L}"
                  has_tag=1
                fi
                if [[ "${PT}" != "__KEEP__" ]]; then
                  rid+="_P${PT}"
                  has_tag=1
                fi
                if [[ "${SEL_B}" != "__KEEP__" ]]; then
                  rid+="_B${SEL_B}"
                  has_tag=1
                fi
                if [[ "${SEL_MINM}" != "__KEEP__" ]]; then
                  rid+="_minm${SEL_MINM}"
                  has_tag=1
                fi
                if [[ "${has_tag}" == "0" ]]; then
                  rid+="_base"
                fi

                cfg_path="${OUT_CFG_DIR}/${rid}.yaml"
                write_cfg "${cfg_path}" "${KS}" "${TAU}" "${MS}" "${RHO}" "${L}" "${PT}" "${SEL_B}" "${SEL_MINM}"

                cmd="cd \"${REPO_ROOT}\""
                cmd+=" && mkdir -p \"outputs/${RUN_NAME}/${rid}/mine\""
                cmd+=" && cp \"${ART_CAND}\" \"outputs/${RUN_NAME}/${rid}/mine/candidates.jsonl\""
                cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" select"
                cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" memory"
                cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" eval"

                jobs+=( "${rid}|${cfg_path}|${cmd}" )
              done
            done
          done
        done
      done
    done
  done
done

echo "[mode] online (select+memory+eval)"
echo "[base_cfg] ${BASE_CFG}"
echo "[artifact_run_dir] ${ARTIFACT_RUN_DIR}"
echo "[artifact_root] ${ARTIFACT_ROOT}"
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
echo "[all done] online sweep_id=${SWEEP_ID}"
