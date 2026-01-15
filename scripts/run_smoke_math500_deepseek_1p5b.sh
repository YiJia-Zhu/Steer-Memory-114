#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG="${1:-configs/default}"
# Default to GPU=1 (historical); override via env: GPU=0 ./scripts/run_smoke_...
GPU="${GPU:-1}"

# Use current python by default (assumes you're already in the easysteer env).
# If you prefer forcing conda-run, set USE_CONDA_RUN=1.
USE_CONDA_RUN="${USE_CONDA_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-easysteer}"

PY=(python)
if [[ "${USE_CONDA_RUN}" == "1" ]]; then
  PY=(conda run -n "${CONDA_ENV}" python)
fi

if [[ ! -f "${CFG}" ]]; then
  # Allow passing configs/default (no extension).
  if [[ -f "${CFG}.yaml" ]]; then
    CFG="${CFG}.yaml"
  elif [[ -f "${CFG}.yml" ]]; then
    CFG="${CFG}.yml"
  fi
fi

run_stage() {
  local stage="$1"
  shift
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY[@]}" run.py --config "${CFG}" "$@" "${stage}"
}

echo "[cfg] ${CFG}"
echo "[gpu] ${GPU}"
echo

run_stage mine
run_stage select --run-id latest
run_stage memory --run-id latest
run_stage eval --run-id latest

RUN_NAME="$("${PY[@]}" -c "from esm.config import load_config; print(load_config('${CFG}').outputs.run_name)")"
RID="$(cat "outputs/${RUN_NAME}/LATEST")"

echo
echo "[done] outputs/${RUN_NAME}/${RID}/"
