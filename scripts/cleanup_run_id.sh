#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Delete runs under outputs/<run_name>/<run_id> (and outputs/_suite/<run_id>).

Usage:
  scripts/cleanup_run_id.sh --run-id 20260116_224426
  scripts/cleanup_run_id.sh --prefix 20260116_224426
  scripts/cleanup_run_id.sh --prefix 20260116_224426 --dry-run
  scripts/cleanup_run_id.sh --run-id 20260116_224426 --root /path/to/outputs --yes

Options:
  --run-id <id>   Exact run_id to delete.
  --prefix <id>   Delete run_ids that start with this prefix (e.g., date id).
  --root <path>   Outputs root (default: outputs).
  --dry-run       Only list matching directories.
  --yes, -y       Skip confirmation prompt.
  -h, --help      Show this help.
USAGE
}

RUN_ID=""
PREFIX=""
ROOT="outputs"
DRY_RUN=0
YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --prefix)
      PREFIX="${2:-}"
      shift 2
      ;;
    --root)
      ROOT="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes|-y)
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -n "${RUN_ID}" && -n "${PREFIX}" ]]; then
  echo "[error] --run-id and --prefix are mutually exclusive" >&2
  exit 2
fi
if [[ -z "${RUN_ID}" && -z "${PREFIX}" ]]; then
  echo "[error] must provide --run-id or --prefix" >&2
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${ROOT}" != /* ]]; then
  ROOT="${REPO_ROOT}/${ROOT}"
fi

if [[ ! -d "${ROOT}" ]]; then
  echo "[error] outputs root not found: ${ROOT}" >&2
  exit 1
fi

pattern="${RUN_ID}"
if [[ -n "${PREFIX}" ]]; then
  pattern="${PREFIX}*"
fi

mapfile -t targets < <(find "${ROOT}" -mindepth 2 -maxdepth 2 -type d -name "${pattern}" -print | sort)
if [[ "${#targets[@]}" -eq 0 ]]; then
  echo "[info] no matches under ${ROOT} for pattern: ${pattern}"
  exit 0
fi

echo "[match] ${#targets[@]} directory(s):"
printf '  %s\n' "${targets[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ "${YES}" != "1" ]]; then
  read -r -p "Delete these directories? [y/N] " ans
  case "${ans}" in
    y|Y|yes|YES) ;;
    *)
      echo "[abort] no changes made"
      exit 1
      ;;
  esac
fi

rm -rf -- "${targets[@]}"
echo "[done] deleted ${#targets[@]} directory(s)"
