#!/usr/bin/env python3
"""
Summarize grid / sweep runs under outputs/<run_name>/<run_id>/ into one CSV.

This script is designed to work with the sweep scripts:
- scripts/sweep_stage1_math500_deepseek_1p5b.sh
- scripts/sweep_online_math500_deepseek_1p5b.sh

Typical usage:
  python scripts/summarize_sweep_results.py --run-name gs_stage1_math500_deepseek_1p5b
  python scripts/summarize_sweep_results.py --run-name gs_online_math500_deepseek_1p5b
  - 只看某个 sweep_id（run_id 子串过滤）：
      - python scripts/summarize_sweep_results.py --run-name gs_online_math500_deepseek_1p5b --contains 20260114_120000
  - 指定输出路径：
      - python scripts/summarize_sweep_results.py --run-name ... --out /tmp/summary.csv
      
Optional filtering:
  python scripts/summarize_sweep_results.py --run-name gs_online_math500_deepseek_1p5b --contains 20260114_120000

Outputs:
  - Default: outputs/_grid/<run_name>/grid_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_latest_run_dir(p: Path) -> Path:
    """
    Resolve ".../<run_name>/latest" -> ".../<run_name>/<LATEST>" if possible.
    Mirrors esm/online/esm.py:_resolve_artifact_root behavior.
    """
    if p.is_dir():
        return p
    if p.name.lower() != "latest":
        return p
    latest_path = p.parent / "LATEST"
    if not latest_path.exists():
        return p
    rid = latest_path.read_text(encoding="utf-8").strip()
    if not rid:
        return p
    cand = p.parent / rid
    return cand if cand.is_dir() else p


def _find_eval_summary(run_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """
    Find the ESM summary.jsonl file.

    We prefer the largest-budget ESM_T* folder if multiple exist.
    """
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        return None, None

    candidates: list[tuple[int, Path]] = []
    for p in eval_dir.glob("ESM_T*/summary.jsonl"):
        try:
            t = int(p.parent.name.split("ESM_T", 1)[1])
        except Exception:
            continue
        candidates.append((t, p))
    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _t, summary_path = candidates[0]
    try:
        line = summary_path.read_text(encoding="utf-8").strip().splitlines()[0]
        obj = json.loads(line)
        if not isinstance(obj, dict):
            return summary_path, None
        return summary_path, obj
    except Exception:
        return summary_path, None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_memory_meta(mem_dir: Path) -> tuple[int | None, int | None]:
    """
    Read memory/meta.txt if present. Expected format:
      N=24 H=1536
    """
    p = mem_dir / "meta.txt"
    if not p.exists():
        return None, None
    try:
        s = p.read_text(encoding="utf-8").strip()
        parts = s.replace("\n", " ").split()
        n = None
        h = None
        for tok in parts:
            if tok.startswith("N="):
                n = int(tok.split("=", 1)[1])
            if tok.startswith("H="):
                h = int(tok.split("=", 1)[1])
        return n, h
    except Exception:
        return None, None


def _parse_sweep_id(run_id: str) -> tuple[str | None, str | None]:
    if "__" not in run_id:
        return None, None
    a, b = run_id.split("__", 1)
    return a or None, b or None


@dataclass(frozen=True)
class Row:
    data: dict[str, Any]


def _build_row(
    *,
    outputs_root: Path,
    run_name: str,
    run_id: str,
) -> Row:
    run_dir = outputs_root / run_name / run_id
    sweep_id, tag = _parse_sweep_id(run_id)

    base: dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "sweep_id": sweep_id,
        "tag": tag,
        "run_dir": str(run_dir),
        "status": "ok",
    }

    cfg_path = run_dir / "config_resolved.json"
    cfg = _load_json(cfg_path) if cfg_path.exists() else None
    if cfg is None:
        base["status"] = "missing_config"
        cfg = {}

    # Eval summary (ESM only).
    summary_path, summary = _find_eval_summary(run_dir)
    base["eval_summary_path"] = str(summary_path) if summary_path is not None else None
    if summary is None:
        base["status"] = "missing_eval"
        summary = {}

    base["dataset"] = _get(cfg, "task.dataset")
    base["eval_split"] = _get(cfg, "task.eval_split")
    base["n_eval"] = _safe_int(summary.get("n"))
    base["T_max"] = _safe_int(summary.get("T_max"))
    base["acc"] = _safe_float(summary.get("acc"))

    # Key hyperparameters (current run config).
    base["offline_K"] = _safe_int(_get(cfg, "offline_mine.K"))
    base["offline_eta0"] = _safe_float(_get(cfg, "offline_mine.eta0"))
    cand_layers = _get(cfg, "offline_mine.candidate_layers")
    if isinstance(cand_layers, list):
        base["offline_candidate_layers"] = ",".join(str(int(x)) for x in cand_layers if str(x).strip() != "")
    else:
        base["offline_candidate_layers"] = None

    base["offline_select_B"] = _safe_int(_get(cfg, "offline_select.B"))
    base["offline_select_min_per_control_point"] = _safe_int(_get(cfg, "offline_select.min_per_control_point"))

    base["online_k_scale"] = _safe_float(_get(cfg, "online.k_scale"))
    base["online_tau_null"] = _safe_float(_get(cfg, "online.tau_null"))
    base["online_min_sim"] = _safe_float(_get(cfg, "online.min_sim"))
    base["online_rho"] = _safe_float(_get(cfg, "online.rho"))
    base["online_L"] = _safe_int(_get(cfg, "online.L"))
    base["online_probe_tokens"] = _safe_int(_get(cfg, "online.probe_tokens"))
    base["online_k_retrieve"] = _safe_int(_get(cfg, "online.k_retrieve"))
    base["online_batch_size_examples"] = _safe_int(_get(cfg, "online.batch_size_examples"))

    base["eval_artifact_run_dir"] = _get(cfg, "eval.artifact_run_dir")

    # Memory metadata for this run (only exists if memory stage ran in this run_dir).
    mem_n, mem_h = _read_memory_meta(run_dir / "memory")
    base["memory_N"] = mem_n
    base["memory_H"] = mem_h

    # If this is an online-only run, try to also summarize the artifact run.
    art_raw = str(base.get("eval_artifact_run_dir") or "").strip()
    if art_raw != "":
        art_dir = _resolve_latest_run_dir(Path(art_raw))
        base["artifact_run_dir_resolved"] = str(art_dir) if art_dir.is_dir() else None
        art_mem_n, art_mem_h = _read_memory_meta(art_dir / "memory")
        base["artifact_memory_N"] = art_mem_n
        base["artifact_memory_H"] = art_mem_h
        art_cfg = _load_json(art_dir / "config_resolved.json") if (art_dir / "config_resolved.json").exists() else None
        if isinstance(art_cfg, dict):
            base["artifact_offline_K"] = _safe_int(_get(art_cfg, "offline_mine.K"))
            base["artifact_offline_eta0"] = _safe_float(_get(art_cfg, "offline_mine.eta0"))
            art_layers = _get(art_cfg, "offline_mine.candidate_layers")
            if isinstance(art_layers, list):
                base["artifact_offline_candidate_layers"] = ",".join(
                    str(int(x)) for x in art_layers if str(x).strip() != ""
                )
            else:
                base["artifact_offline_candidate_layers"] = None
            base["artifact_offline_select_B"] = _safe_int(_get(art_cfg, "offline_select.B"))
            base["artifact_offline_select_min_per_control_point"] = _safe_int(
                _get(art_cfg, "offline_select.min_per_control_point")
            )

    return Row(data=base)


def _iter_run_ids(run_root: Path) -> list[str]:
    if not run_root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(run_root.iterdir()):
        if not p.is_dir():
            continue
        # run_id folders are directories; skip helper folders if any.
        out.append(p.name)
    return out


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.data.keys():
            if k in seen:
                continue
            fieldnames.append(k)
            seen.add(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=str, default="outputs")
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--contains", type=str, default=None, help="Only include run_id containing this substring.")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help='Output CSV path (default: outputs/_grid/<run_name>/grid_summary.csv).',
    )
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    run_name = str(args.run_name)

    run_root = outputs_root / run_name
    run_ids = _iter_run_ids(run_root)
    if args.contains:
        needle = str(args.contains)
        run_ids = [rid for rid in run_ids if needle in rid]

    rows: list[Row] = []
    for rid in run_ids:
        rows.append(_build_row(outputs_root=outputs_root, run_name=run_name, run_id=rid))

    def _sort_key(r: Row) -> tuple[int, float, str]:
        status = str(r.data.get("status") or "")
        ok_rank = 0 if status == "ok" else 1
        acc = r.data.get("acc")
        acc_v = float(acc) if isinstance(acc, (int, float)) else -1.0
        return (ok_rank, -acc_v, str(r.data.get("run_id") or ""))

    rows.sort(key=_sort_key)

    out_path = Path(args.out) if args.out else (outputs_root / "_grid" / run_name / "grid_summary.csv")
    _write_csv(out_path, rows)

    # Print a tiny hint for quick selection.
    best = next((r for r in rows if str(r.data.get("status")) == "ok" and r.data.get("acc") is not None), None)
    if best is not None:
        print(f"[ok] wrote: {out_path}")
        print(
            f"[best] run_id={best.data.get('run_id')} acc={best.data.get('acc')} "
            f"T_max={best.data.get('T_max')} dir={best.data.get('run_dir')}"
        )
    else:
        print(f"[ok] wrote: {out_path}")
        print("[best] none (no successful eval found)")


if __name__ == "__main__":
    main()
