#!/usr/bin/env python3
"""
Summarize "main experiments" runs into one CSV.

This script aggregates:
  outputs/<run_name>/<run_id>/tables/main_results_single.csv
and enriches rows with model/dataset metadata from:
  outputs/<run_name>/<run_id>/config_resolved.json

Typical usage:
  python scripts/summarize_main_experiments.py --run-id 20260114_120000 --run-name-prefix main
  python scripts/summarize_main_experiments.py --run-name-prefix main --contains ds_r1_qwen_1p5b

Default output:
  outputs/_main/<run_id>/main_summary.csv   (when --run-id is provided)
  outputs/_main/main_summary.csv           (otherwise)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_main_results(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            row = next(r, None)
        if not header or not row:
            return None
        d: dict[str, Any] = {}
        for k, v in zip(header, row):
            k = str(k)
            vv: Any = v
            try:
                if k.lower() in {"t_max"}:
                    vv = int(v)
                elif k.lower() in {"acc", "greedy-cot", "esm"}:
                    vv = float(v)
            except Exception:
                vv = v
            d[k] = vv
        return d
    except Exception:
        return None


def _iter_run_names(outputs_root: Path, prefix: str | None, contains: str | None) -> list[str]:
    if not outputs_root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(outputs_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        name = p.name
        if prefix and not name.startswith(prefix):
            continue
        if contains and contains not in name:
            continue
        out.append(name)
    return out


@dataclass(frozen=True)
class Row:
    data: dict[str, Any]


def _build_row(run_dir: Path) -> Row:
    cfg = _load_json(run_dir / "config_resolved.json") or {}
    base: dict[str, Any] = {
        "run_name": run_dir.parent.name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": "ok",
        "model": (cfg.get("model") or {}).get("name_or_path"),
        "tensor_parallel_size": (cfg.get("model") or {}).get("tensor_parallel_size"),
        "dataset": (cfg.get("task") or {}).get("dataset"),
        "train_split": (cfg.get("task") or {}).get("train_split"),
        "eval_split": (cfg.get("task") or {}).get("eval_split"),
        "max_train_examples": (cfg.get("task") or {}).get("max_train_examples"),
        "max_eval_examples": (cfg.get("task") or {}).get("max_eval_examples"),
        "T_max_cfg": (cfg.get("decode") or {}).get("max_new_tokens"),
        "methods": ",".join(str(m) for m in ((cfg.get("eval") or {}).get("methods") or [])),
    }
    mr = _read_main_results(run_dir / "tables" / "main_results_single.csv")
    if mr is None:
        base["status"] = "missing_main_results"
        return Row(data=base)

    # Merge, but keep metadata columns stable.
    for k, v in mr.items():
        if k in base:
            continue
        base[k] = v
    return Row(data=base)


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
    ap.add_argument("--run-name-prefix", type=str, default=None, help="Only include run_name with this prefix.")
    ap.add_argument("--contains", type=str, default=None, help="Only include run_name containing this substring.")
    ap.add_argument("--run-id", type=str, default=None, help="Only include this run_id under each run_name.")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: outputs/_main/<run_id>/main_summary.csv when --run-id is set).",
    )
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    run_names = _iter_run_names(outputs_root, args.run_name_prefix, args.contains)

    rows: list[Row] = []
    for rn in run_names:
        run_root = outputs_root / rn
        if not run_root.is_dir():
            continue
        if args.run_id:
            rid = str(args.run_id)
            run_dir = run_root / rid
            if run_dir.is_dir():
                rows.append(_build_row(run_dir))
            continue
        for p in sorted(run_root.iterdir()):
            if not p.is_dir():
                continue
            rows.append(_build_row(p))

    def _sort_key(r: Row) -> tuple[int, str, str]:
        status = str(r.data.get("status") or "")
        ok_rank = 0 if status == "ok" else 1
        return (ok_rank, str(r.data.get("dataset") or ""), str(r.data.get("run_name") or ""))

    rows.sort(key=_sort_key)

    if args.out:
        out_path = Path(args.out)
    elif args.run_id:
        out_path = outputs_root / "_main" / str(args.run_id) / "main_summary.csv"
    else:
        out_path = outputs_root / "_main" / "main_summary.csv"

    _write_csv(out_path, rows)
    print(f"[ok] wrote: {out_path} (rows={len(rows)})")


if __name__ == "__main__":
    main()

