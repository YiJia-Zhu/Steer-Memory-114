#!/usr/bin/env python3
"""
Aggregate outputs/_grid/<run_name>/grid_summary.csv into a single CSV.

Typical usage:
  python scripts/summarize_grid_summaries.py
  python scripts/summarize_grid_summaries.py --grid-root outputs/_grid --out outputs/_grid/all_grid_summary.csv
  python scripts/summarize_grid_summaries.py --contains qwen_7b
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable


def _iter_grid_summary_paths(grid_root: Path, *, contains: str | None) -> list[tuple[str, Path]]:
    if not grid_root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for p in sorted(grid_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("_"):
            continue
        if contains and contains not in name:
            continue
        csv_path = p / "grid_summary.csv"
        if csv_path.exists():
            out.append((name, csv_path))
    return out


def _read_grid_csv(csv_path: Path, *, run_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if not row.get("run_name"):
                row["run_name"] = run_name
            rows.append(row)
    if "run_name" not in fieldnames:
        fieldnames.insert(0, "run_name")
    return rows, fieldnames


def _merge_fieldnames(fieldnames: list[str], incoming: Iterable[str]) -> None:
    seen = set(fieldnames)
    for name in incoming:
        if name in seen:
            continue
        fieldnames.append(name)
        seen.add(name)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", type=str, default="outputs/_grid")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--contains", type=str, default=None, help="Only include grid folders containing this substring.")
    args = ap.parse_args()

    grid_root = Path(args.grid_root)
    paths = _iter_grid_summary_paths(grid_root, contains=args.contains)

    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for run_name, csv_path in paths:
        cur_rows, cur_fields = _read_grid_csv(csv_path, run_name=run_name)
        _merge_fieldnames(fieldnames, cur_fields)
        rows.extend(cur_rows)

    out_path = Path(args.out) if args.out else (grid_root / "all_grid_summary.csv")
    _write_csv(out_path, rows, fieldnames)

    print(f"[ok] wrote: {out_path}")
    print(f"[stats] grids={len(paths)} rows={len(rows)}")


if __name__ == "__main__":
    main()
