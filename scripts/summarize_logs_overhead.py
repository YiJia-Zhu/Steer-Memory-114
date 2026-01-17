#!/usr/bin/env python3
"""
Summarize grid log files into a CSV with model, dataset, params, accuracy, and wall time.

Example:
  python scripts/summarize_logs_overhead.py --log-dir logs/grid_full_20260116_224426
  python scripts/summarize_logs_overhead.py --log-dir logs/grid_full_20260116_224426 --out /tmp/grid.csv
  python scripts/summarize_logs_overhead.py --outputs-root outputs --log-file stdout.log
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from pathlib import Path


_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| ")
_ACC_RE = re.compile(r"([A-Za-z0-9_.-]+) eval done\. n=(\d+)\s+acc=([0-9.]+)")
_MODEL_RE = re.compile(r"model(?:=|':)\s*'([^']+)'")
_DATASET_RE = re.compile(r"Loaded \d+ examples for mining \(([^/]+)/")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _parse_param_value(raw: str) -> float | None:
    if not raw:
        return None
    s = raw.replace("p", ".")
    try:
        return float(s)
    except Exception:
        return None


def _parse_filename(path: Path) -> dict[str, str | float | None]:
    stem = path.stem
    idx = ""
    rest = stem
    if "__" in stem:
        idx, rest = stem.split("__", 1)
    tokens = rest.split("_")

    date_idx = None
    for i in range(len(tokens) - 1):
        if re.fullmatch(r"\d{8}", tokens[i]) and re.fullmatch(r"\d{6}", tokens[i + 1]):
            date_idx = i
            break

    run_id = None
    param_tokens: list[str] = []
    if date_idx is not None:
        run_id = f"{tokens[date_idx]}_{tokens[date_idx + 1]}"
        param_tokens = tokens[date_idx + 2 :]
        if param_tokens:
            run_id = f"{run_id}__{'_'.join(param_tokens)}"

    params_raw = "_".join(param_tokens) if param_tokens else None
    layer_spec = None
    eta = None
    ks = None
    for tok in param_tokens:
        if tok.startswith("l"):
            layer_spec = _parse_param_value(tok[1:])
        elif tok.startswith("eta"):
            eta = _parse_param_value(tok[3:])
        elif tok.startswith("ks"):
            ks = _parse_param_value(tok[2:])

    return {
        "idx": idx,
        "run_id": run_id,
        "params": params_raw,
        "layer_spec": layer_spec,
        "eta": eta,
        "ks": ks,
    }


def _run_id_from_path(path: Path) -> str | None:
    if path.parent.name == "logs":
        return path.parent.parent.name
    return None


def _parse_run_id_params(run_id: str | None) -> dict[str, str | float | None]:
    if not run_id:
        return {"params": None, "layer_spec": None, "eta": None, "ks": None}

    params_raw = None
    layer_spec = None
    eta = None
    ks = None

    if "__" in run_id:
        _, suffix = run_id.split("__", 1)
        params_raw = suffix
        for tok in suffix.split("_"):
            if not tok:
                continue
            low = tok.lower()
            if low.startswith("l"):
                layer_spec = _parse_param_value(tok[1:])
            elif low.startswith("eta"):
                eta = _parse_param_value(tok[3:])
            elif low.startswith("ks"):
                ks = _parse_param_value(tok[2:])

    return {
        "params": params_raw,
        "layer_spec": layer_spec,
        "eta": eta,
        "ks": ks,
    }


def _parse_name_info(path: Path) -> dict[str, str | float | None]:
    info = _parse_filename(path)
    run_id = info.get("run_id") or _run_id_from_path(path)
    params_info = _parse_run_id_params(run_id)

    if info.get("params") is None:
        info["params"] = params_info.get("params")
    if info.get("layer_spec") is None:
        info["layer_spec"] = params_info.get("layer_spec")
    if info.get("eta") is None:
        info["eta"] = params_info.get("eta")
    if info.get("ks") is None:
        info["ks"] = params_info.get("ks")
    info["run_id"] = run_id
    return info


def _collect_outputs_logs(outputs_root: Path, log_file: str | None) -> tuple[list[Path], str]:
    candidates = [log_file] if log_file else ["stdout.log", "run.log"]
    for lf in candidates:
        logs = sorted(outputs_root.rglob(f"logs/{lf}"))
        if logs:
            return logs, lf
    return [], (log_file or "stdout.log")


def _parse_log(path: Path) -> dict[str, str | float | int | None]:
    model_path = None
    dataset = None

    start_ts: dt.datetime | None = None
    end_ts: dt.datetime | None = None

    last_any: tuple[str, int, float] | None = None
    last_esm: tuple[str, int, float] | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_clean = _strip_ansi(line.strip())

            m_ts = _TS_RE.match(line_clean)
            if m_ts:
                try:
                    ts = dt.datetime.strptime(m_ts.group(1), "%Y-%m-%d %H:%M:%S,%f")
                except Exception:
                    ts = None
                if ts is not None:
                    if start_ts is None:
                        start_ts = ts
                    end_ts = ts

            if dataset is None:
                m_ds = _DATASET_RE.search(line_clean)
                if m_ds:
                    dataset = m_ds.group(1)

            if model_path is None:
                m_model = _MODEL_RE.search(line_clean)
                if m_model:
                    model_path = m_model.group(1)

            m_acc = _ACC_RE.search(line_clean)
            if m_acc:
                tag = m_acc.group(1)
                n = int(m_acc.group(2))
                acc = float(m_acc.group(3))
                last_any = (tag, n, acc)
                if tag.upper().startswith("ESM"):
                    last_esm = (tag, n, acc)

    acc_tag = None
    acc_n = None
    acc_val = None
    if last_esm is not None:
        acc_tag, acc_n, acc_val = last_esm
    elif last_any is not None:
        acc_tag, acc_n, acc_val = last_any

    duration_seconds = None
    if start_ts is not None and end_ts is not None:
        duration_seconds = float((end_ts - start_ts).total_seconds())

    return {
        "model_path": model_path,
        "model": Path(model_path).name if model_path else None,
        "dataset": dataset,
        "acc": acc_val,
        "n": acc_n,
        "acc_tag": acc_tag,
        "start_time": start_ts.isoformat(sep=" ") if start_ts else None,
        "end_time": end_ts.isoformat(sep=" ") if end_ts else None,
        "duration_seconds": duration_seconds,
    }


def _status_for_row(row: dict[str, str | float | int | None]) -> str:
    missing = []
    if row.get("acc") is None:
        missing.append("missing_acc")
    if row.get("duration_seconds") is None:
        missing.append("missing_time")
    if row.get("model") is None:
        missing.append("missing_model")
    if row.get("dataset") is None:
        missing.append("missing_dataset")
    return "ok" if not missing else ",".join(missing)


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize grid logs into CSV.")
    p.add_argument("--log-dir", type=str, default=None, help="Directory containing *.log files (flat layout).")
    p.add_argument("--outputs-root", type=str, default=None, help="Outputs root to scan for per-run logs.")
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log filename under outputs/<run_name>/<run_id>/logs/ (default: stdout.log, fallback run.log).",
    )
    p.add_argument("--pattern", type=str, default="*.log", help="Glob pattern for log files in --log-dir.")
    p.add_argument("--out", type=str, default=None, help="Output CSV path.")
    args = p.parse_args()

    if not args.log_dir and not args.outputs_root:
        raise SystemExit("must provide --log-dir or --outputs-root")

    logs: list[Path] = []
    if args.outputs_root:
        outputs_root = Path(args.outputs_root)
        if not outputs_root.is_dir():
            raise SystemExit(f"outputs root not found: {outputs_root}")
        logs, used_log = _collect_outputs_logs(outputs_root, args.log_file)
        if not logs:
            raise SystemExit(f"no log files found under outputs: {outputs_root} (log_file={args.log_file})")
        if args.log_file is None and used_log != "stdout.log":
            print(f"[note] no stdout.log found; using {used_log} instead")
        out_path = Path(args.out) if args.out else (outputs_root / "grid_log_summary.csv")
    else:
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            raise SystemExit(f"log dir not found: {log_dir}")
        logs = sorted(log_dir.glob(args.pattern))
        if not logs:
            raise SystemExit(f"no log files found under: {log_dir} (pattern={args.pattern})")
        out_path = Path(args.out) if args.out else (log_dir / "grid_log_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float | int | None]] = []
    for log_path in logs:
        name_info = _parse_name_info(log_path)
        log_info = _parse_log(log_path)
        row = {
            "log_file": log_path.name,
            "run_id": name_info.get("run_id"),
            "model": log_info.get("model"),
            "model_path": log_info.get("model_path"),
            "dataset": log_info.get("dataset"),
            "params": name_info.get("params"),
            "layer_spec": name_info.get("layer_spec"),
            "eta": name_info.get("eta"),
            "ks": name_info.get("ks"),
            "acc": log_info.get("acc"),
            "n": log_info.get("n"),
            "acc_tag": log_info.get("acc_tag"),
            "start_time": log_info.get("start_time"),
            "end_time": log_info.get("end_time"),
            "duration_seconds": log_info.get("duration_seconds"),
        }
        row["status"] = _status_for_row(row)
        rows.append(row)

    fieldnames = [
        "log_file",
        "run_id",
        "model",
        "model_path",
        "dataset",
        "params",
        "layer_spec",
        "eta",
        "ks",
        "acc",
        "n",
        "acc_tag",
        "start_time",
        "end_time",
        "duration_seconds",
        "status",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
