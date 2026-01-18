#!/usr/bin/env python3
"""
Summarize "main experiments" runs into one CSV.

This script aggregates:
  outputs/<run_name>/<run_id>/tables/main_results_single.csv
and enriches rows with model/dataset metadata from:
  outputs/<run_name>/<run_id>/config_resolved.json

Typical usage:
  python scripts/summarize_main_experiments.py --run-id 20260116_192320 --run-name-prefix main
  python scripts/summarize_main_experiments.py --run-name-prefix grid --contains ds_r1_qwen_1p5b

Default output:
  outputs/_main/<run_id>/main_summary.csv   (when --run-id is provided)
  outputs/_main/main_summary.csv           (otherwise)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _percentile_int(xs: list[int], p: float) -> int | None:
    if not xs:
        return None
    xs_sorted = sorted(int(x) for x in xs)
    if len(xs_sorted) == 1:
        return int(xs_sorted[0])
    # Nearest-rank on [0, n-1] (dependency-free; sufficient for summaries).
    k = int(round((float(p) / 100.0) * float(len(xs_sorted) - 1)))
    k = max(0, min(int(k), len(xs_sorted) - 1))
    return int(xs_sorted[k])


def _safe_mean_int(xs: list[int]) -> float | None:
    if not xs:
        return None
    return float(sum(int(x) for x in xs)) / float(len(xs))


def _parse_finish_reason(raw: str) -> str | None:
    s = str(raw).strip()
    if s == "" or s == "null":
        return None
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        # Minimal unquoting; finish_reason is typically "stop"/"length"/"eos".
        s = s[1:-1]
        s = s.replace('\\"', '"').replace("\\\\", "\\")
    return s


_RE_TOKENS_USED = re.compile(r"\"tokens_used\"\s*:\s*(\d+)")
_RE_BUDGET_USED = re.compile(r"\"budget_used\"\s*:\s*(\d+)")
_RE_PROBE_USED = re.compile(r"\"probe_tokens_used\"\s*:\s*(\d+)")
_RE_FINISH_REASON = re.compile(r"\"finish_reason\"\s*:\s*(null|\"(?:\\\\.|[^\"])*\")")
_RE_EVAL_METHOD_T = re.compile(r"^(?P<method>[A-Za-z0-9_-]+)_T(?P<T>\d+)$")

_MODEL_NAME_MAP = {
    "deepseek-r1-distill-qwen-1.5b": "DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-Distill-Qwen-7B",
    "ds-r1-qwen-1.5b": "DeepSeek-R1-Distill-Qwen-1.5B",
    "ds-r1-qwen-7b": "DeepSeek-R1-Distill-Qwen-7B",
    "qwen2.5-3b-instruct": "Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct",
    "qwen2.5-3b": "Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen2.5-7B-Instruct",
}


def _normalize_model_key(s: str) -> str:
    low = s.strip().lower()
    low = low.replace("\\", "/")
    low = low.replace("_", "-")
    low = low.replace(" ", "")
    low = low.replace("2p5", "2.5")
    low = low.replace("1p5", "1.5")
    low = re.sub(r"-+", "-", low)
    return low


def _canonical_model_name(raw: Any, *, fallback_texts: list[str] | None = None) -> str | None:
    if raw is not None:
        s = str(raw).strip()
        if s:
            name = s.replace("\\", "/").split("/")[-1]
            key = _normalize_model_key(name)
            canon = _MODEL_NAME_MAP.get(key)
            return canon if canon is not None else name
    if fallback_texts:
        for text in fallback_texts:
            if not text:
                continue
            key = _normalize_model_key(str(text))
            for alias, canon in _MODEL_NAME_MAP.items():
                if alias in key:
                    return canon
    return None


def _read_token_stats_from_per_example(per_example_path: Path, *, T_max: int | None) -> dict[str, Any]:
    """
    Extract token/budget usage stats from eval/*/per_example.jsonl.

    Best-effort and streaming: avoids json parsing (ESM rows can be very large due to `steps` and `text`).
    """
    toks: list[int] = []
    buds: list[int] = []
    probes: list[int] = []
    overhead: list[int] = []

    n = 0
    finish_seen = 0
    trunc = 0

    with per_example_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n += 1

            m_tok = _RE_TOKENS_USED.search(s)
            tok = int(m_tok.group(1)) if m_tok else 0
            toks.append(int(tok))

            m_fr = _RE_FINISH_REASON.search(s)
            fr = _parse_finish_reason(m_fr.group(1)) if m_fr else None
            if m_fr:
                finish_seen += 1

            is_trunc = False
            if fr is not None:
                is_trunc = str(fr).lower() == "length"
            elif T_max is not None:
                is_trunc = int(tok) >= int(T_max)
            trunc += int(is_trunc)

            m_b = _RE_BUDGET_USED.search(s)
            m_p = _RE_PROBE_USED.search(s)
            if m_b and m_p:
                b = int(m_b.group(1))
                p = int(m_p.group(1))
                buds.append(int(b))
                probes.append(int(p))
                overhead.append(max(0, int(b) - int(tok)))

    out: dict[str, Any] = {
        "n_per_example": int(n),
        "tokens_used_mean": _safe_mean_int(toks),
        "tokens_used_p50": _percentile_int(toks, 50),
        "tokens_used_p90": _percentile_int(toks, 90),
        "tokens_used_max": int(max(toks)) if toks else None,
        "trunc_rate": (float(trunc) / float(max(1, n))) if n > 0 else None,
        "finish_reason_seen_rate": (float(finish_seen) / float(max(1, n))) if n > 0 else None,
    }

    if len(buds) == n and n > 0:
        out.update(
            {
                "budget_used_mean": _safe_mean_int(buds),
                "budget_used_p90": _percentile_int(buds, 90),
                "budget_used_max": int(max(buds)) if buds else None,
                "probe_tokens_used_mean": _safe_mean_int(probes),
                "probe_tokens_used_p90": _percentile_int(probes, 90),
                "probe_tokens_used_max": int(max(probes)) if probes else None,
                "overhead_mean": _safe_mean_int(overhead),
                "overhead_p90": _percentile_int(overhead, 90),
                "overhead_max": int(max(overhead)) if overhead else None,
            }
        )
    return out


def _find_esm_per_example(run_dir: Path, T_max: int | None) -> Path | None:
    """
    Return per_example.jsonl under eval/*_T*/ that best matches the eval method.

    Prefers methods listed in config (when available), otherwise defaults to ESM/greedy.
    Tries to match T_max exactly; falls back to the closest (and then largest) T value.
    """
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        return None

    def _parse_eval_dir(d: Path) -> tuple[str | None, int | None]:
        m = _RE_EVAL_METHOD_T.match(d.name)
        if not m:
            return (None, None)
        method = m.group("method")
        try:
            t_val = int(m.group("T"))
        except Exception:
            t_val = None
        return (method, t_val)

    methods_pref: list[str] = []
    cfg = _load_json(run_dir / "config_resolved.json") or {}
    for m in (cfg.get("eval") or {}).get("methods") or []:
        methods_pref.append(str(m).lower())
    for m in ["esm", "greedy"]:
        if m not in methods_pref:
            methods_pref.append(m)

    def _method_rank(method: str | None) -> int:
        if method is None:
            return len(methods_pref)
        try:
            return methods_pref.index(method.lower())
        except ValueError:
            return len(methods_pref)

    candidates: list[tuple[tuple[int, int, int], Path]] = []
    for p in eval_dir.iterdir():
        if not p.is_dir():
            continue
        method, t_val = _parse_eval_dir(p)
        per_example = p / "per_example.jsonl"
        if method is None or not per_example.exists():
            continue
        pref_rank = _method_rank(method)
        t_penalty = abs(int(t_val) - int(T_max)) if (T_max is not None and t_val is not None) else 0
        t_rank = -int(t_val) if t_val is not None else 0
        candidates.append(((pref_rank, t_penalty, t_rank), per_example))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


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
    raw_model = (cfg.get("model") or {}).get("name_or_path")
    model_name = _canonical_model_name(raw_model, fallback_texts=[run_dir.parent.name, run_dir.name])
    base: dict[str, Any] = {
        "run_name": run_dir.parent.name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": "ok",
        "model": raw_model,
        "models": model_name,
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

    # Token/budget stats (best-effort; from per_example.jsonl).
    for k, v in {
        "eval_per_example_path": None,
        "n_per_example": None,
        "tokens_used_mean": None,
        "tokens_used_p50": None,
        "tokens_used_p90": None,
        "tokens_used_max": None,
        "budget_used_mean": None,
        "budget_used_p90": None,
        "budget_used_max": None,
        "probe_tokens_used_mean": None,
        "probe_tokens_used_p90": None,
        "probe_tokens_used_max": None,
        "overhead_mean": None,
        "overhead_p90": None,
        "overhead_max": None,
        "trunc_rate": None,
        "finish_reason_seen_rate": None,
    }.items():
        if k not in base:
            base[k] = v

    t_max = _safe_int(base.get("T_max"))
    if t_max is None:
        t_max = _safe_int(base.get("T_max_cfg"))
    per_example_path = _find_esm_per_example(run_dir, t_max)
    if per_example_path is not None and per_example_path.exists():
        base["eval_per_example_path"] = str(per_example_path)
        try:
            base.update(_read_token_stats_from_per_example(per_example_path, T_max=t_max))
        except Exception:
            pass
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
