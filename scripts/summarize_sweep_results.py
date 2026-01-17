#!/usr/bin/env python3
"""
Summarize grid / sweep runs under outputs/<run_name>/<run_id>/ into one CSV.

This script is designed to work with sweep scripts like:
- scripts/run_grid_sweep.sh

Typical usage:
  python scripts/summarize_sweep_results.py --run-name gs_stage1_math500_deepseek_1p5b
  python scripts/summarize_sweep_results.py --run-name gs_online_math500_deepseek_1p5b
  python scripts/summarize_sweep_results.py --run-name gs_full_math500_deepseek_1p5b
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
import re
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


_RE_TOKENS_USED = re.compile(r"\"tokens_used\"\s*:\s*(\d+)")
_RE_BUDGET_USED = re.compile(r"\"budget_used\"\s*:\s*(\d+)")
_RE_PROBE_USED = re.compile(r"\"probe_tokens_used\"\s*:\s*(\d+)")
_RE_FINISH_REASON = re.compile(r"\"finish_reason\"\s*:\s*(null|\"(?:\\\\.|[^\"])*\")")

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


def _percentile_int(xs: list[int], p: float) -> int | None:
    if not xs:
        return None
    xs_sorted = sorted(int(x) for x in xs)
    if len(xs_sorted) == 1:
        return int(xs_sorted[0])
    # Nearest-rank on [0, n-1] (dependency-free; sufficient for sweep summaries).
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

    raw_model = _get(cfg, "model.name_or_path")
    base["model"] = raw_model
    base["models"] = _canonical_model_name(raw_model, fallback_texts=[run_name, run_id])

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

    # Token/budget stats (best-effort; from per_example.jsonl).
    base.update(
        {
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
        }
    )
    per_example_path = summary_path.parent / "per_example.jsonl" if summary_path is not None else None
    if per_example_path is not None and per_example_path.exists():
        base["eval_per_example_path"] = str(per_example_path)
        try:
            base.update(_read_token_stats_from_per_example(per_example_path, T_max=base.get("T_max")))
        except Exception:
            pass

    # Key hyperparameters (current run config).
    base["offline_K"] = _safe_int(_get(cfg, "offline_mine.K"))
    base["offline_eta0"] = _safe_float(_get(cfg, "offline_mine.eta0"))
    cand_layers = _get(cfg, "offline_mine.candidate_layers")
    if isinstance(cand_layers, list):
        base["offline_candidate_layers"] = ",".join(str(x).strip() for x in cand_layers if str(x).strip() != "")
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
                    str(x).strip() for x in art_layers if str(x).strip() != ""
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
