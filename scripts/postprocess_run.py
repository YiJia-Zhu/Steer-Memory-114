#!/usr/bin/env python3
"""
One-stop postprocessor for a single run directory.

Outputs:
- <run_dir>/report.md        : Markdown summary (tables, diag, cases, logs, figures)
- <run_dir>/figures/dashboard.png : Quick overall dashboard (acc vs budget + key diagnostics)
- <run_dir>/figures/offline_*.png : Offline stage summaries (mine/library/memory) when available
- <run_dir>/figures/eval_analysis_T*.png : Per-budget eval analysis (token usage, overhead, tool usage)
- <run_dir>/tables/postprocess_*.csv : Extra analysis tables (tool usage, control-point stats, example diffs)

Usage:
  python scripts/postprocess_run.py --run-dir outputs/gsm8k_recommended_small/<RUN_ID>
  # or
  python scripts/postprocess_run.py --run-name gsm8k_recommended_small --run-id latest
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| ")


def _resolve_run_dir(run_name: str, run_id: str, outputs_root: Path) -> Path:
    if run_id.lower() == "latest":
        latest_path = outputs_root / run_name / "LATEST"
        rid = latest_path.read_text(encoding="utf-8").strip()
        if not rid:
            raise ValueError(f"{latest_path} is empty; cannot resolve run_id.")
        run_id = rid
    return outputs_root / run_name / run_id


def _load_csv(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            rows.append(row)
    return rows


def _load_acc_vs_budget(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    curves: Dict[str, List[Tuple[int, float]]] = {}
    rows = _load_csv(path)
    if not rows:
        return curves
    header = rows[0]
    budget_cols = [(i, h) for i, h in enumerate(header) if h.startswith("T")]
    for row in rows[1:]:
        if not row:
            continue
        method = row[0]
        pts: List[Tuple[int, float]] = []
        for i, h in budget_cols:
            try:
                acc = float(row[i])
            except Exception:
                continue
            budget = int(h[1:])
            pts.append((budget, acc))
        curves[method] = sorted(pts, key=lambda x: x[0])
    return curves


def _load_latest_diag(tables_dir: Path) -> Dict[str, Dict[str, float]]:
    diag_files = sorted(tables_dir.glob("diagnostics_T*.csv"))
    if not diag_files:
        return {}
    latest = diag_files[-1]
    by_method: Dict[str, Dict[str, float]] = {}
    with latest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get("method", "")
            if not m:
                continue
            by_method[m] = {
                "trunc_rate": float(row.get("trunc_rate", 0.0)),
                "overhead_mean": float(row.get("overhead_mean", 0.0)),
                "tool_step_rate": float(row.get("tool_step_rate", 0.0)),
                "tokens_mean": float(row.get("tokens_mean", 0.0)),
                "acc": float(row.get("acc", 0.0)),
            }
    return by_method


def make_dashboard(run_dir: Path) -> Path | None:
    if plt is None:
        return None
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    diag = _load_latest_diag(tables_dir)
    main_csv = tables_dir / "main_results_single.csv"
    main_T: int | None = None
    acc_by_method: Dict[str, float] = {}
    if main_csv.exists():
        rows = _load_csv(main_csv)
        if len(rows) >= 2:
            header = rows[0]
            row = rows[1]
            if "T_max" in header:
                try:
                    main_T = int(row[header.index("T_max")])
                except Exception:
                    main_T = None
            for i in range(3, min(len(header), len(row))):
                try:
                    acc_by_method[header[i]] = float(row[i])
                except Exception:
                    continue
    if not acc_by_method and diag:
        acc_by_method = {m: float(diag[m].get("acc", 0.0)) for m in diag.keys()}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Accuracy (single T_max)
    ax = axes[0][0]
    if acc_by_method:
        methods = list(acc_by_method.keys())
        vals = [acc_by_method[m] for m in methods]
        ax.bar(methods, vals)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy @ T_max={main_T}" if main_T is not None else "Accuracy")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "main_results_single.csv not found", ha="center", va="center")
        ax.axis("off")

    def _bar(ax, metric: str, title: str, pct: bool = False) -> None:
        if not diag:
            ax.text(0.5, 0.5, "diagnostics not found", ha="center", va="center")
            ax.axis("off")
            return
        methods = list(diag.keys())
        vals = [diag[m].get(metric, 0.0) * (100 if pct else 1) for m in methods]
        ax.bar(methods, vals)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if pct:
            ax.set_ylabel("%")
        else:
            ax.set_ylabel(metric)

    _bar(axes[0][1], "trunc_rate", "Truncation Rate", pct=True)
    _bar(axes[1][0], "overhead_mean", "Probe Overhead (tokens)")
    _bar(axes[1][1], "tool_step_rate", "Tool Step Rate", pct=True)

    fig.tight_layout()
    out_path = figures_dir / "dashboard.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _percentile(xs: List[int], p: float) -> int:
    if not xs:
        return 0
    ys = sorted(int(x) for x in xs)
    if len(ys) == 1:
        return int(ys[0])
    k = (float(p) / 100.0) * float(len(ys) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return int(ys[lo])
    w = float(k - lo)
    return int(round((1.0 - w) * float(ys[lo]) + w * float(ys[hi])))


def _read_config(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "config_resolved.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _plot_bar_counts(ax: Any, counts: Dict[int, int], *, title: str, xlabel: str) -> None:
    if not counts:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.axis("off")
        return
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]
    ax.bar([str(x) for x in xs], ys)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", alpha=0.3)


def analyze_offline_mine(run_dir: Path) -> Tuple[Path | None, dict[str, Any]]:
    mine_dir = run_dir / "mine"
    cand_path = mine_dir / "candidates.jsonl"
    roll_path = mine_dir / "rollouts.jsonl"
    if not cand_path.exists():
        return None, {}
    if plt is None:
        return None, {}

    qualities: List[float] = []
    m_counts: Dict[int, int] = {}
    layer_counts: Dict[int, int] = {}
    for r in _iter_jsonl(cand_path):
        q = _safe_float(r.get("quality", 0.0))
        qualities.append(float(q))
        mm = _safe_int(r.get("control_point_m", -1))
        if mm > 0:
            m_counts[mm] = m_counts.get(mm, 0) + 1
        layer = _safe_int(r.get("layer", -1))
        if layer >= 0:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

    rollout_correct = 0
    rollout_total = 0
    any_correct_by_ex: Dict[str, bool] = {}
    rewards_correct: List[float] = []
    rewards_incorrect: List[float] = []
    if roll_path.exists():
        for r in _iter_jsonl(roll_path):
            if str(r.get("source", "rollout")) != "rollout":
                continue
            rollout_total += 1
            ex_id = str(r.get("example_id", ""))
            c = bool(r.get("correct"))
            if c:
                rollout_correct += 1
            any_correct_by_ex[ex_id] = bool(any_correct_by_ex.get(ex_id, False) or c)
            rew = _safe_float(r.get("reward", 0.0))
            (rewards_correct if c else rewards_incorrect).append(float(rew))

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "offline_mine.png"

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0][0]
    if qualities:
        ax.hist(qualities, bins=30, alpha=0.85)
        ax.set_title(f"Stage I candidates quality (N={len(qualities)})")
        ax.set_xlabel("quality")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no candidates", ha="center", va="center")
        ax.axis("off")

    _plot_bar_counts(axes[0][1], m_counts, title="Candidates by control_point_m", xlabel="m")

    # Layers: keep top-20 bars for readability
    ax = axes[1][0]
    if layer_counts:
        items = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        xs = [str(k) for k, _v in items]
        ys = [int(v) for _k, v in items]
        ax.bar(xs, ys)
        ax.set_title("Candidates by layer (top-20)")
        ax.set_xlabel("layer")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.axis("off")

    ax = axes[1][1]
    if rollout_total > 0:
        ax.hist(rewards_incorrect, bins=30, alpha=0.6, label="incorrect")
        ax.hist(rewards_correct, bins=30, alpha=0.6, label="correct")
        ax.set_title("Rollout rewards (source=rollout)")
        ax.set_xlabel("reward")
        ax.grid(True, alpha=0.3)
        ax.legend()
        any_correct_rate = sum(1 for v in any_correct_by_ex.values() if v) / max(1, len(any_correct_by_ex))
        ax.text(
            0.02,
            0.98,
            f"rollout_correct_rate={rollout_correct/max(1,rollout_total):.3f}\\n"
            f"example_any_correct_rate={any_correct_rate:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    else:
        ax.text(0.5, 0.5, "rollouts.jsonl missing", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    summary = {
        "candidates": int(len(qualities)),
        "candidates_by_m": {int(k): int(v) for k, v in sorted(m_counts.items())},
        "candidates_by_layer_top20": {int(k): int(v) for k, v in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:20]},
        "rollout_total": int(rollout_total),
        "rollout_correct_rate": float(rollout_correct / max(1, rollout_total)) if rollout_total > 0 else None,
        "example_any_correct_rate": float(sum(1 for v in any_correct_by_ex.values() if v) / max(1, len(any_correct_by_ex))) if any_correct_by_ex else None,
    }
    return out_path, summary


def _try_load_vectors(vector_paths: List[str], max_vecs: int = 128) -> List[List[float]]:
    """
    Best-effort load unit vectors from local .pt paths to estimate diversity.
    Returns a list of float lists; empty if dependency is missing.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return []

    vecs: List[List[float]] = []
    for vp in vector_paths[: max_vecs]:
        try:
            obj = torch.load(str(vp), map_location="cpu", weights_only=False)
            if torch.is_tensor(obj):
                v = obj.detach().to(torch.float32).reshape(-1).tolist()
            else:
                v = torch.tensor(obj, dtype=torch.float32).reshape(-1).tolist()
            if not v:
                continue
            s2 = 0.0
            for x in v:
                s2 += float(x) * float(x)
            n = math.sqrt(max(1e-12, s2))
            vecs.append([float(x) / n for x in v])
        except Exception:
            continue
    return vecs


def _pairwise_cos_sims(vecs: List[List[float]]) -> List[float]:
    sims: List[float] = []
    for i in range(len(vecs)):
        vi = vecs[i]
        for j in range(i + 1, len(vecs)):
            vj = vecs[j]
            s = 0.0
            for a, b in zip(vi, vj):
                s += float(a) * float(b)
            sims.append(float(s))
    return sims


def analyze_library(run_dir: Path) -> Tuple[Path | None, dict[str, Any]]:
    lib_path = run_dir / "library" / "library.jsonl"
    if not lib_path.exists():
        return None, {}
    if plt is None:
        return None, {}

    qualities: List[float] = []
    m_counts: Dict[int, int] = {}
    layer_counts: Dict[int, int] = {}
    alphas: List[float] = []
    vec_paths: List[str] = []
    for r in _iter_jsonl(lib_path):
        qualities.append(_safe_float(r.get("quality", 0.0)))
        mm = _safe_int(r.get("control_point_m", -1))
        if mm > 0:
            m_counts[mm] = m_counts.get(mm, 0) + 1
        layer = _safe_int(r.get("layer", -1))
        if layer >= 0:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        alphas.append(_safe_float(r.get("alpha", r.get("scale", 0.0))))
        vp = r.get("vector_path", None)
        if vp:
            vec_paths.append(str(vp))

    vecs = _try_load_vectors(vec_paths, max_vecs=min(128, len(vec_paths)))
    sims = _pairwise_cos_sims(vecs) if len(vecs) >= 2 else []

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "offline_library.png"

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    _plot_bar_counts(axes[0][0], m_counts, title=f"Library by control_point_m (B={len(qualities)})", xlabel="m")

    ax = axes[0][1]
    if qualities:
        ax.hist(qualities, bins=30, alpha=0.85)
        ax.set_title("Library quality distribution")
        ax.set_xlabel("quality")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.axis("off")

    ax = axes[1][0]
    if layer_counts:
        items = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        xs = [str(k) for k, _v in items]
        ys = [int(v) for _k, v in items]
        ax.bar(xs, ys)
        ax.set_title("Library by layer (top-20)")
        ax.set_xlabel("layer")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.axis("off")

    ax = axes[1][1]
    if sims:
        ax.hist(sims, bins=30, alpha=0.85)
        ax.set_title(f"Vector cosine similarity (pairs={len(sims)})")
        ax.set_xlabel("cosine similarity")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.98,
            f"mean={sum(sims)/len(sims):.3f}\\nmax={max(sims):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    else:
        ax.hist(alphas, bins=20, alpha=0.85)
        ax.set_title("Alpha(scale) distribution (vector load skipped)")
        ax.set_xlabel("alpha")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    summary = {
        "library": int(len(qualities)),
        "library_by_m": {int(k): int(v) for k, v in sorted(m_counts.items())},
        "vector_similarity_pairs": int(len(sims)),
        "vector_similarity_mean": float(sum(sims) / len(sims)) if sims else None,
        "vector_similarity_max": float(max(sims)) if sims else None,
    }
    return out_path, summary


def analyze_memory(run_dir: Path) -> Tuple[Path | None, dict[str, Any]]:
    mem_dir = run_dir / "memory"
    ent_path = mem_dir / "entries.jsonl"
    stats_path = mem_dir / "tool_stats.jsonl"
    if not ent_path.exists():
        return None, {}
    if plt is None:
        return None, {}

    adv: List[float] = []
    m_counts: Dict[int, int] = {}
    tool_counts: Dict[str, int] = {}
    base_correct = 0
    cf_correct = 0
    for r in _iter_jsonl(ent_path):
        a = _safe_float(r.get("advantage", 0.0))
        adv.append(float(a))
        mm = _safe_int(r.get("control_point_m", -1))
        if mm > 0:
            m_counts[mm] = m_counts.get(mm, 0) + 1
        tn = str(r.get("tool_name", ""))
        if tn:
            tool_counts[tn] = tool_counts.get(tn, 0) + 1
        base_correct += int(bool(r.get("base_correct")))
        cf_correct += int(bool(r.get("cf_correct")))

    tool_stats: List[dict[str, Any]] = []
    if stats_path.exists():
        tool_stats = list(_iter_jsonl(stats_path))

    adv_pos = sum(1 for x in adv if x > 0)
    adv_neg = sum(1 for x in adv if x < 0)
    adv_zero = len(adv) - adv_pos - adv_neg

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "offline_memory.png"

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    _plot_bar_counts(axes[0][0], m_counts, title=f"Memory entries by control_point_m (N={len(adv)})", xlabel="m")

    ax = axes[0][1]
    if adv:
        ax.hist(adv, bins=40, alpha=0.85)
        ax.set_title("Advantage distribution")
        ax.set_xlabel("advantage")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.98,
            f"pos={adv_pos/len(adv):.3f} zero={adv_zero/len(adv):.3f} neg={adv_neg/len(adv):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    else:
        ax.text(0.5, 0.5, "empty", ha="center", va="center")
        ax.axis("off")

    ax = axes[1][0]
    if tool_counts:
        items = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        xs = [k for k, _v in items]
        ys = [int(v) for _k, v in items]
        ax.barh(xs[::-1], ys[::-1])
        ax.set_title("Top tools by memory count")
        ax.grid(True, axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.axis("off")

    ax = axes[1][1]
    if tool_stats:
        items = sorted(tool_stats, key=lambda r: _safe_float(r.get("adv_mean", 0.0)), reverse=True)[:10]
        xs = [str(r.get("tool_name", "")) for r in items]
        ys = [_safe_float(r.get("adv_mean", 0.0)) for r in items]
        ax.barh(xs[::-1], ys[::-1])
        ax.set_title("Top tools by adv_mean")
        ax.grid(True, axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "tool_stats missing", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    summary = {
        "memory_entries": int(len(adv)),
        "memory_by_m": {int(k): int(v) for k, v in sorted(m_counts.items())},
        "adv_pos_frac": float(adv_pos / max(1, len(adv))),
        "adv_zero_frac": float(adv_zero / max(1, len(adv))),
        "adv_neg_frac": float(adv_neg / max(1, len(adv))),
        "base_correct_frac": float(base_correct / max(1, len(adv))),
        "cf_correct_frac": float(cf_correct / max(1, len(adv))),
        "unique_tools": int(len(tool_counts)),
    }
    return out_path, summary


def _infer_budgets_from_eval_dir(eval_dir: Path) -> List[int]:
    budgets: set[int] = set()
    if not eval_dir.exists():
        return []
    for p in eval_dir.iterdir():
        m = re.match(r"^(?:greedy|ESM)_T(\d+)$", p.name)
        if m:
            budgets.add(int(m.group(1)))
    return sorted(budgets)


def analyze_eval(run_dir: Path, budgets: List[int]) -> Tuple[List[Path], dict[str, Any]]:
    eval_dir = run_dir / "eval"
    tables_dir = run_dir / "tables"
    figs_dir = run_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig_paths: List[Path] = []
    summary: dict[str, Any] = {"budgets": budgets, "points": {}}

    frontier_points: List[dict[str, Any]] = []

    for T in budgets:
        greedy_path = eval_dir / f"greedy_T{int(T)}" / "per_example.jsonl"
        esm_path = eval_dir / f"ESM_T{int(T)}" / "per_example.jsonl"
        if not greedy_path.exists() or not esm_path.exists():
            continue

        greedy: Dict[str, dict[str, Any]] = {}
        for r in _iter_jsonl(greedy_path):
            ex_id = str(r.get("example_id", ""))
            greedy[ex_id] = {
                "correct": bool(r.get("correct")),
                "tokens_used": _safe_int(r.get("tokens_used", 0)),
                "finish_reason": r.get("finish_reason", None),
            }

        esm: Dict[str, dict[str, Any]] = {}
        tool_counts: Dict[str, int] = {}
        cp_steps: Dict[int, dict[str, float]] = {}
        mem_reason_counts: Dict[str, int] = {}
        toks_e_all: List[int] = []
        buds_e_all: List[int] = []
        overhead_e_all: List[int] = []

        for r in _iter_jsonl(esm_path):
            ex_id = str(r.get("example_id", ""))
            tok = _safe_int(r.get("tokens_used", 0))
            bud = _safe_int(r.get("budget_used", 0))
            oh = max(0, bud - tok)
            toks_e_all.append(tok)
            buds_e_all.append(bud)
            overhead_e_all.append(oh)

            steps = r.get("steps", []) or []
            tool_steps = 0
            null_steps = 0
            for s in steps:
                phase = str(s.get("phase", "") or "")
                mm = _safe_int(s.get("mem_m", s.get("m", -1)))
                chosen = str(s.get("chosen", "null"))
                if phase == "tail":
                    continue
                if mm <= 0:
                    continue
                reason = str(s.get("mem_reason", "") or "")
                if not reason:
                    reason = "unknown"
                mem_reason_counts[reason] = mem_reason_counts.get(reason, 0) + 1
                st = cp_steps.setdefault(mm, {"total": 0.0, "tool": 0.0, "null": 0.0})
                st["total"] += 1.0
                if chosen == "null":
                    st["null"] += 1.0
                    null_steps += 1
                else:
                    st["tool"] += 1.0
                    tool_steps += 1
                    tool_counts[chosen] = tool_counts.get(chosen, 0) + 1

            esm[ex_id] = {
                "correct": bool(r.get("correct")),
                "tokens_used": tok,
                "budget_used": bud,
                "overhead": oh,
                "finish_reason": r.get("finish_reason", None),
                "tool_steps": int(tool_steps),
                "null_steps": int(null_steps),
                "steps": int(len(steps)),
            }

        # Example-level join + outcomes
        wins = loses = both_correct = both_wrong = 0
        correct_g = correct_e = 0
        ex_rows: List[List[Any]] = []
        toks_g: List[int] = []
        toks_e: List[int] = []
        buds_e: List[int] = []
        overhead_e: List[int] = []
        for ex_id, g in greedy.items():
            e = esm.get(ex_id, None)
            if e is None:
                continue
            g_ok = bool(g["correct"])
            e_ok = bool(e["correct"])
            correct_g += int(g_ok)
            correct_e += int(e_ok)
            toks_g.append(int(g["tokens_used"]))
            toks_e.append(int(e["tokens_used"]))
            buds_e.append(int(e["budget_used"]))
            overhead_e.append(int(e["overhead"]))

            if e_ok and not g_ok:
                outcome = "win"
                wins += 1
            elif (not e_ok) and g_ok:
                outcome = "lose"
                loses += 1
            elif e_ok and g_ok:
                outcome = "both_correct"
                both_correct += 1
            else:
                outcome = "both_wrong"
                both_wrong += 1

            ex_rows.append(
                [
                    ex_id,
                    int(g_ok),
                    int(e_ok),
                    int(g["tokens_used"]),
                    int(e["tokens_used"]),
                    int(e["budget_used"]),
                    int(e["overhead"]),
                    int(e["tool_steps"]),
                    int(e["null_steps"]),
                    int(e["steps"]),
                    outcome,
                ]
            )

        n = max(1, len(ex_rows))
        acc_g = correct_g / n
        acc_e = correct_e / n
        delta = acc_e - acc_g
        overhead_mean = sum(overhead_e) / max(1, len(overhead_e)) if overhead_e else 0.0
        tool_step_rate = (
            sum(v.get("tool", 0.0) for v in cp_steps.values()) / max(1.0, sum(v.get("total", 0.0) for v in cp_steps.values()))
            if cp_steps
            else 0.0
        )

        # Write extra tables
        out_examples = tables_dir / f"postprocess_examples_T{int(T)}.csv"
        with out_examples.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "example_id",
                    "greedy_correct",
                    "esm_correct",
                    "greedy_tokens_used",
                    "esm_tokens_used",
                    "esm_budget_used",
                    "esm_overhead",
                    "esm_tool_steps",
                    "esm_null_steps",
                    "esm_steps",
                    "outcome",
                ]
            )
            for row in ex_rows:
                w.writerow(row)

        out_tools = tables_dir / f"postprocess_tool_usage_T{int(T)}.csv"
        with out_tools.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tool_name", "chosen_count"])
            for tn, c in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
                w.writerow([tn, int(c)])

        out_reasons = tables_dir / f"postprocess_mem_reason_T{int(T)}.csv"
        with out_reasons.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mem_reason", "count", "frac"])
            total = max(1, sum(int(c) for c in mem_reason_counts.values()))
            for reason, c in sorted(mem_reason_counts.items(), key=lambda x: x[1], reverse=True):
                w.writerow([reason, int(c), float(int(c)) / float(total)])

        out_cp = tables_dir / f"postprocess_control_points_T{int(T)}.csv"
        with out_cp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["m", "total_steps", "tool_steps", "tool_step_rate", "null_steps", "null_step_rate"])
            for mm in sorted(cp_steps.keys()):
                st = cp_steps[mm]
                total = float(st.get("total", 0.0))
                tool = float(st.get("tool", 0.0))
                null = float(st.get("null", 0.0))
                w.writerow(
                    [
                        int(mm),
                        int(total),
                        int(tool),
                        float(tool / max(1.0, total)),
                        int(null),
                        float(null / max(1.0, total)),
                    ]
                )

        out_eff = tables_dir / f"postprocess_efficiency_T{int(T)}.csv"
        with out_eff.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "n", "acc", "correct", "tokens_sum", "tokens_mean", "tokens_per_correct", "budget_sum", "budget_mean", "budget_per_correct"])
            g_tokens_sum = int(sum(toks_g))
            e_tokens_sum = int(sum(toks_e)) if toks_e else 0
            e_budget_sum = int(sum(buds_e)) if buds_e else 0
            w.writerow(["greedy", n, acc_g, int(correct_g), g_tokens_sum, g_tokens_sum / n, g_tokens_sum / max(1, correct_g), "", "", ""])
            w.writerow(
                [
                    "esm",
                    n,
                    acc_e,
                    int(correct_e),
                    e_tokens_sum,
                    e_tokens_sum / n,
                    e_tokens_sum / max(1, correct_e),
                    e_budget_sum,
                    e_budget_sum / n,
                    e_budget_sum / max(1, correct_e),
                ]
            )

        # Figure (single-page per budget)
        if plt is not None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            ax = axes[0][0]
            if toks_g and toks_e:
                ax.hist(toks_g, bins=30, alpha=0.6, label="greedy committed")
                ax.hist(toks_e, bins=30, alpha=0.6, label="esm committed")
                ax.axvline(int(T), color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="T_max")
                ax.set_title("Committed tokens distribution")
                ax.set_xlabel("tokens_used")
                ax.grid(True, alpha=0.25)
                ax.legend()
            else:
                ax.text(0.5, 0.5, "missing per_example", ha="center", va="center")
                ax.axis("off")

            ax = axes[0][1]
            if overhead_e:
                ax.hist(overhead_e, bins=30, alpha=0.85)
                ax.set_title("ESM overhead (budget_used - tokens_used)")
                ax.set_xlabel("overhead tokens")
                ax.grid(True, alpha=0.25)
                ax.text(
                    0.02,
                    0.98,
                    f"mean={overhead_mean:.1f}\\n"
                    f"p50={_percentile(overhead_e,50)} p90={_percentile(overhead_e,90)}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )
            else:
                ax.text(0.5, 0.5, "no overhead", ha="center", va="center")
                ax.axis("off")

            ax = axes[1][0]
            if cp_steps:
                xs = sorted(cp_steps.keys())
                ys = [float(cp_steps[m]["tool"] / max(1.0, cp_steps[m]["total"])) for m in xs]
                ax.bar([str(x) for x in xs], [y * 100.0 for y in ys])
                ax.set_title("Tool step rate by control_point_m")
                ax.set_xlabel("m")
                ax.set_ylabel("%")
                ax.grid(True, axis="y", alpha=0.25)
            else:
                ax.text(0.5, 0.5, "no step logs", ha="center", va="center")
                ax.axis("off")

            ax = axes[1][1]
            if tool_counts:
                items = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                xs = [k for k, _v in items]
                ys = [int(v) for _k, v in items]
                ax.barh(xs[::-1], ys[::-1])
                ax.set_title("Top chosen tools (ESM)")
                ax.grid(True, axis="x", alpha=0.25)
            else:
                ax.text(0.5, 0.5, "all-null (no tools)", ha="center", va="center")
                ax.axis("off")

            fig.suptitle(
                f"T={int(T)}  acc_g={acc_g:.3f}  acc_esm={acc_e:.3f}  Δ={delta:+.3f}  "
                f"wins={wins} loses={loses}  tool_step_rate={tool_step_rate:.2f}",
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            out_fig = figs_dir / f"eval_analysis_T{int(T)}.png"
            fig.savefig(out_fig, dpi=200)
            plt.close(fig)
            fig_paths.append(out_fig)

        summary["points"][str(int(T))] = {
            "n": int(n),
            "acc_greedy": float(acc_g),
            "acc_esm": float(acc_e),
            "delta_acc": float(delta),
            "wins": int(wins),
            "loses": int(loses),
            "both_correct": int(both_correct),
            "both_wrong": int(both_wrong),
            "overhead_mean": float(overhead_mean),
            "tool_step_rate": float(tool_step_rate),
            "top_tools": [tn for tn, _c in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]],
        }

        frontier_points.append(
            {
                "T": int(T),
                "acc_greedy": float(acc_g),
                "acc_esm": float(acc_e),
                "greedy_tokens_mean": float(sum(toks_g) / max(1, len(toks_g))) if toks_g else 0.0,
                "esm_tokens_mean": float(sum(toks_e) / max(1, len(toks_e))) if toks_e else 0.0,
                "esm_budget_mean": float(sum(buds_e) / max(1, len(buds_e))) if buds_e else 0.0,
            }
        )

    # Efficiency frontier across budgets (if we have >=1 point)
    if frontier_points and plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # 1) committed tokens vs acc
        ax = axes[0]
        xs_g = [p["greedy_tokens_mean"] for p in frontier_points]
        ys_g = [p["acc_greedy"] for p in frontier_points]
        xs_e = [p["esm_tokens_mean"] for p in frontier_points]
        ys_e = [p["acc_esm"] for p in frontier_points]
        ax.plot(xs_g, ys_g, marker="o", label="greedy (committed)")
        ax.plot(xs_e, ys_e, marker="o", label="esm (committed)")
        for p in frontier_points:
            ax.annotate(f"T{p['T']}", (p["greedy_tokens_mean"], p["acc_greedy"]), fontsize=8, alpha=0.7)
        ax.set_title("Accuracy vs committed tokens")
        ax.set_xlabel("mean committed tokens")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend()

        # 2) total tokens (budget used) vs acc
        ax = axes[1]
        xs_g2 = xs_g
        ys_g2 = ys_g
        xs_e2 = [p["esm_budget_mean"] for p in frontier_points]
        ys_e2 = ys_e
        ax.plot(xs_g2, ys_g2, marker="o", label="greedy (total)")
        ax.plot(xs_e2, ys_e2, marker="o", label="esm (total=budget_used)")
        for p in frontier_points:
            ax.annotate(f"T{p['T']}", (p["esm_budget_mean"], p["acc_esm"]), fontsize=8, alpha=0.7)
        ax.set_title("Accuracy vs total tokens")
        ax.set_xlabel("mean total tokens")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend()

        fig.tight_layout()
        out_frontier = figs_dir / "efficiency_frontier.png"
        fig.savefig(out_frontier, dpi=200)
        plt.close(fig)
        fig_paths.append(out_frontier)

    return fig_paths, summary


def parse_run_log_timings(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "logs" / "run.log"
    if not log_path.exists():
        return {}
    stage_markers = {
        "mine": ("Stage I mining: start", "Stage I mining: done"),
        "select": ("Stage II selection: start", "Stage II selection: done"),
        "memory": ("Stage III memory: start", "Stage III memory: done"),
    }
    ts_by_msg: dict[str, List[dt.datetime]] = {}
    eval_first: dt.datetime | None = None
    eval_last: dt.datetime | None = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _TS_RE.match(line)
            if not m:
                continue
            try:
                ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
            except Exception:
                continue
            for stage, (start_s, end_s) in stage_markers.items():
                if start_s in line or end_s in line:
                    ts_by_msg.setdefault(stage, []).append(ts)
            if "Eval budget sweep" in line:
                eval_first = ts if eval_first is None else eval_first
                eval_last = ts
            if "Wrote sweep figure" in line or "Wrote ablation results" in line or "Wrote diagnostics" in line:
                eval_last = ts
    out: dict[str, Any] = {}
    for stage, (start_s, end_s) in stage_markers.items():
        # We don't know which line is start/end if multiple runs appended; take first/last.
        ts_list = ts_by_msg.get(stage, [])
        if len(ts_list) >= 2:
            dur = (max(ts_list) - min(ts_list)).total_seconds()
            out[stage] = {"seconds": float(dur)}
    if eval_first is not None and eval_last is not None:
        out["eval"] = {"seconds": float((eval_last - eval_first).total_seconds())}
    return out


def build_report(run_dir: Path) -> Path:
    lines: List[str] = []
    lines.append(f"# Run Report: {run_dir}")
    cfg = _read_config(run_dir)
    if cfg:
        lines.append("")
        lines.append(f"- dataset: `{cfg.get('task', {}).get('dataset', '')}` split: `{cfg.get('task', {}).get('eval_split', '')}`")
        lines.append(f"- T_max (decode.max_new_tokens): {cfg.get('decode', {}).get('max_new_tokens', '')}")
        lines.append(f"- prompt: `{cfg.get('prompt', {}).get('template', '')}`")
        lines.append(f"- run_name: `{cfg.get('outputs', {}).get('run_name', '')}`  run_id: `{cfg.get('outputs', {}).get('run_id', '')}`")

    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    cases_dir = run_dir / "cases"
    logs_dir = run_dir / "logs"

    def _append_csv(title: str, path: Path, head_rows: int = None) -> None:
        if not path.exists():
            return
        rows = _load_csv(path)
        lines.append("")
        lines.append(f"## {title}")
        if head_rows is None:
            show = rows
        else:
            show = rows[:head_rows]
        for row in show:
            lines.append("- " + ", ".join(row))

    _append_csv("acc_vs_budget.csv", tables_dir / "acc_vs_budget.csv")
    _append_csv("main_results_single.csv", tables_dir / "main_results_single.csv")
    _append_csv("ablation.csv", tables_dir / "ablation.csv")
    eff_files = sorted(tables_dir.glob("postprocess_efficiency_T*.csv"))
    if eff_files:
        _append_csv("postprocess_efficiency_T*.csv (latest)", eff_files[-1], head_rows=3)

    diag_files = sorted(tables_dir.glob("diagnostics_T*.csv"))
    if diag_files:
        lines.append("")
        lines.append("## diagnostics (latest per T)")
        p = diag_files[-1]
        rows = _load_csv(p)
        head = rows[:2] if rows else []
        lines.append(f"- {p.name}: " + (" | ".join([", ".join(r) for r in head]) if head else "empty"))

    cmp_files = sorted(tables_dir.glob("compare_T*.csv"))
    if cmp_files:
        lines.append("")
        lines.append("## method comparison (latest per T)")
        p = cmp_files[-1]
        rows = _load_csv(p)
        head = rows[:2] if rows else []
        lines.append(f"- {p.name}: " + (" | ".join([", ".join(r) for r in head]) if head else "empty"))

    if figures_dir.exists():
        figs = sorted(figures_dir.glob("*.pdf")) + sorted(figures_dir.glob("*.png"))
        if figs:
            lines.append("")
            lines.append("## figures")
            for p in figs:
                lines.append(f"- {p.relative_to(run_dir)}")

    pp_tables = sorted(tables_dir.glob("postprocess_*.csv"))
    if pp_tables:
        lines.append("")
        lines.append("## analysis tables (postprocess_*.csv)")
        for p in pp_tables:
            lines.append(f"- {p.relative_to(run_dir)}")

    # Embed key images for quick scan (if present).
    embed_imgs = [
        "figures/dashboard.png",
        "figures/offline_mine.png",
        "figures/offline_library.png",
        "figures/offline_memory.png",
        "figures/efficiency_frontier.png",
    ]
    embed_imgs.extend([p.relative_to(run_dir).as_posix() for p in sorted(figures_dir.glob("eval_analysis_T*.png"))])
    imgs_exist = [p for p in embed_imgs if (run_dir / p).exists()]
    if imgs_exist:
        lines.append("")
        lines.append("## quick visuals")
        for rel in imgs_exist:
            lines.append(f"### {rel}")
            lines.append("")
            lines.append(f"![{rel}]({rel})")
            lines.append("")

    if cases_dir.exists():
        md_cases = sorted(cases_dir.glob("*.md"))
        if md_cases:
            lines.append("")
            lines.append("## case studies")
            for p in md_cases:
                lines.append(f"- {p.relative_to(run_dir)}")

    if logs_dir.exists():
        logs = sorted(logs_dir.glob("*"))
        if logs:
            lines.append("")
            lines.append("## logs")
            for p in logs:
                lines.append(f"- {p.relative_to(run_dir)}")
            timings = parse_run_log_timings(run_dir)
            if timings:
                lines.append("")
                lines.append("## stage timings (rough)")
                for k, v in timings.items():
                    lines.append(f"- {k}: {v.get('seconds', 0.0):.1f}s")

    out_path = run_dir / "report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Path to outputs/<run_name>/<run_id>.")
    parser.add_argument("--outputs-root", type=str, default="outputs", help="Root outputs directory.")
    parser.add_argument("--run-name", type=str, default=None, help="run_name (if run_dir not given).")
    parser.add_argument("--run-id", type=str, default=None, help="run_id or 'latest' (if run_dir not given).")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        if not args.run_name or not args.run_id:
            raise ValueError("Either --run-dir or both --run-name and --run-id are required.")
        run_dir = _resolve_run_dir(args.run_name, args.run_id, Path(args.outputs_root))

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    cfg = _read_config(run_dir)
    budgets: List[int] = []
    try:
        budgets = [int(x) for x in (cfg.get("eval", {}).get("budgets", []) or [])]
    except Exception:
        budgets = []
    if not budgets:
        budgets = _infer_budgets_from_eval_dir(run_dir / "eval")

    dash = make_dashboard(run_dir)
    mine_fig, _mine_summary = analyze_offline_mine(run_dir)
    lib_fig, _lib_summary = analyze_library(run_dir)
    mem_fig, _mem_summary = analyze_memory(run_dir)
    eval_figs, _eval_summary = analyze_eval(run_dir, budgets=budgets)

    report = build_report(run_dir)
    print(f"Wrote report: {report}")
    if dash:
        print(f"Wrote dashboard: {dash}")
    else:
        print("Skip dashboard (matplotlib unavailable).")
    if mine_fig:
        print(f"Wrote offline mine fig: {mine_fig}")
    if lib_fig:
        print(f"Wrote library fig: {lib_fig}")
    if mem_fig:
        print(f"Wrote memory fig: {mem_fig}")
    for p in eval_figs:
        print(f"Wrote eval fig: {p}")


if __name__ == "__main__":
    main()
