from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from esm.utils.io import read_jsonl

logger = logging.getLogger(__name__)


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


@dataclass(frozen=True)
class MethodComparison:
    T_max: int
    n: int
    acc_greedy: float
    acc_esm: float
    acc_diff: float
    win: int
    lose: int
    both_correct: int
    both_wrong: int
    win_rate: float
    lose_rate: float
    net_win: int
    tokens_greedy_mean: float
    tokens_esm_mean: float
    tokens_diff_mean: float
    tokens_diff_mean_on_win: float
    tokens_diff_mean_on_lose: float
    hard_n: int
    acc_esm_on_hard: float
    tokens_diff_mean_on_hard: float
    esm_budget_mean: float
    esm_overhead_mean: float


def write_method_comparison(
    *,
    run_dir: str | Path,
    T_max: int,
    greedy_tag: str,
    esm_tag: str,
    out_prefix: str = "",
) -> dict[str, Any]:
    """
    Compare greedy vs ESM at one budget point.

    Writes:
    - outputs/<run>/tables/{out_prefix}compare_T{T}.csv
    - outputs/<run>/tables/{out_prefix}compare_T{T}.json
    """
    run_dir = Path(run_dir)
    out_tables = run_dir / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    greedy_path = run_dir / "eval" / greedy_tag / "per_example.jsonl"
    esm_path = run_dir / "eval" / esm_tag / "per_example.jsonl"
    if not greedy_path.exists() or not esm_path.exists():
        logger.warning("Comparison skipped (missing per_example): %s %s", greedy_path, esm_path)
        return {}

    greedy_rows = read_jsonl(greedy_path)
    esm_rows = read_jsonl(esm_path)
    g_by_id = {str(r.get("example_id")): r for r in greedy_rows if r.get("example_id") is not None}
    e_by_id = {str(r.get("example_id")): r for r in esm_rows if r.get("example_id") is not None}
    ids = sorted(set(g_by_id.keys()) & set(e_by_id.keys()))
    if not ids:
        logger.warning("Comparison skipped (no overlapping example_id).")
        return {}

    wins = 0
    loses = 0
    both_correct = 0
    both_wrong = 0

    tok_g: list[int] = []
    tok_e: list[int] = []
    tok_diff: list[int] = []
    tok_diff_win: list[int] = []
    tok_diff_lose: list[int] = []
    tok_diff_hard: list[int] = []

    hard = 0
    hard_esm_correct = 0

    esm_budget: list[int] = []
    esm_overhead: list[int] = []

    for ex_id in ids:
        g = g_by_id[ex_id]
        e = e_by_id[ex_id]
        g_ok = bool(g.get("correct"))
        e_ok = bool(e.get("correct"))

        tg = int(g.get("tokens_used", 0) or 0)
        te = int(e.get("tokens_used", 0) or 0)
        tok_g.append(tg)
        tok_e.append(te)
        d = te - tg
        tok_diff.append(d)

        b = int(e.get("budget_used", 0) or 0)
        esm_budget.append(b)
        esm_overhead.append(max(0, b - te))

        if (not g_ok) and e_ok:
            wins += 1
            tok_diff_win.append(d)
        elif g_ok and (not e_ok):
            loses += 1
            tok_diff_lose.append(d)
        elif g_ok and e_ok:
            both_correct += 1
        else:
            both_wrong += 1

        if not g_ok:
            hard += 1
            if e_ok:
                hard_esm_correct += 1
            tok_diff_hard.append(d)

    acc_g = _safe_mean([1.0 if bool(g_by_id[i].get("correct")) else 0.0 for i in ids])
    acc_e = _safe_mean([1.0 if bool(e_by_id[i].get("correct")) else 0.0 for i in ids])
    win_rate = float(wins) / float(len(ids))
    lose_rate = float(loses) / float(len(ids))
    net_win = int(wins - loses)

    summary = MethodComparison(
        T_max=int(T_max),
        n=int(len(ids)),
        acc_greedy=float(acc_g),
        acc_esm=float(acc_e),
        acc_diff=float(acc_e - acc_g),
        win=int(wins),
        lose=int(loses),
        both_correct=int(both_correct),
        both_wrong=int(both_wrong),
        win_rate=float(win_rate),
        lose_rate=float(lose_rate),
        net_win=int(net_win),
        tokens_greedy_mean=_safe_mean([float(x) for x in tok_g]),
        tokens_esm_mean=_safe_mean([float(x) for x in tok_e]),
        tokens_diff_mean=_safe_mean([float(x) for x in tok_diff]),
        tokens_diff_mean_on_win=_safe_mean([float(x) for x in tok_diff_win]),
        tokens_diff_mean_on_lose=_safe_mean([float(x) for x in tok_diff_lose]),
        hard_n=int(hard),
        acc_esm_on_hard=float(hard_esm_correct / max(1, hard)),
        tokens_diff_mean_on_hard=_safe_mean([float(x) for x in tok_diff_hard]),
        esm_budget_mean=_safe_mean([float(x) for x in esm_budget]),
        esm_overhead_mean=_safe_mean([float(x) for x in esm_overhead]),
    )

    prefix = str(out_prefix or "")
    out_json = out_tables / f"{prefix}compare_T{int(T_max)}.json"
    out_csv = out_tables / f"{prefix}compare_T{int(T_max)}.csv"

    payload = asdict(summary)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(payload.keys()))
        w.writerow(list(payload.values()))

    logger.info("Wrote method comparison: %s %s", out_csv, out_json)
    return payload

