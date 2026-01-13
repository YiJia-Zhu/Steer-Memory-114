from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esm.offline.stage2 import _load_vector_from_pt  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _cos_stats(V: np.ndarray) -> dict[str, float]:
    if V.size == 0:
        return {}
    V = V.astype(np.float64)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    V = V / norms
    G = V @ V.T
    mask = ~np.eye(G.shape[0], dtype=bool)
    vals = G[mask]
    out: dict[str, float] = {
        "n": float(V.shape[0]),
        "d": float(V.shape[1]),
        "cos_mean": float(vals.mean()) if vals.size else 0.0,
        "cos_std": float(vals.std()) if vals.size else 0.0,
        "cos_min": float(vals.min()) if vals.size else 0.0,
        "cos_p50": float(np.median(vals)) if vals.size else 0.0,
        "cos_p90": float(np.percentile(vals, 90)) if vals.size else 0.0,
        "cos_p99": float(np.percentile(vals, 99)) if vals.size else 0.0,
        "cos_max": float(vals.max()) if vals.size else 0.0,
    }
    return out


def _print_kv(title: str, d: dict[str, float]) -> None:
    if not d:
        print(f"{title}: (empty)")
        return
    keys = list(d.keys())
    line = ", ".join(f"{k}={d[k]:.6g}" for k in keys)
    print(f"{title}: {line}")


def _resolve_run_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    # Treat as run_name under outputs/
    run_root = Path("outputs") / arg
    latest = run_root / "LATEST"
    if latest.exists():
        rid = latest.read_text(encoding="utf-8").strip()
        cand = run_root / rid
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Cannot resolve run dir from: {arg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        type=str,
        default="gsm8k_recommended_small",
        help="Run directory path, or outputs/<run_name> (will use LATEST).",
    )
    ap.add_argument("--max-vectors", type=int, default=512, help="Max vectors to load for cosine stats.")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(str(args.run))
    print(f"run_dir={run_dir}")

    # Stage I candidates
    cand_path = run_dir / "mine" / "candidates.jsonl"
    candidates = _read_jsonl(cand_path)
    if candidates:
        paths = [c["vector_path"] for c in candidates][: int(args.max_vectors)]
        V = np.stack([_load_vector_from_pt(p) for p in paths], axis=0)
        _print_kv("candidates.cos", _cos_stats(V))
        by_m = Counter(int(c.get("control_point_m", -1)) for c in candidates)
        by_layer = Counter(int(c.get("layer", -1)) for c in candidates)
        print(f"candidates.n={len(candidates)} by_m={dict(sorted(by_m.items()))}")
        print(f"candidates.by_layer(top10)={by_layer.most_common(10)}")
    else:
        print("candidates: missing/empty")

    # Stage II library
    lib = _read_jsonl(run_dir / "library" / "library.jsonl")
    if lib:
        paths = [r["vector_path"] for r in lib][: int(args.max_vectors)]
        V = np.stack([_load_vector_from_pt(p) for p in paths], axis=0)
        _print_kv("library.cos", _cos_stats(V))
        by_m = Counter(int(r.get("control_point_m", -1)) for r in lib)
        print(f"library.n={len(lib)} by_m={dict(sorted(by_m.items()))}")
    else:
        print("library: missing/empty")

    # Stage III memory advantages
    entries = _read_jsonl(run_dir / "memory" / "entries.jsonl")
    if entries:
        adv = np.array([float(e.get("advantage", 0.0)) for e in entries], dtype=np.float64)
        ms = np.array([int(e.get("control_point_m", -1)) for e in entries], dtype=int)
        print(
            "memory.adv: "
            f"n={len(entries)} min={adv.min():.6g} mean={adv.mean():.6g} "
            f"p50={np.median(adv):.6g} p90={np.percentile(adv,90):.6g} max={adv.max():.6g} "
            f"nonzero_frac={(np.mean(np.abs(adv) > 1e-9)):.3f}"
        )
        for m in sorted(set(ms.tolist())):
            if m <= 0:
                continue
            a = adv[ms == m]
            print(f"memory.adv[m={m}]: n={len(a)} mean={a.mean():.6g} p90={np.percentile(a,90):.6g} max={a.max():.6g}")
        tools = Counter(str(e.get("tool_name", "")) for e in entries)
        print(f"memory.tools: n_tools={len(tools)} top5={tools.most_common(5)}")
    else:
        print("memory: missing/empty")

    # Online ESM step stats
    esm_eval = run_dir / "eval" / "ESM_T2048" / "per_example.jsonl"
    rows = _read_jsonl(esm_eval)
    if rows:
        chosen = Counter()
        max_ahat = []
        top_sim = []
        skip_probe = 0
        null_steps = 0
        tool_steps = 0
        for r in rows:
            for s in r.get("steps", []):
                name = str(s.get("chosen"))
                if name == "null":
                    null_steps += 1
                else:
                    tool_steps += 1
                    chosen[name] += 1
                if str(s.get("note", "")).startswith("skip_probe:"):
                    skip_probe += 1
                if s.get("mem_max_ahat") is not None:
                    max_ahat.append(float(s["mem_max_ahat"]))
                if s.get("mem_top_sim") is not None:
                    top_sim.append(float(s["mem_top_sim"]))
        if max_ahat:
            a = np.array(max_ahat, dtype=np.float64)
            print(
                "esm.mem_max_ahat: "
                f"n={len(a)} min={a.min():.6g} mean={a.mean():.6g} p50={np.median(a):.6g} "
                f"p90={np.percentile(a,90):.6g} max={a.max():.6g}"
            )
        if top_sim:
            a = np.array(top_sim, dtype=np.float64)
            print(
                "esm.mem_top_sim: "
                f"n={len(a)} min={a.min():.6g} mean={a.mean():.6g} p50={np.median(a):.6g} "
                f"p90={np.percentile(a,90):.6g} max={a.max():.6g}"
            )
        total = null_steps + tool_steps
        print(f"esm.steps: total={total} null={null_steps} tool={tool_steps} skip_probe={skip_probe}")
        print(f"esm.chosen_tools(top10)={chosen.most_common(10)}")
    else:
        print(f"esm eval logs missing/empty: {esm_eval}")


if __name__ == "__main__":
    main()
