from __future__ import annotations

from pathlib import Path

from esm.analysis.cases import _index_by_id
from esm.config import ESMConfig
from esm.data.loaders import load_task_dataset
from esm.online.esm import run_esm_dataset
from esm.online.greedy import run_greedy_dataset
from esm.utils.io import read_jsonl, write_text


def _sanitize_tag(s: str) -> str:
    import re

    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "case"


def _format_esm_steps(steps: list[dict]) -> str:
    if not steps:
        return "_(no steps logged)_"
    lines = []
    for s in steps:
        m = s.get("m")
        chosen = s.get("chosen")
        per_max = s.get("per_max")
        spent = s.get("spent")
        spent_probe = s.get("spent_probe", 0)
        spent_extra = s.get("spent_extra", 0)
        extra_max = s.get("extra_max", 0)
        cands = s.get("candidates") or []
        top = cands[:5]
        top_str = ", ".join(
            f"{c.get('name')}:{float(c.get('score', 0.0)):.3f}" for c in top if isinstance(c, dict)
        )
        lines.append(
            f"- m={m} chosen={chosen} per_max={per_max} spent={spent} "
            f"(probe={spent_probe}, extra={spent_extra}, extra_max={extra_max})"
            + (f" | top={top_str}" if top_str else "")
        )
    return "\n".join(lines)


def run_case(
    cfg: ESMConfig,
    *,
    example_ids: list[str],
    max_new_tokens: int | None = None,
    methods: list[str] | None = None,
    tag: str = "case",
) -> str:
    """
    Run greedy/ESM on a small set of examples and write a debug markdown under outputs/.../cases/.
    Intended for fast, case-level iteration.
    """
    methods = [m.strip().lower() for m in (methods or ["greedy", "esm"]) if str(m).strip() != ""]
    methods_set = set(methods)
    if not methods_set.issubset({"greedy", "esm"}):
        raise ValueError("methods must be a subset of {greedy, esm}.")

    ids = [str(x).strip() for x in example_ids if str(x).strip() != ""]
    if not ids:
        raise ValueError("example_ids is empty.")
    ids_set = set(ids)

    T_max = int(max_new_tokens) if max_new_tokens is not None else int(cfg.decode.max_new_tokens)

    # Load full split then filter (fast enough for the datasets we use).
    examples_all = load_task_dataset(
        task=cfg.task.dataset,
        split=cfg.task.eval_split,
        max_examples=None,
        seed=cfg.seed,
        data_root=cfg.task.data_root,
    )
    examples = [ex for ex in examples_all if str(ex.id) in ids_set]
    if not examples:
        raise ValueError(
            f"未找到指定 example_id(s)={sorted(ids_set)} 于 dataset={cfg.task.dataset}/{cfg.task.eval_split}。"
        )

    run_root = Path(cfg.outputs.run_dir)
    tag = _sanitize_tag(tag)
    ids_tag = _sanitize_tag("_".join(ids[:8]) + ("_etc" if len(ids) > 8 else ""))

    out_tags: dict[str, str] = {}
    if "greedy" in methods_set:
        out_tags["greedy"] = f"{tag}_greedy_{ids_tag}_T{T_max}"
        run_greedy_dataset(cfg, max_new_tokens=T_max, out_tag=out_tags["greedy"], examples=examples)
    if "esm" in methods_set:
        out_tags["esm"] = f"{tag}_esm_{ids_tag}_T{T_max}"
        run_esm_dataset(cfg, max_new_tokens=T_max, out_tag=out_tags["esm"], examples=examples)

    greedy_rows = {}
    esm_rows = {}
    if "greedy" in out_tags:
        greedy_rows = _index_by_id(read_jsonl(run_root / "eval" / out_tags["greedy"] / "per_example.jsonl"))
    if "esm" in out_tags:
        esm_rows = _index_by_id(read_jsonl(run_root / "eval" / out_tags["esm"] / "per_example.jsonl"))

    # Render markdown
    lines = []
    lines.append(f"# Case debug ({cfg.task.dataset}/{cfg.task.eval_split}, T_max={T_max})")
    lines.append("")
    lines.append(f"- run: `{cfg.outputs.run_dir}`")
    lines.append(f"- ids: `{', '.join(ids)}`")
    lines.append(f"- tags: `{out_tags}`")
    lines.append("")

    q_by_id = {str(e.id): e.question for e in examples}
    for ex_id in ids:
        q = q_by_id.get(ex_id, "")
        g = greedy_rows.get(ex_id, {})
        e = esm_rows.get(ex_id, {})

        lines.append(f"## id={ex_id}")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(q)
        lines.append("")
        if e.get("gold") is not None:
            lines.append(f"**Gold**: {e.get('gold')}")
            lines.append("")
        elif g.get("gold") is not None:
            lines.append(f"**Gold**: {g.get('gold')}")
            lines.append("")

        if g:
            lines.append(f"**Greedy**: pred={g.get('pred')} correct={g.get('correct')} tokens={g.get('tokens_used')}")
            lines.append("")
            lines.append("```")
            lines.append(str(g.get("text", ""))[:4000])
            lines.append("```")
            lines.append("")

        if e:
            lines.append(
                f"**ESM**: pred={e.get('pred')} correct={e.get('correct')} "
                f"tokens={e.get('tokens_used')} budget_used={e.get('budget_used')}"
            )
            lines.append("")
            steps = e.get("steps") or []
            if isinstance(steps, list):
                lines.append("**ESM step log (summary)**")
                lines.append("")
                lines.append(_format_esm_steps(steps))
                lines.append("")
            lines.append("```")
            lines.append(str(e.get("text", ""))[:4000])
            lines.append("```")
            lines.append("")

    out_dir = run_root / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_{ids_tag}_T{T_max}.md"
    write_text(out_path, "\n".join(lines) + "\n")
    return str(out_path)
