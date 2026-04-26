"""Aggregate the 9 canonical metric CSVs into the reproduced Table 5.

Expected inputs (written by scripts/collect_metrics.py):

    metrics/train_{elliot,recbole,warprec}_eval_{elliot,recbole,warprec}.csv

Each canonical CSV has columns:
    model,nDCG,Precision,Recall,MRR,MAP,Gini,ShannonEntropy

Outputs (both overwritten on each run):
    table.md   — GitHub-flavoured Markdown version of Table 5
    table.tex  — ACM sigconf LaTeX version (same visual layout as the
                 pristine table.tex shipped with the paper; we keep that
                 pristine copy at table.tex.orig for diffing)

Stdlib-only. Floats are rendered with 4 decimal places to match the paper.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "metrics"

FRAMEWORKS = ("Elliot", "RecBole", "WarpRec")  # display order in the table
SLUGS = {"Elliot": "elliot", "RecBole": "recbole", "WarpRec": "warprec"}
MODELS = ("ItemKNN", "LightGCN")
COLS = ("nDCG", "Precision", "Recall", "MRR", "MAP", "Gini")  # Table 5 drops ShannonEntropy
COL_HEADERS = ("nDCG", "Precis.", "Recall", "MRR", "MAP", "Gini")


def _load_cell(train: str, eval_: str) -> dict[str, dict[str, str]]:
    path = METRICS_DIR / f"train_{SLUGS[train]}_eval_{SLUGS[eval_]}.csv"
    if not path.is_file():
        raise SystemExit(f"[aggregate] missing {path}")
    out: dict[str, dict[str, str]] = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            out[row["model"]] = row
    return out


def _fmt(val: str) -> str:
    if val is None or val == "":
        return "--"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "--"


def _render_markdown(cells: dict[tuple[str, str, str], dict[str, str]]) -> str:
    header = "| Model | Train | Eval | " + " | ".join(COL_HEADERS) + " |"
    sep = "|" + "|".join(["---"] * (3 + len(COL_HEADERS))) + "|"
    lines = ["# Table 5 — Cross-framework performance on MovieLens-1M (cutoff @ 10)", "", header, sep]
    for model in MODELS:
        for train in FRAMEWORKS:
            for eval_ in FRAMEWORKS:
                row = cells[(model, train, eval_)]
                lines.append(
                    f"| {model} | {train} | {eval_} | "
                    + " | ".join(_fmt(row.get(c, "")) for c in COLS)
                    + " |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_latex(cells: dict[tuple[str, str, str], dict[str, str]]) -> str:
    # Mirrors the structure of the pristine table.tex.orig: two \multirow{9}
    # blocks (one per model) with three nested \multirow{3} Train groups each.
    def _rowcells(model: str, train: str, eval_: str) -> str:
        row = cells[(model, train, eval_)]
        return " & ".join(_fmt(row.get(c, "")) for c in COLS)

    lines: list[str] = []
    lines.append(r"\begin{table}")
    lines.append(r"\caption{Cross-framework performance evaluation with a cutoff$@10$ for ItemKNN and LightGCN on MovieLens-1M on epoch $1$.}")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"    \renewcommand{\arraystretch}{0.85}")
    lines.append(r"    \setlength{\tabcolsep}{0.5em} ")
    lines.append("")
    lines.append(r"    \begin{tabular}{lllrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    & \textbf{Train} & \textbf{Eval} & \textbf{nDCG} & \textbf{Precis.} & \textbf{Recall} & \textbf{MRR} & \textbf{MAP} & \textbf{Gini} \\")
    lines.append(r"    \midrule")
    lines.append("")

    for m_idx, model in enumerate(MODELS):
        if m_idx > 0:
            lines.append(r"    \midrule")
            lines.append("")
        lines.append(r"    \multirow{9}{*}{\rotatebox{90}{\textbf{" + model + r"}}} ")
        for t_idx, train in enumerate(FRAMEWORKS):
            if t_idx > 0:
                lines.append(r"    \cmidrule{2-9}")
            for e_idx, eval_ in enumerate(FRAMEWORKS):
                prefix = r"    & \multirow{3}{*}{" + train + "} & " if e_idx == 0 else r"    & & "
                lines.append(f"{prefix}{eval_} & {_rowcells(model, train, eval_)} \\\\")
        lines.append("")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    ")
    lines.append(r"\label{tab:metrics_evaluation}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    cells: dict[tuple[str, str, str], dict[str, str]] = {}
    for train in FRAMEWORKS:
        for eval_ in FRAMEWORKS:
            by_model = _load_cell(train, eval_)
            for model in MODELS:
                cells[(model, train, eval_)] = by_model.get(model, {})

    md_path = ROOT / "table.md"
    tex_path = ROOT / "table.tex"
    md_path.write_text(_render_markdown(cells))
    tex_path.write_text(_render_latex(cells))
    print(f"[aggregate] wrote {md_path}")
    print(f"[aggregate] wrote {tex_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
