"""Aggregate the 9 canonical metric CSVs into the reproduced Table 5.

Inputs (written by scripts/collect_metrics.py):
    results/metrics/train_{warprec,elliot,recbole}_eval_{warprec,elliot,recbole}.csv

Outputs:
    results/table.md   — Markdown version of Table 5
    results/table.tex  — LaTeX version mirroring paper_table_5.tex
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
OUT_DIR = ROOT / "results"

FRAMEWORKS = ("Elliot", "RecBole", "WarpRec")
SLUGS = {"Elliot": "elliot", "RecBole": "recbole", "WarpRec": "warprec"}
MODELS = ("ItemKNN", "LightGCN")
COLS = ("nDCG", "Precision", "Recall", "MRR", "MAP", "Gini")
COL_HEADERS = ("nDCG", "Precis.", "Recall", "MRR", "MAP", "Gini")


def _load_cell(train, eval_):
    path = METRICS_DIR / f"train_{SLUGS[train]}_eval_{SLUGS[eval_]}.csv"
    if not path.is_file():
        raise SystemExit(f"[aggregate] missing {path}")
    out = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            out[row["model"]] = row
    return out


def _fmt(val):
    if val is None or val == "":
        return "--"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "--"


def _render_markdown(cells):
    header = "| Model | Train | Eval | " + " | ".join(COL_HEADERS) + " |"
    sep = "|" + "|".join(["---"] * (3 + len(COL_HEADERS))) + "|"
    lines = ["# Table 5 — Cross-framework performance on MovieLens-1M (cutoff @ 10, epoch 1)",
             "", header, sep]
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


def _render_latex(cells):
    def _rowcells(model, train, eval_):
        row = cells[(model, train, eval_)]
        return " & ".join(_fmt(row.get(c, "")) for c in COLS)

    lines = []
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
    cells = {}
    for train in FRAMEWORKS:
        for eval_ in FRAMEWORKS:
            by_model = _load_cell(train, eval_)
            for model in MODELS:
                cells[(model, train, eval_)] = by_model.get(model, {})

    md_path = OUT_DIR / "table.md"
    tex_path = OUT_DIR / "table.tex"
    md_path.write_text(_render_markdown(cells))
    tex_path.write_text(_render_latex(cells))
    print(f"[aggregate] wrote {md_path}")
    print(f"[aggregate] wrote {tex_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
