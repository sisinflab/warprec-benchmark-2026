"""Collect per-framework evaluation metrics into the canonical `metrics/` layout.

For each (train_framework, eval_framework) combination the three frameworks
emit metrics at different paths and with different column conventions. This
script normalises the output to:

    metrics/train_<train>_eval_<eval>.csv

with columns `model,nDCG,Precision,Recall,MRR,MAP,Gini,ShannonEntropy` and two
rows (ItemKNN, LightGCN).

Usage:
    python scripts/collect_metrics.py <train_fw> <eval_fw>
    # <train_fw>, <eval_fw> ∈ {warprec, elliot, recbole}

Stdlib-only. Safe to re-run (overwrites the canonical CSV).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "metrics"
CANONICAL_COLS = ("nDCG", "Precision", "Recall", "MRR", "MAP", "Gini", "ShannonEntropy")

# Raw metric name -> canonical.
# Comparison is done after lowercasing and stripping "@k" and non-alpha chars.
NAME_MAP = {
    "ndcg": "nDCG",
    "precision": "Precision",
    "precis": "Precision",
    "recall": "Recall",
    "mrr": "MRR",
    "map": "MAP",
    "gini": "Gini",
    "giniindex": "Gini",
    "sentropy": "ShannonEntropy",
    "shannonentropy": "ShannonEntropy",
}


def _canonical_metric_name(raw: str) -> str | None:
    # Strip "@k" and non-alphanumeric characters, lowercase.
    key = re.sub(r"@\d+", "", raw).strip()
    key = re.sub(r"[^A-Za-z]", "", key).lower()
    return NAME_MAP.get(key)


def _model_bucket(name: str) -> str | None:
    low = name.lower()
    if "itemknn" in low:
        return "ItemKNN"
    if "lightgcn" in low:
        return "LightGCN"
    return None


def _newest(paths: list[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)


def _read_delim(path: Path, sep: str) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        return list(reader)


def _write_canonical(out_path: Path, rows_by_model: dict[str, dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("model",) + CANONICAL_COLS)
        for model in ("ItemKNN", "LightGCN"):
            row = rows_by_model.get(model)
            if row is None:
                writer.writerow([model] + [""] * len(CANONICAL_COLS))
                print(f"[collect_metrics] warning: {model} missing for {out_path.name}",
                      file=sys.stderr)
            else:
                writer.writerow([model] + [row.get(c, "") for c in CANONICAL_COLS])
    print(f"[collect_metrics] wrote {out_path}")


def _parse_warprec_overall(path: Path) -> dict[str, dict[str, str]]:
    # Columns: Model, Top@k, <metric>, <metric>, ...
    rows = _read_delim(path, "\t")
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        model = _model_bucket(r.get("Model", ""))
        if model is None:
            continue
        canonical: dict[str, str] = {}
        for k, v in r.items():
            c = _canonical_metric_name(k)
            if c:
                canonical[c] = v
        out[model] = canonical
    return out


def _parse_elliot_performance(path: Path) -> dict[str, dict[str, str]]:
    # Columns: model, nDCG, Precision, Recall, MRR, MAP, Gini, SEntropy
    rows = _read_delim(path, "\t")
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        model = _model_bucket(r.get("model", ""))
        if model is None:
            continue
        canonical: dict[str, str] = {}
        for k, v in r.items():
            if k == "model":
                continue
            c = _canonical_metric_name(k)
            if c:
                canonical[c] = v
        out[model] = canonical
    return out


def _parse_recbole_metrics(path: Path) -> dict[str, dict[str, str]]:
    # Columns: model, ndcg@10, precision@10, ...
    rows = _read_delim(path, ",")
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        model = _model_bucket(r.get("model", ""))
        if model is None:
            continue
        canonical: dict[str, str] = {}
        for k, v in r.items():
            if k == "model":
                continue
            c = _canonical_metric_name(k)
            if c:
                canonical[c] = v
        out[model] = canonical
    return out


def _collect_warprec_eval(train_fw: str) -> dict[str, dict[str, str]]:
    """eval_fw == warprec; train_fw is the framework whose recs we evaluated."""
    dataset_prefix = {
        "warprec": "WarpRec_Reproducibility",
        "elliot": "WarpRec_Elliot_{model}_Reproducibility",
        "recbole": "WarpRec_RecBole_{model}_Reproducibility",
    }[train_fw]

    out: dict[str, dict[str, str]] = {}
    if train_fw == "warprec":
        d = ROOT / "experiment" / dataset_prefix / "evaluation"
        files = sorted(d.glob("Overall_Results_*.tsv"))
        if not files:
            raise SystemExit(f"[collect_metrics] no Overall_Results_*.tsv under {d}")
        out.update(_parse_warprec_overall(_newest(files)))
    else:
        for model in ("ItemKNN", "LightGCN"):
            d = ROOT / "experiment" / dataset_prefix.format(model=model) / "evaluation"
            files = sorted(d.glob("Overall_Results_*.tsv"))
            if not files:
                raise SystemExit(f"[collect_metrics] no Overall_Results_*.tsv under {d}")
            parsed = _parse_warprec_overall(_newest(files))
            # Each cross-eval run has exactly one model's metrics; pick them regardless of label.
            if parsed:
                key = next(iter(parsed))
                out[model] = parsed[key]
    return out


def _collect_elliot_eval(train_fw: str) -> dict[str, dict[str, str]]:
    """eval_fw == elliot."""
    dataset_names = {
        "elliot": ["ElliotRepr"],
        "recbole": ["ElliotRecBoleItemKNNRepr", "ElliotRecBoleLightGCNRepr"],
        "warprec": ["ElliotWarpRecItemKNNRepr", "ElliotWarpRecLightGCNRepr"],
    }[train_fw]

    out: dict[str, dict[str, str]] = {}
    for dn in dataset_names:
        d = ROOT / "results" / dn / "performance"
        files = sorted(d.glob("rec_cutoff_10_*.tsv"))
        if not files:
            raise SystemExit(f"[collect_metrics] no rec_cutoff_10_*.tsv under {d}")
        parsed = _parse_elliot_performance(_newest(files))
        for k, v in parsed.items():
            out[k] = v
        if train_fw != "elliot" and parsed:
            # Cross-eval: the dataset is model-specific; if auto-detection failed
            # (Proxy's row name), infer from the dataset_name.
            inferred = "ItemKNN" if "ItemKNN" in dn else "LightGCN"
            if inferred not in out:
                first = next(iter(parsed))
                out[inferred] = parsed[first]
    return out


def _collect_recbole_eval(train_fw: str) -> dict[str, dict[str, str]]:
    """eval_fw == recbole."""
    recbole_results = ROOT / "frameworks" / "recbole" / "results"
    if train_fw == "recbole":
        path = recbole_results / "metrics.csv"
    else:
        path = recbole_results / "external" / f"{train_fw}_metrics.csv"
    if not path.is_file():
        raise SystemExit(f"[collect_metrics] missing {path}")
    return _parse_recbole_metrics(path)


COLLECTORS = {
    "warprec": _collect_warprec_eval,
    "elliot": _collect_elliot_eval,
    "recbole": _collect_recbole_eval,
}


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/collect_metrics.py <train_fw> <eval_fw>", file=sys.stderr)
        return 2
    train_fw, eval_fw = sys.argv[1], sys.argv[2]
    if train_fw not in COLLECTORS or eval_fw not in COLLECTORS:
        print(f"frameworks must be one of {list(COLLECTORS)}", file=sys.stderr)
        return 2

    rows = COLLECTORS[eval_fw](train_fw)
    out = METRICS_DIR / f"train_{train_fw}_eval_{eval_fw}.csv"
    _write_canonical(out, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
