"""Normalize each (train_fw, eval_fw) cell into results/metrics/train_<a>_eval_<b>.csv.

Each canonical CSV has columns:
    model,nDCG,Precision,Recall,MRR,MAP,Gini,ShannonEntropy
with rows for ItemKNN and LightGCN. The aggregator reads these into the final
table layout.

Usage (from `tables_reproducibility/table_5/`):
    python scripts/collect_metrics.py <train_fw> <eval_fw>
    # frameworks ∈ {warprec, elliot, recbole}
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
CANONICAL_COLS = ("nDCG", "Precision", "Recall", "MRR", "MAP", "Gini", "ShannonEntropy")

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


def _canonical(raw: str):
    key = re.sub(r"@\d+", "", raw).strip()
    key = re.sub(r"[^A-Za-z]", "", key).lower()
    return NAME_MAP.get(key)


def _model_bucket(name: str):
    low = name.lower()
    if "itemknn" in low:
        return "ItemKNN"
    if "lightgcn" in low:
        return "LightGCN"
    return None


def _newest(paths):
    return max(paths, key=lambda p: p.stat().st_mtime)


def _read_delim(path, sep):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f, delimiter=sep))


def _write_canonical(out_path, rows_by_model):
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


def _parse_warprec_overall(path, fallback_model=None):
    """Parse WarpRec's `Overall_Results_*.tsv`.

    For native runs the `Model` column carries the model name (ItemKNN /
    LightGCN). For cross-eval runs it's `ProxyRecommender`; in that case
    we use `fallback_model` to slot the row into the right bucket.
    """
    rows = _read_delim(path, "\t")
    out = {}
    for r in rows:
        model = _model_bucket(r.get("Model", "")) or fallback_model
        if model is None:
            continue
        canonical = {}
        for k, v in r.items():
            c = _canonical(k)
            if c:
                canonical[c] = v
        out[model] = canonical
    return out


def _parse_elliot_performance(path):
    """Elliot reports Gini as a concentration index (low = diverse).
    The paper and WarpRec/RecBole report it as a diversity index, so we
    flip it here: paper_Gini = 1 - elliot_Gini.
    """
    rows = _read_delim(path, "\t")
    out = {}
    for r in rows:
        model = _model_bucket(r.get("model", ""))
        if model is None:
            continue
        canonical = {}
        for k, v in r.items():
            if k == "model":
                continue
            c = _canonical(k)
            if c:
                canonical[c] = v
        if "Gini" in canonical:
            try:
                canonical["Gini"] = f"{1.0 - float(canonical['Gini']):.16f}"
            except (TypeError, ValueError):
                pass
        out[model] = canonical
    return out


def _parse_recbole_metrics(path):
    rows = _read_delim(path, ",")
    out = {}
    for r in rows:
        model = _model_bucket(r.get("model", ""))
        if model is None:
            continue
        canonical = {}
        for k, v in r.items():
            if k == "model":
                continue
            c = _canonical(k)
            if c:
                canonical[c] = v
        out[model] = canonical
    return out


def _collect_warprec_eval(train_fw):
    """eval_fw == warprec; train_fw is the framework whose recs we evaluated."""
    dataset_prefix = {
        "warprec": "WarpRec_Reproducibility",
        "elliot": "WarpRec_Elliot_{model}_Reproducibility",
        "recbole": "WarpRec_RecBole_{model}_Reproducibility",
    }[train_fw]

    out = {}
    if train_fw == "warprec":
        d = ROOT / "results" / "experiment" / dataset_prefix / "evaluation"
        files = sorted(d.glob("Overall_Results_*.tsv"))
        if not files:
            raise SystemExit(f"[collect_metrics] no Overall_Results_*.tsv under {d}")
        out.update(_parse_warprec_overall(_newest(files)))
    else:
        for model in ("ItemKNN", "LightGCN"):
            d = ROOT / "results" / "experiment" / dataset_prefix.format(model=model) / "evaluation"
            files = sorted(d.glob("Overall_Results_*.tsv"))
            if not files:
                raise SystemExit(f"[collect_metrics] no Overall_Results_*.tsv under {d}")
            parsed = _parse_warprec_overall(_newest(files), fallback_model=model)
            if parsed:
                out[model] = parsed.get(model, parsed[next(iter(parsed))])
    return out


def _collect_elliot_eval(train_fw):
    dataset_names = {
        "elliot": ["ElliotRepr"],
        "recbole": ["ElliotRecBoleItemKNNRepr", "ElliotRecBoleLightGCNRepr"],
        "warprec": ["ElliotWarpRecItemKNNRepr", "ElliotWarpRecLightGCNRepr"],
    }[train_fw]

    out = {}
    for dn in dataset_names:
        d = ROOT / "results" / "elliot_runs" / dn / "performance"
        files = sorted(d.glob("rec_cutoff_10_*.tsv"))
        if not files:
            raise SystemExit(f"[collect_metrics] no rec_cutoff_10_*.tsv under {d}")
        parsed = _parse_elliot_performance(_newest(files))
        for k, v in parsed.items():
            out[k] = v
        if train_fw != "elliot" and parsed:
            inferred = "ItemKNN" if "ItemKNN" in dn else "LightGCN"
            if inferred not in out:
                first = next(iter(parsed))
                out[inferred] = parsed[first]
    return out


def _collect_recbole_eval(train_fw):
    recbole_results = ROOT / "results" / "recbole_runs"
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
        print("Usage: python scripts/collect_metrics.py <train_fw> <eval_fw>",
              file=sys.stderr)
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
