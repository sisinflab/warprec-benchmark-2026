"""Normalize each framework's recs to results/recs/<framework>/<model>.tsv.

Canonical format: 3 columns (user_id, item_id, score), no header, tab-separated.
This is what every cross-evaluator (Elliot ProxyRecommender, WarpRec
ProxyRecommender, RecBole evaluate_external_recs) expects to ingest.

Usage (from `tables_reproducibility/table_5/`):
    python scripts/collect_recs.py {warprec|elliot|recbole}
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "recs"
MODELS = ("ItemKNN", "LightGCN")


def _pick_most_recent(candidates):
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _normalize_to_tsv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with src.open("r") as fin, dst.open("w") as fout:
        for idx, raw in enumerate(fin):
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            sep = "\t" if "\t" in line else ","
            if idx == 0:
                first = line.split(sep)[0].strip()
                if not first.replace(".", "").replace("-", "").isdigit():
                    continue  # skip header row
            parts = line.split(sep)
            if len(parts) < 3:
                skipped += 1
                continue
            u, i, s = parts[0].strip(), parts[1].strip(), parts[2].strip()
            fout.write(f"{u}\t{i}\t{s}\n")
            kept += 1
    print(f"[collect_recs] {src} -> {dst} (kept {kept}, skipped {skipped})")


def _collect_warprec() -> None:
    exp = ROOT / "results" / "experiment" / "WarpRec_Reproducibility" / "recs"
    if not exp.is_dir():
        raise SystemExit(f"[collect_recs] missing {exp} — did WarpRec training run?")
    for model in MODELS:
        hits = sorted(exp.glob(f"{model}_*.tsv")) + sorted(exp.glob(f"{model}_*.csv"))
        if not hits:
            raise SystemExit(f"[collect_recs] no {model}_* under {exp}")
        _normalize_to_tsv(_pick_most_recent(hits), OUT / "warprec" / f"{model}.tsv")


def _collect_elliot() -> None:
    elliot_recs_dir = ROOT / "results" / "elliot_runs" / "ElliotRepr" / "recs"
    if not elliot_recs_dir.is_dir():
        raise SystemExit(
            f"[collect_recs] missing {elliot_recs_dir} — did Elliot training run?"
        )
    for model in MODELS:
        hits = sorted(elliot_recs_dir.glob(f"{model}*.tsv"))
        if not hits:
            raise SystemExit(f"[collect_recs] no {model}* under {elliot_recs_dir}")
        _normalize_to_tsv(_pick_most_recent(hits), OUT / "elliot" / f"{model}.tsv")


def _collect_recbole() -> None:
    recbole_dir = ROOT / "results" / "recbole_runs"
    for model in MODELS:
        src = recbole_dir / f"{model}_recs.tsv"
        if not src.is_file():
            raise SystemExit(
                f"[collect_recs] missing {src} — did RecBole training run?"
            )
        _normalize_to_tsv(src, OUT / "recbole" / f"{model}.tsv")


HANDLERS = {
    "warprec": _collect_warprec,
    "elliot": _collect_elliot,
    "recbole": _collect_recbole,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in HANDLERS:
        print("Usage: python scripts/collect_recs.py {warprec|elliot|recbole}",
              file=sys.stderr)
        return 2
    HANDLERS[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
