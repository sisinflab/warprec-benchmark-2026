"""Collect per-framework recommendation TSVs into the canonical `recs/` layout.

Each training framework produces top-K recommendations in a framework-specific
location and with a framework-specific filename convention. This script
normalises everything to:

    recs/<framework>/<ModelName>.tsv      # 3 cols (user, item, score); no header; tab-separated

Usage:
    python scripts/collect_recs.py warprec
    python scripts/collect_recs.py elliot
    python scripts/collect_recs.py recbole

Stdlib-only. Re-runnable (overwrites targets).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "recs"
MODELS = ("ItemKNN", "LightGCN")


def _pick_most_recent(candidates: list[Path]) -> Path:
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _normalize_to_tsv(src: Path, dst: Path) -> None:
    """Read src (any of tsv/csv, with or without header) and write 3-column TSV
    with no header. Rows that cannot be parsed into (user, item, score) are
    skipped with a warning."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with src.open("r") as fin, dst.open("w") as fout:
        for idx, raw in enumerate(fin):
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if idx == 0 and not line.replace(".", "").replace("-", "").split(
                "\t" if "\t" in line else ","
            )[0].strip().isdigit():
                # Header row — skip.
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) < 3:
                skipped += 1
                continue
            u, i, s = parts[0].strip(), parts[1].strip(), parts[2].strip()
            fout.write(f"{u}\t{i}\t{s}\n")
            kept += 1
    print(f"[collect_recs] {src} -> {dst} (kept {kept}, skipped {skipped})")


def _collect_warprec() -> None:
    exp = ROOT / "experiment" / "WarpRec_Reproducibility" / "recs"
    if not exp.is_dir():
        raise SystemExit(f"[collect_recs] missing {exp} — did `make train-warprec` run?")
    for model in MODELS:
        hits = sorted(exp.glob(f"{model}_*.tsv"))
        if not hits:
            raise SystemExit(f"[collect_recs] no {model}_*.tsv under {exp}")
        _normalize_to_tsv(_pick_most_recent(hits), OUT / "warprec" / f"{model}.tsv")


def _collect_elliot() -> None:
    # Elliot's default recs location: `<config-parent>/../results/<dataset>/recs/<model>_<hash>.tsv`
    # With our configs that resolves to `ROOT/results/ElliotRepr/recs/`.
    dir_ = ROOT / "results" / "ElliotRepr" / "recs"
    if not dir_.is_dir():
        raise SystemExit(f"[collect_recs] missing {dir_} — did `make train-elliot` run?")
    for model in MODELS:
        hits = sorted(dir_.glob(f"{model}*.tsv"))
        if not hits:
            raise SystemExit(f"[collect_recs] no {model}*.tsv under {dir_}")
        _normalize_to_tsv(_pick_most_recent(hits), OUT / "elliot" / f"{model}.tsv")


def _collect_recbole() -> None:
    # RecBole's run_experiment.py writes these explicit paths:
    dir_ = ROOT / "frameworks" / "recbole" / "results"
    for model in MODELS:
        src = dir_ / f"{model}_recs.tsv"
        if not src.is_file():
            raise SystemExit(f"[collect_recs] missing {src} — did `make train-recbole` run?")
        _normalize_to_tsv(src, OUT / "recbole" / f"{model}.tsv")


HANDLERS = {
    "warprec": _collect_warprec,
    "elliot": _collect_elliot,
    "recbole": _collect_recbole,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in HANDLERS:
        print("Usage: python scripts/collect_recs.py {warprec|elliot|recbole}", file=sys.stderr)
        return 2
    HANDLERS[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
