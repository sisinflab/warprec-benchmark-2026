"""Copy WarpRec's generated temporal-holdout split into the canonical location.

WarpRec writes its split and recs somewhere under `experiment/` (the exact path
depends on WarpRec's internal dataset-naming conventions). This script locates
them by globbing and stages:

    experiment/**/train.tsv  -> data/split/train.tsv
    experiment/**/test.tsv   -> data/split/test.tsv

The canonical TSV files are consumed by the Elliot and RecBole training stages.
Stdlib-only; safe to re-run.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiment"
SPLIT_DIR = ROOT / "data" / "split"


def _locate_split_file(name: str) -> Path:
    # WarpRec may write split files as train.tsv / test.tsv or train.csv /
    # test.csv depending on its writer settings; search for both extensions.
    candidates = sorted(
        list(EXP_DIR.rglob(f"{name}.tsv")) + list(EXP_DIR.rglob(f"{name}.csv"))
    )
    if not candidates:
        raise SystemExit(
            f"[export_split] no {name}.tsv/.csv found under {EXP_DIR}. "
            f"Did `warprec -c frameworks/warprec/train.yml -p train` run?"
        )
    if len(candidates) > 1:
        # Prefer the one sitting under the top-level WarpRec_Reproducibility run.
        for c in candidates:
            if "WarpRec_Reproducibility" in str(c) and "recommendation" not in str(c):
                return c
        print(
            f"[export_split] warning: multiple {name} files found; picking {candidates[0]}",
            file=sys.stderr,
        )
    return candidates[0]


def main() -> int:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test"):
        src = _locate_split_file(name)
        dst = SPLIT_DIR / f"{name}.tsv"
        with src.open("r") as fin, dst.open("w") as fout:
            for line in fin:
                # Normalize to tab-separated regardless of WarpRec's output separator.
                fout.write("\t".join(line.rstrip("\n").split(",")) + "\n"
                           if "," in line and "\t" not in line
                           else line)
        print(f"[export_split] {src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
