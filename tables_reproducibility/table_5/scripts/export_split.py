"""Stage WarpRec's generated temporal-holdout split into data/split/.

WarpRec writes the split somewhere under results/experiment/<dataset>/split/
(exact filenames depend on the writer config). This script locates train.tsv
and test.tsv and copies them to data/split/{train,test}.tsv (header-free,
tab-separated) so Elliot and RecBole can consume them.

Run from `tables_reproducibility/table_5/`:
    python scripts/export_split.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "results" / "experiment"
SPLIT_DIR = ROOT / "data" / "split"


def _locate_split_file(name: str) -> Path:
    candidates = sorted(
        list(EXP_DIR.rglob(f"{name}.tsv")) + list(EXP_DIR.rglob(f"{name}.csv"))
    )
    if not candidates:
        raise SystemExit(
            f"[export_split] no {name}.tsv/.csv found under {EXP_DIR}. "
            f"Did the WarpRec training step run?"
        )
    if len(candidates) > 1:
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
                stripped = line.rstrip("\n")
                if "," in stripped and "\t" not in stripped:
                    fout.write("\t".join(stripped.split(",")) + "\n")
                else:
                    fout.write(line)
        print(f"[export_split] {src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
