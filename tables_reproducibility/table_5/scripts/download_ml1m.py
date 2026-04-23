"""Download MovieLens-1M and convert it to the CSV format WarpRec consumes.

Input  : https://files.grouplens.org/datasets/movielens/ml-1m.zip
Output : data/raw/movielens.csv   (user,item,rating,timestamp — comma-separated, no header)

Stdlib-only; runs from any of the three framework envs or the host interpreter.
Idempotent: re-running with an already-downloaded archive is a no-op.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_MD5 = "c4d9eecfca2ab87c1945afe126590906"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "ml-1m.zip"
CSV_PATH = RAW_DIR / "movielens.csv"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and _md5(ZIP_PATH) == EXPECTED_MD5:
        print(f"[download] {ZIP_PATH} already present (md5 OK)")
        return
    print(f"[download] fetching {URL}")
    with urllib.request.urlopen(URL) as resp, ZIP_PATH.open("wb") as out:
        shutil.copyfileobj(resp, out)
    got = _md5(ZIP_PATH)
    if got != EXPECTED_MD5:
        raise SystemExit(
            f"[download] md5 mismatch: expected {EXPECTED_MD5}, got {got}"
        )
    print(f"[download] wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")


def _convert() -> None:
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        print(f"[convert] {CSV_PATH} already present — skipping")
        return
    print(f"[convert] extracting ratings.dat from {ZIP_PATH.name}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("ml-1m/ratings.dat") as src, CSV_PATH.open("w") as dst:
            n = 0
            for raw in src:
                line = raw.decode("latin-1").rstrip("\n")
                if not line:
                    continue
                u, i, r, t = line.split("::")
                dst.write(f"{u},{i},{r},{t}\n")
                n += 1
    print(f"[convert] wrote {CSV_PATH} ({n} rows)")


def main() -> None:
    _download()
    _convert()
    print("[done]")


if __name__ == "__main__":
    sys.exit(main())
