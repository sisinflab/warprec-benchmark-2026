"""Thin wrapper: run Elliot on a config file.

Elliot's CLI is `python -c "from elliot.run import run_experiment; run_experiment(...)"`.
This wrapper preserves the CWD (the reproducibility root) so that relative
paths in the YAML — e.g. `data/split/train.tsv`, `recs/recbole/LightGCN.tsv`,
`path_output_rec_result` — resolve correctly.

Usage:
    python scripts/run_elliot.py <path-to-config.yml>
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_elliot.py <config.yml>", file=sys.stderr)
        return 2
    cfg = Path(sys.argv[1])
    if not cfg.is_file():
        print(f"Config not found: {cfg}", file=sys.stderr)
        return 2

    from elliot.run import run_experiment  # imported here to keep --help fast

    run_experiment(str(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
