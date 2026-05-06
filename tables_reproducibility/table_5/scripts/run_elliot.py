"""Wrapper around `elliot.run.run_experiment` that normalizes paths.

Elliot resolves paths in two different ways depending on the key:
  * `data_config.{train_path,validation_path,test_path}` and
    `models.ProxyRecommender.path` go through `_safe_set_path`, which
    treats the string as relative to the CONFIG DIR (not CWD) when it
    "looks like" a path (matches an internal regex).
  * `path_output_rec_{result,weight,performance}` go through
    `os.path.abspath`, i.e. relative to CWD.

To keep the YAMLs human-readable we let users write simple paths relative
to the project root (e.g. `data/split/train.tsv`); this wrapper:
  1. chdir's to ROOT (= tables_reproducibility/table_5/), so CWD-based
     `path_output_rec_*` resolve correctly.
  2. rewrites the data/proxy paths to absolute form before launching
     Elliot, so they don't get re-resolved against the config dir.

Usage:
    python scripts/run_elliot.py frameworks/elliot/train.yml
"""
import os
import sys
import tempfile
from pathlib import Path

import yaml
from elliot.run import run_experiment

ROOT = Path(__file__).resolve().parent.parent  # tables_reproducibility/table_5/

DATA_CFG_KEYS = ("train_path", "validation_path", "test_path", "side_information")
PROXY_KEYS = ("path", "recommendation_file")


def _abs(p: str) -> str:
    return str((ROOT / p).resolve()) if not os.path.isabs(p) else p


def _rewrite_paths(cfg: dict) -> dict:
    exp = cfg.get("experiment", {})
    dc = exp.get("data_config", {})
    if isinstance(dc, dict):
        for k in DATA_CFG_KEYS:
            v = dc.get(k)
            if isinstance(v, str) and ("/" in v or v.endswith((".tsv", ".csv"))):
                dc[k] = _abs(v)
    ext_path = exp.get("external_models_path")
    if isinstance(ext_path, str) and not os.path.isabs(ext_path):
        exp["external_models_path"] = _abs(ext_path)
    models = exp.get("models", {})
    if isinstance(models, dict):
        for _name, params in models.items():
            if isinstance(params, dict):
                for k in PROXY_KEYS:
                    v = params.get(k)
                    if isinstance(v, str) and ("/" in v or v.endswith((".tsv", ".csv"))):
                        params[k] = _abs(v)
    return cfg


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_elliot.py <config.yml>", file=sys.stderr)
        return 2
    cfg_arg = Path(sys.argv[1])
    if not cfg_arg.is_absolute():
        cfg_arg = (ROOT / cfg_arg).resolve()
    if not cfg_arg.is_file():
        print(f"config not found: {cfg_arg}", file=sys.stderr)
        return 2

    os.chdir(ROOT)

    with cfg_arg.open() as f:
        cfg = yaml.safe_load(f)
    cfg = _rewrite_paths(cfg)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, dir=str(ROOT)
    ) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp_path = tmp.name
    try:
        run_experiment(tmp_path)
    finally:
        os.unlink(tmp_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
