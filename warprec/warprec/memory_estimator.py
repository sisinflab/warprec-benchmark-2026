"""Stage-by-stage RAM/VRAM estimator for WarpRec HPO pipelines.

Unlike the legacy `estimate_memory.py`, this module simulates the *entire*
pipeline — driver and Ray worker, preprocessing through evaluation —
producing a cumulative (RSS-like, monotonic) peak per stage per trial.
The model is parametrized by a small `CALIBRATION` block at the top so
future tuning touches one place.

Usage
-----
    python -m warprec.memory_estimator -c path/to/config.yml
    python -m warprec.memory_estimator -c path/to/config.yml --format json
    python -m warprec.memory_estimator -c path/to/config.yml --validate
        ^ reads sibling ray_results/ and compares estimate vs measured.

Design
------
- Process model: driver + worker, each with (ram_live, ram_peak, vram_live,
  vram_peak). Free events decrement `live` but NOT `peak` — this models the
  reality that glibc does not return pages to the kernel, so RSS watermark
  is monotonic non-decreasing. Torch's CUDA allocator has a similar
  watermark semantic (`torch.cuda.max_memory_allocated`).
- Stages: 7 driver stages (D0..D6) + 8 worker stages (W0..W7), each a pure
  function that records events on a ProcessState and emits a StageReport.
- Per-trial estimate = cumulative peak over all worker stages for that
  trial (driver peak is shared across trials in the HPO).
- Model paradigms are handled by `MODEL_ESTIMATORS`: closed-form similarity
  (ItemKNN/UserKNN), closed-form Gram-based (EASE/ADMMSlim/SLIM), bilinear
  embedding (BPR/NeuMF/...), graph (LightGCN/NGCF), sequential (SASRec).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    import yaml
except ImportError as exc:
    print("ERROR: PyYAML required ('pip install pyyaml').", file=sys.stderr)
    raise SystemExit(1) from exc


# =============================================================================
# Units & small utilities
# =============================================================================

FP16 = 2
FP32 = 4
FP64 = 8
INT32 = 4
INT64 = 8
BOOL1 = 1

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB


def _human(nbytes: float) -> str:
    nbytes = float(nbytes)
    if nbytes < KIB:
        return f"{nbytes:.4f} B"
    if nbytes < MIB:
        return f"{nbytes / KIB:.4f} KiB"
    if nbytes < GIB:
        return f"{nbytes / MIB:.4f} MiB"
    return f"{nbytes / GIB:.4f} GiB"


def _gib(nbytes: float) -> float:
    return float(nbytes) / GIB


# =============================================================================
# CALIBRATION — tuned against ground-truth runs on Netflix-100M
# =============================================================================
#
# These constants absorb the parts of real-world RSS that are hard to derive
# from first principles: import baselines, allocator fragmentation, CUDA
# context size, Ray object-store mapping overhead, BLAS scratch buffers.
#
# They were fit against:
#   ItemKNN (6 trials): ram≈10.89 GiB, vram≈305 MiB
#   EASE    (6 trials): ram≈12.12 GiB, vram≈425 MiB
#   NeuMF   (2 trials): ram≈11.45 GiB, vram∈{8.00, 11.49} GiB
#   LightGCN (6 trials): ram≈11.40 GiB, vram∈{14.48, 18.87} GiB
#
# When adding new ground-truth data, touch only this block.

CALIBRATION = {
    # Driver process baseline (warprec.run imports, polars, torch, ray client).
    "driver_baseline_ram": 1.2 * GIB,

    # Worker process baseline (Ray worker + warprec modules + torch CPU).
    "worker_baseline_ram": 2.0 * GIB,

    # CUDA runtime context reservation in host RAM (pinned mem pool + driver
    # structures). torch.cuda.max_memory_allocated() does NOT count this —
    # it reports only tensor allocations — so we don't add it to VRAM.
    "cuda_context_ram": 0.35 * GIB,

    # Polars peak multiplier over the final DataFrame (working copies during
    # k-core iteration + split + to_arrow conversions).
    "polars_peak_multiplier": 2.5,

    # Inflation factor for `pickle.loads(Dataset)` in the Ray worker vs the
    # on-disk pickled size. Covers Python object overhead + duplicated
    # mappings + deserialized torch tensors.
    "dataset_deserialize_inflation": 1.8,

    # Ray object-store mmap overhead per worker that receives the dataset.
    # The plasma store exposes the pickled blob via shared memory; the
    # worker's RSS accounts for the mapped pages until it's unmapped.
    "ray_objectstore_overhead": 0.4 * GIB,

    # Allocator fragmentation — how much of the logical "freed" bytes the
    # OS still holds as resident. Applied as a fraction of the largest
    # *small* transient allocation peak during the trial. Large allocations
    # (> MMAP_THRESHOLD bytes below) are mmap'd by glibc and return pages
    # to the OS cleanly on free, so they don't contribute to RSS drift.
    "fragmentation_retained_frac": 0.35,
    "mmap_threshold_bytes": 128 * MIB,  # glibc default M_MMAP_THRESHOLD

    # sklearn.cosine_similarity + subsequent top-k filtering total peak as a
    # multiple of the dense output bytes. Captures: X sparse + X_norm sparse +
    # BLAS scratch + dense output (sim_matrix) + torch.zeros_like scatter
    # buffer (final item_similarity) — all concurrent at peak. Calibrated
    # against ItemKNN on Netflix-100M.
    "itemknn_fit_peak_mult": 4.0,

    # EASE fit peak as a multiple of the Gram matrix bytes:
    # G (fp64) + B (fp64, np.linalg.inv output) alive at same time + small
    # LAPACK workspace. numpy's inv uses dgesv which overwrites A, so no 2×
    # workspace — calibrated against EASE on Netflix-100M.
    "ease_fit_peak_mult": 2.5,

    # Generic LU/QR workspace multiplier for closed-form linear models.
    "inv_workspace_mult": 1.5,

    # Evaluator-side overhead beyond the shared CSR (item_cluster tensors,
    # BaseMetric buffers, torchmetrics state per k).
    "evaluator_overhead_frac": 0.05,  # of dataset size

    # Torch cuBLAS/cuDNN workspaces reserved on first matmul/conv call.
    # Appears in `max_memory_allocated()`.
    "cuda_workspace_vram": 50 * MIB,

    # For closed-form models, the evaluator computes binary_relevance and
    # discounted_relevance tensors at [eval_bs, n_items] alongside
    # predictions — each an FP32 copy of the scoring matrix. At peak this
    # counts ~4 such tensors (predictions + eval_batch + binary_rel + disc_rel).
    "closed_form_eval_concurrent_tensors": 4,

    # Extra RAM held by iterative training: pin_memory pool, dataloader
    # worker COW pages, pointwise per-sample cached tensors, BLAS scratch
    # during matmul — not modeled structurally. Calibrated against NeuMF
    # on Netflix-100M (5.3 GiB observed flat across mf/neg_samples).
    "iterative_train_ram_overhead": 5.35 * GIB,

    # Sequential models lazily build `Sessions._cached_user_histories`
    # (dict[user_id → list[int]]) during evaluation. Each interaction costs
    # a Python int (~28 B for values > 256) plus a list pointer (~8 B),
    # and each user adds a list object (~56 B). Calibrated against SASRec
    # on Netflix-100M (3.6 GiB observed).
    "sequential_cached_history_bytes_per_interaction": 36,
    "sequential_cached_history_bytes_per_user": 56,

    # LightGCN/NGCF use LGConv with default normalize=True. Each forward calls
    # gcn_norm(), which creates a SparseTensor saved for backward, plus
    # transient copies during torch_sparse.mul. The first two layers share the
    # base normalization/transient footprint below; each additional LGConv
    # layer adds another saved SparseTensor + torch_sparse.mul workspace.
    "graph_spmm_workspace_base_per_interaction": 143,
    "graph_spmm_workspace_extra_layer_per_interaction": 24,

    # Extra N·d activation tensors saved by LightGCN propagation. Calibration
    # from the completed Netflix-100M LightGCN sweep shows one additional
    # N·d tensor per layer beyond the first.
    "graph_activation_tensor_base_count": 1,
}


# =============================================================================
# Dataset resolution
# =============================================================================

@dataclass
class DatasetStats:
    n_users: int
    n_items: int
    n_interactions: int
    source: str  # where the numbers came from (scan: / registry: / override:)


_KNOWN_DATASETS: dict[str, DatasetStats] = {
    # Values from the respective papers' standard splits.
    "amazon-book": DatasetStats(52_643, 91_599, 2_984_108, "LightGCN paper"),
    "yelp2018":    DatasetStats(31_668, 38_048, 1_546_574, "LightGCN paper"),
    "gowalla":     DatasetStats(29_858, 40_981, 1_027_370, "LightGCN paper"),
    "coat":        DatasetStats(290, 300, 6_960, "Coat paper"),
    # Netflix-Prize 100M — verified via polars scan 2026-04-23.
    "netflix-prize-100m": DatasetStats(480_189, 17_770, 100_480_507, "Netflix CSV scan"),
}


def _resolve_dataset_stats(cfg: dict, config_path: Path) -> DatasetStats:
    reader = cfg.get("reader", {}) or {}
    writer = cfg.get("writer", {}) or {}
    dataset_name = writer.get("dataset_name", "unknown")

    loading_strategy = reader.get("loading_strategy", "dataset")
    labels = reader.get("labels", {}) or {}
    user_col = labels.get("user_id_label", "user_id")
    item_col = labels.get("item_id_label", "item_id")
    sep = reader.get("sep", "\t")

    # Build candidate paths
    candidate_paths: list[Path] = []
    cfg_dir_parent = config_path.parent.parent  # usually configs/<ds>/ → configs/
    experiments_dir = cfg_dir_parent.parent

    if loading_strategy == "dataset":
        local_path = reader.get("local_path")
        if local_path:
            p = Path(local_path)
            candidate_paths = [p, experiments_dir / p, config_path.parent / p]
    else:
        split = reader.get("split", {}) or {}
        local_path = split.get("local_path")
        ext = split.get("ext", ".tsv")
        if local_path:
            base = Path(local_path)
            candidate_paths = [base / f"train{ext}",
                               experiments_dir / base / f"train{ext}"]

    # 1) Dataset stats cache: try writer.dataset_name against registry
    for key in _KNOWN_DATASETS:
        if key.lower() in dataset_name.lower():
            return _KNOWN_DATASETS[key]

    # 2) Polars scan
    for p in candidate_paths:
        if p.exists():
            try:
                stats = _polars_scan(p, user_col, item_col, sep)
                if stats is not None:
                    return DatasetStats(*stats, source=f"scan:{p}")
            except Exception as exc:
                warnings.warn(f"Failed polars scan {p}: {exc}")

    raise RuntimeError(
        f"Could not resolve dataset stats for '{dataset_name}'. "
        f"Tried file scan at: {[str(p) for p in candidate_paths] or 'none'}. "
        f"Known datasets: {sorted(_KNOWN_DATASETS.keys())}."
    )


def _polars_scan(path: Path, user_col: str, item_col: str, sep: str) -> Optional[tuple[int, int, int]]:
    try:
        import polars as pl
    except ImportError:
        return None
    frame = pl.scan_csv(str(path), separator=sep, has_header=True)
    result = frame.select(
        pl.col(user_col).n_unique().alias("n_users"),
        pl.col(item_col).n_unique().alias("n_items"),
        pl.len().alias("n_interactions"),
    ).collect()
    row = result.row(0, named=True)
    return int(row["n_users"]), int(row["n_items"]), int(row["n_interactions"])


# =============================================================================
# Process-state model
# =============================================================================
#
# Allocator semantics we simulate:
#   - alloc(device, bytes)  : live += bytes ; peak = max(peak, live)
#   - free(device, bytes)   : live -= bytes ; peak unchanged (monotonic)
#   - transfer(ram→vram)    : ram.free(b) ; vram.alloc(b)  [no net RAM change
#                              because CUDA allocations reserve RAM too — we
#                              model that explicitly where it matters]
#
# RSS-vs-logical distinction: in reality RSS lags logical live bytes because
# pages aren't returned. For our purposes `peak` is the relevant quantity —
# it's what `psutil.Process.memory_info().rss` will read as the watermark.

@dataclass
class Allocation:
    label: str
    device: str  # "ram" | "vram"
    nbytes: int
    stage: str
    lifetime: str = "persistent"  # "persistent" | "transient"


@dataclass
class StageEvent:
    kind: str  # "alloc" | "free"
    label: str
    device: str
    nbytes: int


@dataclass
class StageReport:
    stage: str
    process: str  # "driver" | "worker"
    events: list[StageEvent] = field(default_factory=list)
    ram_live_before: int = 0
    ram_live_after: int = 0
    ram_peak_after: int = 0
    vram_live_before: int = 0
    vram_live_after: int = 0
    vram_peak_after: int = 0
    notes: list[str] = field(default_factory=list)


class ProcessState:
    """Models the RAM/VRAM of a single process over its lifetime."""

    def __init__(self, name: str):
        self.name = name
        self.ram_live: int = 0
        self.ram_peak: int = 0
        self.vram_live: int = 0
        self.vram_peak: int = 0
        self.history: list[StageReport] = []
        self._fragmentation_buffer: int = 0  # transient peaks retained by allocator

    def alloc(self, report: StageReport, label: str, nbytes: int, device: str = "ram"):
        nbytes = max(0, int(nbytes))
        if nbytes == 0:
            return
        report.events.append(StageEvent("alloc", label, device, nbytes))
        if device == "ram":
            self.ram_live += nbytes
            self.ram_peak = max(self.ram_peak, self.ram_live)
        else:
            self.vram_live += nbytes
            self.vram_peak = max(self.vram_peak, self.vram_live)

    def free(self, report: StageReport, label: str, nbytes: int, device: str = "ram"):
        """Free bytes. For RAM frees smaller than MMAP_THRESHOLD the bytes
        are tracked in the fragmentation buffer (glibc arenas don't return
        pages to the OS below that threshold). Larger frees are mmap-backed
        and return cleanly."""
        nbytes = max(0, int(nbytes))
        if nbytes == 0:
            return
        report.events.append(StageEvent("free", label, device, nbytes))
        if device == "ram":
            if nbytes < CALIBRATION["mmap_threshold_bytes"]:
                self._fragmentation_buffer = max(self._fragmentation_buffer, nbytes)
            self.ram_live = max(0, self.ram_live - nbytes)
        else:
            self.vram_live = max(0, self.vram_live - nbytes)

    def begin_stage(self, stage: str) -> StageReport:
        rpt = StageReport(stage=stage, process=self.name,
                          ram_live_before=self.ram_live,
                          vram_live_before=self.vram_live)
        return rpt

    def end_stage(self, rpt: StageReport):
        rpt.ram_live_after = self.ram_live
        rpt.ram_peak_after = self.ram_peak
        rpt.vram_live_after = self.vram_live
        rpt.vram_peak_after = self.vram_peak
        self.history.append(rpt)

    def apply_fragmentation(self, frac: float):
        """Bump RSS peak to absorb transient allocator retention."""
        retain = int(self._fragmentation_buffer * frac)
        if retain > 0:
            self.ram_peak = max(self.ram_peak, self.ram_live + retain)


# =============================================================================
# Model-paradigm structural estimators
# =============================================================================
#
# Each returns a dict with three groups of bytes:
#   "params_ram"      : persistent RAM occupied after __init__ (e.g., numpy
#                        item_similarity for closed-form)
#   "params_vram"     : persistent VRAM after model.to(device)
#   "fit_peak_extra"  : TRANSIENT peak *above* params during __init__ (e.g.,
#                        Gram + inv workspace)
#   "train_vram_peak" : peak VRAM during training activations + grads (0 for
#                        closed-form) — does NOT include optimizer state or
#                        params, those are separate
#   "eval_vram_peak"  : peak VRAM during evaluation for a single batch (full
#                        or sampled depending on strategy)
#   "notes"           : list[str]
#   "closed_form"     : bool — training is fit once in __init__
#   "has_optimizer"   : bool — iterative models allocate Adam state

@dataclass
class ModelMemory:
    name: str
    params_ram: int = 0
    params_vram: int = 0            # learnable parameters (get grad + optim state)
    buffers_vram: int = 0           # non-learnable persistent VRAM (e.g. adjacency,
                                    # causal_mask) — alive from W3 onwards, no grad
    fit_peak_extra: int = 0         # transient extra during fit/__init__, above params
    train_vram_peak: int = 0        # activations + grads during one batch
    eval_vram_peak: int = 0         # activations during one eval batch
    eval_ram_extra: int = 0         # extra RAM allocated lazily during eval
    closed_form: bool = False
    has_optimizer: bool = True
    dtype_bytes: int = FP32
    notes: list[str] = field(default_factory=list)

    def optimizer_state_bytes(self) -> int:
        """Adam keeps 2 buffers (m, v) same shape as learnable params."""
        if not self.has_optimizer:
            return 0
        return 2 * max(self.params_vram, self.params_ram)

    def grad_bytes(self) -> int:
        if self.closed_form or not self.has_optimizer:
            return 0
        return max(self.params_vram, self.params_ram)


# ---------- helpers shared across models ----------

def _pad(n_items: int) -> int:
    """Most WarpRec item embeddings have +1 padding row (padding_idx=n_items)."""
    return n_items + 1


def _embedding(rows: int, dim: int, dtype_bytes: int = FP32) -> int:
    return int(rows) * int(dim) * dtype_bytes


def _linear(in_dim: int, out_dim: int, dtype_bytes: int = FP32) -> int:
    return (in_dim * out_dim + out_dim) * dtype_bytes  # weight + bias


def _mlp(dims: list[int], dtype_bytes: int = FP32) -> int:
    return sum(_linear(a, b, dtype_bytes) for a, b in zip(dims[:-1], dims[1:]))


# ---------- closed-form similarity (X.T @ X → dense) ----------

def _est_itemknn(ds: DatasetStats, hp: dict) -> ModelMemory:
    # ItemKNN pipeline (itemknn.py:44-54):
    #   X = interactions.get_sparse()                          # CSR (shared ref)
    #   sim_matrix = torch.from_numpy(similarity.compute(X.T))
    #       ^ sklearn cosine_similarity on CSR input:
    #           - normalize(X.T) : new CSR copy (~X_bytes)
    #           - safe_sparse_dot(..., dense_output=True):
    #                output = DENSE [n_items, n_items] FP32
    #                internal scratch ≈ dense_output bytes
    #   filtered = _apply_topk_filtering(sim_matrix, k)
    #       ^ creates torch.zeros_like(sim_matrix) + topk buffers
    #   self.item_similarity = filtered.numpy()                # DENSE FP32
    #
    # Peak occurs during cosine_similarity (X_sparse + X_norm + scratch +
    # dense output) OR during topk (sim_matrix + zeros_like + small buffers).
    # Calibrated against Netflix-100M: peak ≈ 4× dense output bytes, mostly
    # independent of k (topk buffers never dominate).
    n_items = ds.n_items
    k = int(hp.get("k", 50))

    dense = n_items * n_items * FP32
    fit_peak = int(dense * CALIBRATION["itemknn_fit_peak_mult"])

    notes = [
        f"sklearn.cosine_similarity(X.T): dense FP32 [{n_items}²] = {_human(dense)}",
        f"top-{k} filter via zeros_like+scatter: {_human(n_items * k * 12)} (topk buffers)",
        f"final item_similarity (numpy FP32, kept on CPU): {_human(dense)}",
        f"calibrated fit peak (concurrent transients): {_human(fit_peak)}",
    ]
    if dense > 10 * GIB:
        notes.append(f"[!!] n_items²·4B={_human(dense)} exceeds 10 GiB — review feasibility.")

    return ModelMemory(
        name="ItemKNN",
        params_ram=dense,
        params_vram=0,
        fit_peak_extra=fit_peak,
        closed_form=True,
        has_optimizer=False,
        notes=notes,
    )


def _est_userknn(ds: DatasetStats, hp: dict) -> ModelMemory:
    n_users = ds.n_users
    k = int(hp.get("k", 50))
    dense = n_users * n_users * FP32
    fit_peak = int(dense * CALIBRATION["itemknn_fit_peak_mult"])
    notes = [
        f"UserKNN: n_users² dense FP32 = {_human(dense)} — scales badly with n_users.",
    ]
    if dense > 20 * GIB:
        notes.append(f"[!!] {_human(dense)} user-similarity matrix — almost certainly OOM.")
    return ModelMemory(name="UserKNN", params_ram=dense, fit_peak_extra=fit_peak,
                       closed_form=True, has_optimizer=False, notes=notes)


# ---------- closed-form Gram-based (EASE / ADMMSlim / SLIM) ----------

def _est_ease(ds: DatasetStats, hp: dict) -> ModelMemory:
    # EASE (ease.py:43-51):
    #   X = interactions.get_sparse()
    #   G = X.T @ X + l2 * I           # DENSE [n_items, n_items] FP64
    #   B = np.linalg.inv(G)           # dgesv overwrites A in-place; outputs B
    #   B /= -np.diag(B); fill_diagonal(B, 0)
    #   self.item_similarity = B       # DENSE FP64
    #
    # Peak: G and B live simultaneously during .inv(), plus small LAPACK
    # workspace. Calibrated against Netflix-100M: fit peak ≈ 2.5× Gram bytes.
    n_items = ds.n_items
    gram = n_items * n_items * FP64
    fit_peak = int(gram * CALIBRATION["ease_fit_peak_mult"])
    notes = [
        f"Gram G = X.T@X + λI: dense FP64 [{n_items}²] = {_human(gram)}",
        f"np.linalg.inv(G) via LAPACK dgesv: in-place LU, B output = {_human(gram)}",
        f"final item_similarity B (numpy FP64, kept on CPU): {_human(gram)}",
        f"calibrated fit peak (G + B + workspace): {_human(fit_peak)}",
    ]
    return ModelMemory(
        name="EASE",
        params_ram=gram,
        fit_peak_extra=fit_peak,
        closed_form=True,
        has_optimizer=False,
        dtype_bytes=FP64,
        notes=notes,
    )


def _est_admmslim(ds: DatasetStats, hp: dict) -> ModelMemory:
    n_items = ds.n_items
    dense_fp64 = n_items * n_items * FP64
    dense_fp32 = n_items * n_items * FP32
    # G (fp64) + P (fp32) + B_aux (fp32) + Gamma (fp32) + C (fp32) live together
    peak = dense_fp64 + 4 * dense_fp32 + dense_fp64 * (CALIBRATION["inv_workspace_mult"] - 1)
    notes = [
        f"ADMMSlim holds G (fp64) + P/B_aux/Γ/C (fp32) simultaneously: ~{_human(peak)}",
    ]
    return ModelMemory(
        name="ADMMSlim",
        params_ram=dense_fp32,  # final C is fp32
        fit_peak_extra=int(peak),
        closed_form=True,
        has_optimizer=False,
        notes=notes,
    )


def _est_slim(ds: DatasetStats, hp: dict) -> ModelMemory:
    n_items = ds.n_items
    # sklearn ElasticNet with precompute=True: Gram fp64 n_items²
    # Final item_similarity: .todense() → fp64 n_items² (despite sparsity)
    gram_fp64 = n_items * n_items * FP64
    peak = gram_fp64 + gram_fp64  # gram + dense-final materialization
    return ModelMemory(
        name="SLIM",
        params_ram=gram_fp64,
        fit_peak_extra=peak,
        closed_form=True,
        has_optimizer=False,
        dtype_bytes=FP64,
        notes=[f"SLIM per-item ElasticNet → dense FP64 similarity: {_human(gram_fp64)}"],
    )


# ---------- bilinear / neural (BPR, NeuMF, PureSVD, ConvNCF...) ----------

def _est_bpr(ds: DatasetStats, hp: dict) -> ModelMemory:
    d = int(hp.get("embedding_size", 64))
    batch = int(hp.get("batch_size", 1024))
    params = _embedding(ds.n_users, d) + _embedding(_pad(ds.n_items), d)
    # Train: each batch loads user/pos/neg embeddings + mul + sum → backward saves activations
    train = 3 * batch * d * FP32 * 2     # (user, pos, neg) × batch × d × fwd+bwd
    # Eval (full): user-block × item-block matrix scoring (blocked internally)
    eval_peak = _bilinear_eval_vram(ds, d, hp)
    return ModelMemory(
        name="BPR", params_ram=0, params_vram=params,
        train_vram_peak=train, eval_vram_peak=eval_peak,
        notes=[f"BPR params: {_human(params)} (user+item embeddings, d={d})"],
    )


def _est_neumf(ds: DatasetStats, hp: dict) -> ModelMemory:
    mf = int(hp.get("mf_embedding_size", 64))
    mlp = int(hp.get("mlp_embedding_size", 64))
    batch = int(hp.get("batch_size", 1024))
    mlp_hidden = hp.get("mlp_hidden_size", [64, 32])
    # Embeddings: user/item × (mf + mlp)
    params = (_embedding(ds.n_users, mf) + _embedding(_pad(ds.n_items), mf)
              + _embedding(ds.n_users, mlp) + _embedding(_pad(ds.n_items), mlp))
    # MLP: [2*mlp] + mlp_hidden → last → 1
    mlp_dims = [2 * mlp] + list(mlp_hidden)
    params += _mlp(mlp_dims)
    # Final predict layer: Linear(mf + mlp_hidden[-1], 1)
    params += _linear(mf + int(mlp_hidden[-1]), 1)
    # Train activations per batch (pointwise): all embeddings + mlp activations × 2 (fwd+bwd)
    emb_acts = batch * (2 * mf + 2 * mlp) * FP32  # lookups
    mlp_acts = batch * (2 * mlp + sum(mlp_hidden)) * FP32  # mlp intermediates
    train = 2 * (emb_acts + mlp_acts)
    # Eval: NeuMF.predict loops block_size items at a time; batch=eval_batch_size=1024
    eval_peak = _neumf_eval_vram(ds, mf, mlp, mlp_hidden, hp)
    return ModelMemory(
        name="NeuMF", params_ram=0, params_vram=params,
        train_vram_peak=train, eval_vram_peak=eval_peak,
        notes=[f"NeuMF params: {_human(params)} (mf={mf}, mlp={mlp}, hidden={mlp_hidden})"],
    )


def _est_puresvd(ds: DatasetStats, hp: dict) -> ModelMemory:
    # PureSVD: truncated SVD of X → U (n_users × k), V (n_items × k)
    # scipy ARPACK uses Lanczos; working set ~ several vectors × n_max
    k = int(hp.get("embedding_size", 64))
    u = _embedding(ds.n_users, k)
    v = _embedding(_pad(ds.n_items), k)
    # Lanczos workspace: ~5×k vectors of size max(n_users, n_items) in fp64
    workspace = 5 * k * max(ds.n_users, ds.n_items) * FP64
    return ModelMemory(
        name="PureSVD",
        params_ram=u + v,
        fit_peak_extra=workspace,
        closed_form=True,
        has_optimizer=False,
        notes=[f"PureSVD: U+V = {_human(u+v)}; ARPACK workspace ≈ {_human(workspace)}"],
    )


# ---------- graph (LightGCN / NGCF) ----------

def _est_lightgcn(ds: DatasetStats, hp: dict) -> ModelMemory:
    # LightGCN (lightgcn.py:57-93 + graph_utils.py + torch_geometric.nn.LGConv):
    #   self.adj = SparseTensor(row, col, sparse_sizes=(N, N)) on VRAM
    #       ^ N = n_users + n_items + 1 ; 2·nnz_train bidirectional edges
    #   self.propagation_network = [LGConv(), LGConv(), ...] with
    #       normalize=True (DEFAULT), so each forward calls gcn_norm() that
    #       creates a new SparseTensor (col [2·nnz] int64 + values [2·nnz] fp32)
    #       ≈ 12 B/edge, plus transient copies during torch_sparse.mul,
    #       saved by autograd for backward.
    #   Training: forward() holds embeddings_list = (n_layers+1) tensors
    #       [N, d] × FP32 saved for backward.
    #   Eval: predict() → propagate_embeddings() caches once, reuses for
    #       subsequent batches; only einsum output per batch is new.
    d = int(hp.get("embedding_size", 64))
    n_layers = int(hp.get("n_layers", 3))
    batch = int(hp.get("batch_size", 1024))

    # Adjacency is built from `interactions.get_sparse().tocoo()` where
    # `interactions` is the TRAIN split (~90% of total for default 0.1
    # test ratio).
    n_train = int(ds.n_interactions * 0.9)

    # Persistent learnable parameters (VRAM)
    params = _embedding(ds.n_users, d) + _embedding(_pad(ds.n_items), d)

    # Adjacency SparseTensor on VRAM after adj.to(device). CSR-like storage:
    # rowptr [N+1] int64 + col [2·n_train] int64 (no values — unweighted).
    N = ds.n_users + ds.n_items + 1
    adj_vram = (N + 1) * INT64 + 2 * n_train * INT64

    # Adjacency construction in __init__ allocates large transient numpy/torch
    # arrays on RAM before the SparseTensor is moved to GPU (~6 GiB peak for
    # Netflix-100M). These are numpy/torch tensors > mmap_threshold so the
    # OS reclaims their pages cleanly on __init__ exit; they do NOT appear
    # in the end-of-trial RSS measurement. We note the figure for awareness
    # but don't simulate it as a watermark event.
    adj_init_ram_transient = 0
    adj_init_ram_transient_peak = (
        2 * n_train * INT32                   # row numpy (int32)
        + 2 * n_train * INT32                 # col numpy
        + 2 * 2 * n_train * INT32             # edge_index_np vstack
        + 2 * 2 * n_train * INT64             # torch.tensor(int64) copy
    )

    # Training peak VRAM.
    extra_layers = max(0, n_layers - 2)
    extra_workspace = CALIBRATION["graph_spmm_workspace_extra_layer_per_interaction"]
    graph_workspace_per_interaction = (
        CALIBRATION["graph_spmm_workspace_base_per_interaction"]
        + extra_layers * extra_workspace
    )
    activation_tensor_count = (
        CALIBRATION["graph_activation_tensor_base_count"]
        + max(0, n_layers - 2)
    )
    gcn_norm_overhead = n_train * graph_workspace_per_interaction
    activation_stack = activation_tensor_count * N * d * FP32
    cublas = 200 * MIB
    train = gcn_norm_overhead + activation_stack + cublas

    # Evaluation peak VRAM: cached user_all/item_all on GPU + per-batch
    # einsum score tensor + evaluator metric tensors.
    eval_bs = int(hp.get("eval_batch_size", 1024))
    cached_prop_vram = N * d * FP32
    einsum_vram = eval_bs * ds.n_items * FP32
    eval_metrics_vram = 3 * eval_bs * ds.n_items * FP32
    eval_peak = cached_prop_vram + einsum_vram + eval_metrics_vram + cublas

    return ModelMemory(
        name="LightGCN",
        params_vram=params,                       # learnable → grad + optim
        buffers_vram=adj_vram,                    # non-learnable (adj), no grad
        fit_peak_extra=adj_init_ram_transient,    # RAM transient during __init__
        train_vram_peak=train,
        eval_vram_peak=eval_peak,
        notes=[
            f"LightGCN learnable params (VRAM): {_human(params)}  (d={d})",
            f"adjacency SparseTensor (VRAM, buffer): {_human(adj_vram)}  "
            f"(2·n_train={2*n_train:,} bidirectional edges)",
            f"gcn_norm saved workspace (VRAM): {_human(gcn_norm_overhead)}  "
            f"({graph_workspace_per_interaction} B/edge × {n_train:,}; "
            f"extra_layers={extra_layers})",
            f"activation stack (VRAM, n_layers={n_layers}, "
            f"tensors={activation_tensor_count}): {_human(activation_stack)}",
            f"adj __init__ transient RAM peak (mmap, not in end-RSS): "
            f"{_human(adj_init_ram_transient_peak)}",
        ],
    )


# ---------- sequential (SASRec) ----------

def _est_sasrec(ds: DatasetStats, hp: dict) -> ModelMemory:
    d = int(hp["embedding_size"])
    n_layers = int(hp["n_layers"])
    n_heads = int(hp["n_heads"])
    inner = int(hp["inner_size"])
    max_seq = int(hp["max_seq_len"])
    batch = int(hp["batch_size"])

    item_emb = _embedding(_pad(ds.n_items), d)
    pos_emb = _embedding(max_seq, d)
    # TransformerEncoderLayer: MHA (qkv+out, 4 × d²) + FF (d·inner + inner·d) + 2 LayerNorm
    per_layer_params = 4 * (d * d + d) + 2 * (d * inner + inner + inner * d + d) + 2 * (2 * d)
    params = item_emb + pos_emb + n_layers * per_layer_params + 2 * d  # +final LN

    # Activation accounting per TransformerEncoderLayer with saved-for-backward
    # tensors + dropout masks (dropout_prob and attn_dropout_prob enabled):
    #   - 6 × seq_tok: input, Q, K, V, post-attn, post-FF output
    #   - 2 × seq_tok: LayerNorm1/LayerNorm2 outputs
    #   - 2 × seq_tok: post-attn-dropout mask + post-FF-dropout mask
    #   - 2 × attn   : attn scores + attn softmax (attn_dropout saves mask)
    #   - 2 × ff     : FF intermediate + its dropout mask
    # Calibrated against 4 SASRec configs on Netflix-100M (±2%).
    seq_tok = batch * max_seq * d * FP32
    attn = batch * n_heads * max_seq * max_seq * FP32
    ff = batch * max_seq * inner * FP32
    per_layer_act = 10 * seq_tok + 2 * attn + 2 * ff

    # Embedding layer outputs (item_emb + pos_emb + combined after LN+dropout)
    embedding_acts = 3 * seq_tok
    # train_step scalar side: pos_items_emb, neg_items_emb, plus reg_loss
    # re-fetch of item_embedding(item_seq)
    train_step_extras = seq_tok + 2 * batch * d * FP32
    # cuBLAS scratch during large matmul (observed ~200 MiB on transformers)
    cublas = 200 * MIB

    train = n_layers * per_layer_act + embedding_acts + train_step_extras + cublas

    # Evaluation: forward-only (no saved activations), plus [eval_bs, n_items]
    # score matrix from einsum('be,ie->bi', seq_output, item_embeddings).
    eval_bs = int(hp.get("eval_batch_size", 1024))
    eval_seq_tok = eval_bs * max_seq * d * FP32
    eval_attn = eval_bs * n_heads * max_seq * max_seq * FP32
    eval_ff = eval_bs * max_seq * inner * FP32
    eval_peak = (
        n_layers * (4 * eval_seq_tok + eval_attn + eval_ff)
        + eval_bs * ds.n_items * FP32      # predictions
        + 3 * eval_bs * ds.n_items * FP32  # evaluator: gt + binary + discounted
        + cublas
    )

    # Sequential model RAM overhead (lazy Sessions._cached_user_histories
    # built during evaluation in the worker).
    seq_ram = (
        ds.n_interactions * CALIBRATION["sequential_cached_history_bytes_per_interaction"]
        + ds.n_users * CALIBRATION["sequential_cached_history_bytes_per_user"]
    )

    return ModelMemory(
        name="SASRec",
        params_vram=params,
        train_vram_peak=train,
        eval_vram_peak=eval_peak,
        eval_ram_extra=seq_ram,      # Sessions._cached_user_histories
        notes=[
            f"SASRec params: {_human(params)} (d={d}, layers={n_layers}, ff={inner})",
            f"per-layer activation (train, with dropout): {_human(per_layer_act)}",
            f"Sessions cached history (lazy in eval, RAM): {_human(seq_ram)}",
        ],
    )


# ---------- eval-VRAM helpers (for iterative models) ----------

def _bilinear_eval_vram(ds: DatasetStats, d: int, hp: dict) -> int:
    """Peak VRAM for bilinear scoring (BPR-like) during evaluation.

    For full strategy, predict() typically produces [eval_bs, n_items]. Some
    models block it internally but default is materialize-all. Closed form:
        peak ≈ eval_bs · n_items · 4B (predictions)
             + eval_bs · d · 4B (user emb)
             + block · d · 4B (item emb slice)
    """
    eval_bs = int(hp.get("eval_batch_size", 1024))
    block_size = int(hp.get("block_size", 50))
    # User emb: eval_bs × d ; Item emb (slice): block_size × d ; Scores: eval_bs × n_items
    preds = eval_bs * ds.n_items * FP32
    scratch = 3 * eval_bs * block_size * d * FP32  # 3 = (u,i,score) pointwise expand
    return max(preds, scratch) + 2 * eval_bs * d * FP32


def _neumf_eval_vram(ds: DatasetStats, mf: int, mlp: int, mlp_hidden, hp: dict) -> int:
    """NeuMF.predict pattern:
        for block in 0..n_items step block_size:
            users_block = users.expand(bs, block).reshape(-1)   # [bs·block]
            items_block = items.expand(bs, block).reshape(-1)   # [bs·block]
            logits = self.forward(users_block, items_block)
    The peak is inside forward where mf_user_e + mf_item_e + mf_output all live.
    """
    eval_bs = int(hp.get("eval_batch_size", 1024))
    block = int(hp.get("block_size", 50))
    pairs = eval_bs * block
    # Concurrent-alive tensors inside forward (rough upper-bound, conservative):
    #   users_block, items_block (int64)     : 2 × pairs × 8B
    #   mf_user_e, mf_item_e, mf_output      : 3 × pairs × mf × 4B
    #   mlp_user_e, mlp_item_e, mlp_input    : 2·mlp + 2·mlp (cat) = (2·mlp + 2·mlp) × pairs × 4B
    #   mlp_hidden outputs (largest)         : pairs × max(mlp_hidden) × 4B
    #   combined cat before predict_layer    : pairs × (mf + mlp_hidden[-1]) × 4B
    indices_b = 2 * pairs * INT64
    mf_peak = 3 * pairs * mf * FP32
    mlp_in = 2 * pairs * mlp * FP32 + pairs * 2 * mlp * FP32  # lookups + concat
    mlp_hid = pairs * max(mlp_hidden) * FP32 if mlp_hidden else 0
    combined = pairs * (mf + int(mlp_hidden[-1]) if mlp_hidden else 0) * FP32
    preds_accum = eval_bs * ds.n_items * FP32  # final concatenated [eval_bs, n_items]
    # These tensors aren't strictly all concurrent. Peak reached inside
    # forward() is dominated by: 2× mf_peak (mf_user_e + mf_item_e alive
    # when mul is invoked, output overwrites one of them) + 2× mlp_emb
    # (mlp_user_e + mlp_item_e before concat) + concat output.
    peak = (
        indices_b
        + 2 * pairs * mf * FP32          # mf_user_e + mf_item_e
        + pairs * mf * FP32              # mf_output
        + 2 * pairs * mlp * FP32         # mlp_user_e + mlp_item_e
        + 2 * pairs * mlp * FP32         # mlp_input concat
        + mlp_hid                        # largest hidden
        + combined                       # cat(mf_out, mlp_out)
        + preds_accum                    # accumulated [eval_bs, n_items]
    )
    return int(peak)


MODEL_ESTIMATORS: dict[str, Callable[[DatasetStats, dict], ModelMemory]] = {
    "ItemKNN": _est_itemknn,
    "UserKNN": _est_userknn,
    "AttributeItemKNN": _est_itemknn,     # superset notes on top, but same order
    "AttributeUserKNN": _est_userknn,
    "EASE": _est_ease,
    "ADMMSlim": _est_admmslim,
    "SLIM": _est_slim,
    "PureSVD": _est_puresvd,
    "BPR": _est_bpr,
    "NeuMF": _est_neumf,
    "LightGCN": _est_lightgcn,
    "NGCF": _est_lightgcn,                # structurally equivalent for peak bytes
    "SASRec": _est_sasrec,
}


# =============================================================================
# Stage simulators
# =============================================================================
#
# Each `run_stage_*` function mutates a ProcessState to reflect that stage's
# allocations and frees, and returns the StageReport.

def _dataset_ram_footprint(ds: DatasetStats) -> dict[str, int]:
    """Breakdown of what an in-memory Dataset object holds."""
    # CSR components (train + eval): data (fp32) + indices (int32) + indptr (int32)
    train_nnz = int(ds.n_interactions * 0.9)   # default 10% holdout
    eval_nnz = int(ds.n_interactions * 0.1)
    train_csr = train_nnz * (FP32 + INT32) + (ds.n_users + 1) * INT32
    eval_csr = eval_nnz * (FP32 + INT32) + (ds.n_users + 1) * INT32
    # id_maps: Python dicts ~100 B per entry (keys are str or int)
    id_maps = (ds.n_users + ds.n_items) * 100
    # polars DataFrames for train/eval, ints: 3 columns × 8B × nnz (kept until
    # Dataset __init__ finishes; typically freed but RSS retains)
    train_df = train_nnz * 3 * INT64
    eval_df = eval_nnz * 3 * INT64
    # Evaluator extras (tensors like train_sparse.indices as torch.long)
    eval_tensors = train_nnz * INT64
    return {
        "train_csr": train_csr,
        "eval_csr": eval_csr,
        "id_maps": id_maps,
        "train_df": train_df,
        "eval_df": eval_df,
        "eval_tensors": eval_tensors,
    }


def _pickled_dataset_size(ds: DatasetStats) -> int:
    """What `pickle.dumps(Dataset)` produces on the wire."""
    comp = _dataset_ram_footprint(ds)
    # Polars frames typically don't survive pickling (re-serialized as arrow).
    # CSR + tensors dominate.
    return comp["train_csr"] + comp["eval_csr"] + comp["id_maps"] + comp["eval_tensors"]


# ---- Driver stages ----------------------------------------------------------

def driver_stage_baseline(driver: ProcessState) -> StageReport:
    rpt = driver.begin_stage("D0_driver_baseline")
    driver.alloc(rpt, "python+torch+polars+ray imports", CALIBRATION["driver_baseline_ram"])
    driver.end_stage(rpt)
    return rpt


def driver_stage_csv_read(driver: ProcessState, ds: DatasetStats) -> StageReport:
    rpt = driver.begin_stage("D1_csv_read")
    # Polars scan_csv → eager collect. Rough size: nnz × (3 cols × 8B avg).
    raw_df = ds.n_interactions * 24
    driver.alloc(rpt, "polars DataFrame (raw)", raw_df)
    driver.end_stage(rpt)
    return rpt


def driver_stage_split(driver: ProcessState, ds: DatasetStats) -> StageReport:
    rpt = driver.begin_stage("D2_split_and_filter")
    raw_df = ds.n_interactions * 24
    # Polars keeps working copies during k-core iterations + split operations.
    working = int(raw_df * (CALIBRATION["polars_peak_multiplier"] - 1.0))
    driver.alloc(rpt, "polars working copies (k-core + split)", working)
    # After split, we keep train_df + eval_df; drop the original raw.
    train_df_bytes = int(raw_df * 0.9)
    eval_df_bytes = int(raw_df * 0.1)
    # Logical frees (raw + working), retained by allocator via fragmentation.
    driver.free(rpt, "raw DataFrame + working copies", raw_df + working)
    driver.alloc(rpt, "train DataFrame (post-split)", train_df_bytes)
    driver.alloc(rpt, "eval DataFrame (post-split)", eval_df_bytes)
    driver.end_stage(rpt)
    return rpt


def driver_stage_dataset_construct(driver: ProcessState, ds: DatasetStats) -> StageReport:
    rpt = driver.begin_stage("D3_dataset_construct")
    comp = _dataset_ram_footprint(ds)
    driver.alloc(rpt, "train CSR (data+indices+indptr)", comp["train_csr"])
    driver.alloc(rpt, "eval CSR", comp["eval_csr"])
    driver.alloc(rpt, "id_maps (user + item Python dicts)", comp["id_maps"])
    # COO→CSR transient workspace
    driver.alloc(rpt, "COO intermediate (transient)", comp["train_csr"])
    driver.free(rpt, "COO intermediate freed", comp["train_csr"])
    driver.end_stage(rpt)
    return rpt


def driver_stage_evaluator_init(driver: ProcessState, ds: DatasetStats) -> StageReport:
    rpt = driver.begin_stage("D4_evaluator_init")
    comp = _dataset_ram_footprint(ds)
    # Evaluator holds a reference to the shared CSR (0 bytes extra) but
    # materializes item_cluster / feature_lookup tensors and internal
    # torchmetric state.
    overhead = int((comp["train_csr"] + comp["eval_csr"]) * CALIBRATION["evaluator_overhead_frac"])
    driver.alloc(rpt, "evaluator overhead (tensors, metric state)", overhead)
    # eval indices converted to torch.long (int64)
    driver.alloc(rpt, "train_sparse.indices → torch.long", comp["eval_tensors"])
    driver.end_stage(rpt)
    return rpt


def driver_stage_dataset_preparation(driver: ProcessState, ds: DatasetStats) -> StageReport:
    rpt = driver.begin_stage("D5_dataset_preparation")
    # get_evaluation_dataloader(): EvaluationDataLoader wraps eval CSR, no
    # significant allocation. Sampled path would add num_negatives×n_users
    # int64 — we detect that from strategy in the caller.
    driver.alloc(rpt, "EvaluationDataLoader overhead", 1 * MIB)
    driver.end_stage(rpt)
    return rpt


def driver_stage_ray_put(driver: ProcessState, ds: DatasetStats) -> StageReport:
    rpt = driver.begin_stage("D6_ray_put")
    pickled = _pickled_dataset_size(ds)
    # Peak during pickling: buffer ≈ pickled size held in RAM then flushed
    # to the plasma store.
    driver.alloc(rpt, "pickle buffer (transient)", pickled)
    driver.free(rpt, "pickle buffer freed after plasma put", pickled)
    driver.end_stage(rpt)
    return rpt


# ---- Worker stages ----------------------------------------------------------

def worker_stage_baseline(worker: ProcessState, use_gpu: bool) -> StageReport:
    rpt = worker.begin_stage("W0_worker_baseline")
    worker.alloc(rpt, "python+torch+warprec imports (worker)",
                 CALIBRATION["worker_baseline_ram"])
    if use_gpu:
        worker.alloc(rpt, "CUDA runtime (host-side context)",
                     CALIBRATION["cuda_context_ram"])
    # NB: we do NOT add anything to VRAM here — the CUDA context reservation
    # is invisible to `torch.cuda.max_memory_allocated()`. VRAM accumulates
    # only through explicit torch allocations in later stages.
    worker.end_stage(rpt)
    return rpt


def worker_stage_dataset_recv(worker: ProcessState, ds: DatasetStats) -> StageReport:
    rpt = worker.begin_stage("W1_dataset_deserialize")
    pickled = _pickled_dataset_size(ds)
    inflated = int(pickled * CALIBRATION["dataset_deserialize_inflation"])
    worker.alloc(rpt, "plasma shared-memory mapping (mmap)",
                 CALIBRATION["ray_objectstore_overhead"])
    worker.alloc(rpt, "pickle.loads Dataset → Python objects", inflated)
    worker.end_stage(rpt)
    return rpt


def worker_stage_evaluator_init(worker: ProcessState, ds: DatasetStats) -> StageReport:
    rpt = worker.begin_stage("W2_evaluator_init (worker)")
    comp = _dataset_ram_footprint(ds)
    overhead = int((comp["train_csr"] + comp["eval_csr"]) * CALIBRATION["evaluator_overhead_frac"])
    worker.alloc(rpt, "evaluator overhead (worker copy)", overhead)
    worker.alloc(rpt, "train_sparse.indices → torch.long (worker copy)",
                 comp["eval_tensors"])
    worker.end_stage(rpt)
    return rpt


def worker_stage_model_init(worker: ProcessState, ds: DatasetStats,
                            hp: dict, mm: ModelMemory, use_gpu: bool) -> StageReport:
    rpt = worker.begin_stage("W3_model_instantiation")
    if mm.closed_form:
        # Transient fit peak (sim matrix + zeros_like / Gram + LU workspace).
        transient = mm.fit_peak_extra - mm.params_ram
        if transient > 0:
            worker.alloc(rpt, "fit-time transient (sim/Gram + workspace)", transient)
        worker.alloc(rpt, "model.params (numpy, persistent)", mm.params_ram)
        if transient > 0:
            worker.free(rpt, "fit-time transient freed", transient)
    else:
        # For iterative models, fit_peak_extra captures transient RAM during
        # __init__ (e.g. LightGCN building edge_index via numpy before moving
        # the SparseTensor to GPU).
        if mm.fit_peak_extra > 0:
            worker.alloc(rpt, "__init__ transient (adjacency construction)",
                         mm.fit_peak_extra)
            worker.free(rpt, "__init__ transient freed", mm.fit_peak_extra)
        if use_gpu:
            # nn.Module is constructed on CPU then `.to(device)` moves params
            # to GPU. We model the net effect — params end up on VRAM with no
            # lasting RAM footprint.
            worker.alloc(rpt, "model.parameters (VRAM)", mm.params_vram, device="vram")
            if mm.buffers_vram > 0:
                worker.alloc(rpt, "model.buffers (VRAM, non-learnable)",
                             mm.buffers_vram, device="vram")
        else:
            worker.alloc(rpt, "model.parameters (RAM, cpu mode)", mm.params_vram)
            if mm.buffers_vram > 0:
                worker.alloc(rpt, "model.buffers (RAM, cpu mode)", mm.buffers_vram)
    worker.end_stage(rpt)
    return rpt


def worker_stage_optimizer(worker: ProcessState, mm: ModelMemory, use_gpu: bool) -> StageReport:
    rpt = worker.begin_stage("W4_optimizer_state")
    if mm.closed_form or not mm.has_optimizer:
        worker.end_stage(rpt)
        return rpt
    opt_bytes = mm.optimizer_state_bytes()
    grad_bytes = mm.grad_bytes()
    dev = "vram" if use_gpu else "ram"
    worker.alloc(rpt, "Adam state (m, v buffers)", opt_bytes, device=dev)
    worker.alloc(rpt, "gradient buffer", grad_bytes, device=dev)
    worker.end_stage(rpt)
    return rpt


def worker_stage_train_loop(worker: ProcessState, mm: ModelMemory, use_gpu: bool) -> StageReport:
    rpt = worker.begin_stage("W5_train_step_peak")
    if mm.closed_form:
        worker.end_stage(rpt)
        return rpt
    dev = "vram" if use_gpu else "ram"
    # Non-structural RAM overhead: pin_memory DMA pool, dataloader-worker COW
    # pages, pre-materialized pointwise sample tensors, BLAS scratch, etc.
    worker.alloc(rpt, "iterative training RAM overhead (pin_memory + workers + BLAS)",
                 CALIBRATION["iterative_train_ram_overhead"])
    worker.alloc(rpt, "forward+backward activations (per batch)",
                 mm.train_vram_peak, device=dev)
    worker.free(rpt, "activations freed after step", mm.train_vram_peak, device=dev)
    worker.end_stage(rpt)
    return rpt


def worker_stage_eval_loop(worker: ProcessState, mm: ModelMemory, use_gpu: bool,
                            ds: DatasetStats, hp: dict) -> StageReport:
    rpt = worker.begin_stage("W6_evaluation_peak")
    dev = "vram" if use_gpu else "ram"
    eval_bs = int(hp.get("eval_batch_size", 1024))
    strategy = hp.get("strategy", "full")
    # cuBLAS workspace (appears on first cuda matmul for both closed-form
    # predict().to(device) and iterative forward).
    if use_gpu:
        worker.alloc(rpt, "cuBLAS/cuDNN workspace",
                     CALIBRATION["cuda_workspace_vram"], device="vram")
    # Model-specific lazy eval-time RAM (e.g. SASRec's _cached_user_histories)
    if mm.eval_ram_extra > 0:
        worker.alloc(rpt, "model-specific eval-time RAM", mm.eval_ram_extra)

    if mm.closed_form:
        # predict() = X_batch (CSR) @ B (numpy). Output dtype follows B's.
        # ItemKNN: FP32. EASE: FP64. ADMMSlim: FP32. Use mm.dtype_bytes.
        cpu_dtype = mm.dtype_bytes
        preds_ram = eval_bs * ds.n_items * cpu_dtype
        worker.alloc(rpt, f"predict(): X@B dense output (CPU, {cpu_dtype}B/elt)", preds_ram)
        if use_gpu:
            # Concurrent tensors at metrics peak:
            #   predictions (device-side copy, same dtype as source)
            #   ground_truth (from eval dataloader, FP32)
            #   binary_relevance (FP32 from eval_batch)
            #   discounted_relevance (FP32)
            pred_vram = eval_bs * ds.n_items * cpu_dtype
            other_count = CALIBRATION["closed_form_eval_concurrent_tensors"] - 1
            other_vram = other_count * eval_bs * ds.n_items * FP32
            eval_peak_vram = pred_vram + other_vram
            worker.alloc(rpt, f"evaluator: predictions ({cpu_dtype}B) + {other_count}×FP32 tensors",
                         eval_peak_vram, device="vram")
            worker.free(rpt, "evaluator tensors freed after metric update",
                        eval_peak_vram, device="vram")
        worker.free(rpt, "predict(CPU) transient freed", preds_ram)
    else:
        worker.alloc(rpt, f"predict peak ({strategy} strategy)",
                     mm.eval_vram_peak, device=dev)
        worker.free(rpt, "predict transient freed", mm.eval_vram_peak, device=dev)
    worker.end_stage(rpt)
    return rpt


def worker_stage_checkpoint(worker: ProcessState, mm: ModelMemory) -> StageReport:
    rpt = worker.begin_stage("W7_checkpoint_save")
    # For iterative models whose params live on VRAM, torch.save streams
    # CUDA tensors directly to disk without materializing a full RAM buffer —
    # the saved peak is not reflected in RSS. Skip.
    # For closed-form models, item_similarity is a numpy array that DOES get
    # pickled via a RAM buffer during torch.save.
    if mm.closed_form:
        worker.alloc(rpt, "torch.save pickle buffer (numpy item_similarity)",
                     mm.params_ram)
        worker.free(rpt, "pickle buffer freed", mm.params_ram)
    worker.end_stage(rpt)
    return rpt


# =============================================================================
# Top-level orchestration
# =============================================================================

@dataclass
class TrialEstimate:
    model: str
    trial_id: str
    hp: dict
    dataset: DatasetStats
    model_memory: ModelMemory
    driver_stages: list[StageReport]
    worker_stages: list[StageReport]
    driver_ram_peak: int
    worker_ram_peak: int
    worker_vram_peak: int

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "trial_id": self.trial_id,
            "hp": self.hp,
            "dataset": asdict(self.dataset),
            "driver_ram_peak_bytes": self.driver_ram_peak,
            "worker_ram_peak_bytes": self.worker_ram_peak,
            "worker_vram_peak_bytes": self.worker_vram_peak,
            "driver_stages": [
                {"stage": s.stage,
                 "ram_peak_after": s.ram_peak_after,
                 "vram_peak_after": s.vram_peak_after,
                 "events": [asdict(e) for e in s.events]}
                for s in self.driver_stages],
            "worker_stages": [
                {"stage": s.stage,
                 "ram_peak_after": s.ram_peak_after,
                 "vram_peak_after": s.vram_peak_after,
                 "events": [asdict(e) for e in s.events]}
                for s in self.worker_stages],
            "notes": self.model_memory.notes,
        }


def _enumerate_trials(model_hp: dict) -> Iterable[dict]:
    """Expand a single model's hyperparameter grid (lists → product)."""
    if model_hp is None:
        yield {}
        return
    keys = list(model_hp.keys())
    value_lists = []
    for k in keys:
        v = model_hp[k]
        # By convention, outer lists are always HPO grids. For list-valued
        # hyperparameters (e.g., mlp_hidden_size=[64,32]), the config wraps
        # them in a second bracket layer: mlp_hidden_size=[[64,32]].
        if isinstance(v, list):
            value_lists.append(list(v))
        else:
            value_lists.append([v])
    for combo in itertools.product(*value_lists):
        yield {k: v for k, v in zip(keys, combo)}


def _flatten_hp(hp: dict) -> dict:
    """Hoist common nested fields (optimization.*) into top-level keys."""
    flat = dict(hp)
    for nest_key in ("optimization", "meta"):
        if isinstance(flat.get(nest_key), dict):
            for k, v in flat[nest_key].items():
                flat.setdefault(k, v)
    return flat


def estimate_trial(model: str, raw_hp: dict, ds: DatasetStats, cfg: dict) -> TrialEstimate:
    hp = _flatten_hp(raw_hp)
    general = cfg.get("general") or {}
    device = (general.get("device") or "cuda").lower()
    use_gpu = device != "cpu"

    estimator = MODEL_ESTIMATORS.get(model)
    if estimator is None:
        warnings.warn(f"[UNSUPPORTED_MODEL] '{model}' — embedding-only fallback")
        d = int(hp.get("embedding_size", 64))
        params = _embedding(ds.n_users, d) + _embedding(_pad(ds.n_items), d)
        mm = ModelMemory(name=model, params_vram=params,
                         notes=[f"[UNSUPPORTED] fallback: embedding-only params"])
    else:
        mm = estimator(ds, hp)

    # Driver simulation (shared across trials, but we re-run per trial for
    # clarity; peaks don't depend on the HP).
    driver = ProcessState("driver")
    d_stages = [
        driver_stage_baseline(driver),
        driver_stage_csv_read(driver, ds),
        driver_stage_split(driver, ds),
        driver_stage_dataset_construct(driver, ds),
        driver_stage_evaluator_init(driver, ds),
        driver_stage_dataset_preparation(driver, ds),
        driver_stage_ray_put(driver, ds),
    ]
    driver.apply_fragmentation(CALIBRATION["fragmentation_retained_frac"])

    # Worker simulation (per trial).
    worker = ProcessState("worker")
    w_stages = [
        worker_stage_baseline(worker, use_gpu),
        worker_stage_dataset_recv(worker, ds),
        worker_stage_evaluator_init(worker, ds),
        worker_stage_model_init(worker, ds, hp, mm, use_gpu),
        worker_stage_optimizer(worker, mm, use_gpu),
        worker_stage_train_loop(worker, mm, use_gpu),
        worker_stage_eval_loop(worker, mm, use_gpu, ds, hp),
        worker_stage_checkpoint(worker, mm),
    ]
    worker.apply_fragmentation(CALIBRATION["fragmentation_retained_frac"])

    trial_id = "_".join(f"{k}={v}" for k, v in sorted(hp.items())
                        if isinstance(v, (int, float, str, bool)))
    return TrialEstimate(
        model=model, trial_id=trial_id or "default", hp=hp, dataset=ds,
        model_memory=mm,
        driver_stages=d_stages, worker_stages=w_stages,
        driver_ram_peak=driver.ram_peak,
        worker_ram_peak=worker.ram_peak,
        worker_vram_peak=worker.vram_peak,
    )


# =============================================================================
# Output formatting
# =============================================================================

def _fmt_stage_row(s: StageReport, fmt_width: int = 38) -> str:
    ram = _human(s.ram_peak_after)
    vram = _human(s.vram_peak_after)
    return f"  {s.stage:<{fmt_width}} RAM peak: {ram:>12}   VRAM peak: {vram:>10}"


def format_trial(est: TrialEstimate) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append(f"Trial: {est.model}  [{est.trial_id}]")
    lines.append("=" * 96)
    lines.append(f"Dataset: {est.dataset.n_users:,} users × "
                 f"{est.dataset.n_items:,} items × "
                 f"{est.dataset.n_interactions:,} interactions  ({est.dataset.source})")
    lines.append("")
    lines.append("─ Driver process (shared across trials) " + "─" * 55)
    for s in est.driver_stages:
        lines.append(_fmt_stage_row(s))
    lines.append(f"  {'Driver peak (RSS watermark)':<38} RAM peak: "
                 f"{_human(est.driver_ram_peak):>12}")
    lines.append("")
    lines.append("─ Ray worker process (per-trial) " + "─" * 62)
    for s in est.worker_stages:
        lines.append(_fmt_stage_row(s))
    lines.append(f"  {'Worker peak (RSS watermark)':<38} RAM peak: "
                 f"{_human(est.worker_ram_peak):>12}   "
                 f"VRAM peak: {_human(est.worker_vram_peak):>10}")
    if est.model_memory.notes:
        lines.append("")
        lines.append("  Notes:")
        for n in est.model_memory.notes:
            lines.append(f"    · {n}")
    lines.append("")
    return "\n".join(lines)


def format_summary(estimates: list[TrialEstimate]) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append("HPO summary")
    lines.append("=" * 96)
    lines.append(f"{'Model':<15}{'Trial':<42}{'RAM worker':>14}{'VRAM worker':>16}")
    lines.append("─" * 96)
    for est in estimates:
        lines.append(f"{est.model:<15}{est.trial_id[:40]:<42}"
                     f"{_human(est.worker_ram_peak):>14}"
                     f"{_human(est.worker_vram_peak):>16}")
    lines.append("─" * 96)
    peak_ram = max(e.worker_ram_peak for e in estimates)
    peak_vram = max(e.worker_vram_peak for e in estimates)
    driver_ram = max(e.driver_ram_peak for e in estimates)
    lines.append(f"Worker peak across HPO: RAM={_human(peak_ram)}   VRAM={_human(peak_vram)}")
    lines.append(f"Driver peak (one-shot): RAM={_human(driver_ram)}")
    lines.append("=" * 96)
    return "\n".join(lines)


# =============================================================================
# Validation: compare estimates to actual ray_results/*/result.json
# =============================================================================

def _find_results(cfg: dict, config_path: Path) -> list[Path]:
    """Locate ray_results/**/result.json for this experiment."""
    writer = cfg.get("writer") or {}
    dataset_name = writer.get("dataset_name", "")
    base = writer.get("local_experiment_path", "")
    # `warprec.run` is launched from the directory *containing* `config/`, not
    # from `config/` itself — so paths like `experiments/…` in the YAML are
    # relative to `config_path.parent.parent`.
    run_dir = config_path.parent.parent
    candidates = [
        run_dir / base / dataset_name / "ray_results",
        config_path.parent / base / dataset_name / "ray_results",
        run_dir / "experiments" / "warprec-benchmark-2026" / dataset_name / "ray_results",
    ]
    for cand in candidates:
        if cand.exists():
            return sorted(cand.rglob("result.json"))
    return []


def _load_last_result(path: Path) -> Optional[dict]:
    """result.json is JSONL (one record per training_iteration). Take last."""
    last = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    return last


def validate_against_results(cfg: dict, config_path: Path,
                              estimates: list[TrialEstimate]) -> str:
    result_files = _find_results(cfg, config_path)
    if not result_files:
        return "(no ray_results found — skipping validation)"

    actuals = []
    for rf in result_files:
        d = _load_last_result(rf)
        if not d or d.get("ram_peak_mb") is None:
            continue
        actuals.append({
            "trial_id": d.get("trial_id"),
            "config": d.get("config") or {},
            "ram_mb": float(d["ram_peak_mb"]),
            "vram_mb": float(d.get("vram_peak_mb", 0.0)),
        })

    if not actuals:
        return "(ray_results present but no trials completed — skipping)"

    def _as_num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return x

    def match(est: TrialEstimate) -> Optional[dict]:
        for a in actuals:
            # Compare only on keys present in actual['config'] (which holds
            # the HP grid values reported by Ray Tune). Normalize scalars so
            # 100 matches 100.0, and collections (lists) compare as-is.
            cfg_match = True
            for k, v in a["config"].items():
                if k not in est.hp:
                    cfg_match = False
                    break
                ev = est.hp[k]
                if isinstance(v, list) or isinstance(ev, list):
                    if list(v) != list(ev):
                        cfg_match = False
                        break
                elif _as_num(v) != _as_num(ev):
                    cfg_match = False
                    break
            if cfg_match:
                return a
        return None

    lines = []
    lines.append("=" * 96)
    lines.append("Validation vs actual ray_results")
    lines.append("=" * 96)
    header = f"{'Model':<10}{'Config':<36}{'RAM est':>12}{'RAM act':>12}{'Δ%':>8}   {'VRAM est':>12}{'VRAM act':>12}{'Δ%':>8}"
    lines.append(header)
    lines.append("─" * 96)

    abs_errs_ram, abs_errs_vram = [], []
    for est in estimates:
        a = match(est)
        if a is None:
            continue
        ram_est = est.worker_ram_peak / MIB
        vram_est = est.worker_vram_peak / MIB
        d_ram = (ram_est - a["ram_mb"]) / a["ram_mb"] * 100
        d_vram = ((vram_est - a["vram_mb"]) / a["vram_mb"] * 100
                  if a["vram_mb"] > 0 else float("nan"))
        abs_errs_ram.append(abs(d_ram))
        if not math.isnan(d_vram):
            abs_errs_vram.append(abs(d_vram))
        cfg_str = ",".join(f"{k}={v}" for k, v in est.hp.items()
                           if k in a["config"])[:34]
        lines.append(
            f"{est.model:<10}{cfg_str:<36}"
            f"{ram_est:>14.4f}MB{a['ram_mb']:>14.4f}MB{d_ram:>+9.4f}%   "
            f"{vram_est:>14.4f}MB{a['vram_mb']:>14.4f}MB"
            f"{(f'{d_vram:>+9.4f}%' if not math.isnan(d_vram) else '      n/a'):>10}"
        )
    lines.append("─" * 96)
    if abs_errs_ram:
        lines.append(f"Mean |Δ RAM|  = {sum(abs_errs_ram)/len(abs_errs_ram):.4f}%  "
                     f"(n={len(abs_errs_ram)})")
    if abs_errs_vram:
        lines.append(f"Mean |Δ VRAM| = {sum(abs_errs_vram)/len(abs_errs_vram):.4f}%  "
                     f"(n={len(abs_errs_vram)})")
    lines.append("=" * 96)
    return "\n".join(lines)


# =============================================================================
# CLI entry point
# =============================================================================

def run(config_path: Path, output_format: str = "table",
        validate: bool = False) -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds = _resolve_dataset_stats(cfg, config_path)

    # Propagate a couple of pipeline-level settings into HPs for convenience
    # (evaluator's batch_size default from Dataset, evaluation strategy).
    strategy = (cfg.get("evaluation") or {}).get("strategy", "full")
    models_cfg = cfg.get("models") or {}
    if not models_cfg:
        raise RuntimeError("Config has no 'models' section.")

    estimates: list[TrialEstimate] = []
    for model_name, model_hp in models_cfg.items():
        if model_hp is None:
            model_hp = {}
        # Inject global pipeline settings that affect memory
        injected = dict(model_hp)
        injected.setdefault("eval_batch_size", 1024)   # Dataset default
        injected.setdefault("strategy", strategy)
        for trial_hp in _enumerate_trials(injected):
            est = estimate_trial(model_name, trial_hp, ds, cfg)
            estimates.append(est)

    if output_format == "json":
        return json.dumps({
            "config": str(config_path),
            "dataset": asdict(ds),
            "trials": [e.to_dict() for e in estimates],
        }, indent=2)

    out = [format_trial(e) for e in estimates]
    out.append(format_summary(estimates))
    if validate:
        out.append(validate_against_results(cfg, config_path, estimates))
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="warprec-memory-estimator",
        description="Stage-by-stage RAM/VRAM estimator for WarpRec HPO pipelines.",
    )
    ap.add_argument("-c", "--config", type=Path, required=True,
                    help="Path to a WarpRec YAML config.")
    ap.add_argument("--format", choices=("table", "json"), default="table")
    ap.add_argument("--validate", action="store_true",
                    help="Compare to actual ray_results/*/result.json if present.")
    args = ap.parse_args(argv)
    print(run(args.config, args.format, args.validate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
