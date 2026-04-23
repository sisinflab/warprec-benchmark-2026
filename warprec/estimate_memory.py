"""A-priori RAM/VRAM estimator for WarpRec configurations.

Usage:
    python -m experiments.scripts.estimate_memory -c path/to/config.yml
    python -m experiments.scripts.estimate_memory -c path/to/config.yml --format json
    python -m experiments.scripts.estimate_memory --self-test

Parses a WarpRec YAML config, resolves dataset statistics (polars scan or registry),
enumerates all trial configurations implied by HPO search spaces, and prints a
per-trial breakdown of the RAM/VRAM the model + pipeline is expected to consume.

The estimator has NO runtime dependency on WarpRec for the estimation path: it
reads the config as plain YAML and applies closed-form formulas. The --self-test
mode imports WarpRec to cross-check parameter counts against actual nn.Module
instantiations.
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
from typing import Any, Callable, Iterable

try:
    import yaml
except ImportError as exc:
    print("ERROR: PyYAML required ('pip install pyyaml').", file=sys.stderr)
    raise SystemExit(1) from exc


# -----------------------------------------------------------------------------
# Unit constants
# -----------------------------------------------------------------------------

FP32 = 4
FP16 = 2
INT64 = 8
INT32 = 4

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB


def _human(nbytes: float) -> str:
    if nbytes < KIB:
        return f"{nbytes:.0f} B"
    if nbytes < MIB:
        return f"{nbytes / KIB:.4f} KiB"
    if nbytes < GIB:
        return f"{nbytes / MIB:.4f} MiB"
    return f"{nbytes / GIB:.4f} GiB"


# -----------------------------------------------------------------------------
# Dataset stats registry — used only as a fallback when data files are absent.
# Numbers come from config comments and the LightGCN / NGCF papers (the canonical
# splits WarpRec ships with).
# -----------------------------------------------------------------------------

@dataclass
class DatasetStats:
    n_users: int
    n_items: int
    n_interactions: int
    source: str  # where the numbers came from (for transparency)


_KNOWN_DATASETS: dict[str, DatasetStats] = {
    # From experiments/configs/amazon-book/backbone.yml comment + LightGCN paper
    "amazon-book": DatasetStats(52_643, 91_599, 2_380_730 + 603_378, "LightGCN paper"),
    # From LightGCN paper (standard 80/20 split after k=10 filtering)
    "yelp2018": DatasetStats(31_668, 38_048, 1_237_259 + 309_315, "LightGCN paper"),
    # From LightGCN paper (standard Gowalla split)
    "gowalla": DatasetStats(29_858, 40_981, 810_128 + 217_242, "LightGCN paper"),
    # Coat rating dataset (used as a small sanity-check dataset)
    "coat": DatasetStats(290, 300, 6_960, "Coat paper (Schnabel et al., 2016)"),
}


# -----------------------------------------------------------------------------
# ModelBreakdown — per-trial structural summary
# -----------------------------------------------------------------------------

@dataclass
class ModelBreakdown:
    model: str
    params_bytes: int = 0
    persistent_buffers_bytes: int = 0
    fwd_activation_bytes: int = 0
    notes: list[str] = field(default_factory=list)
    closed_form: bool = False  # True for models fit analytically (EASE, KNN, ADMMSlim, …)

    def trainable_bytes(self) -> int:
        """Bytes subject to gradients / optimizer state (params only, not buffers).

        Returns 0 for closed-form models: there are no gradients or optimizer
        state to allocate. `params_bytes` still counts the final artifact kept
        in memory (e.g., EASE's dense similarity matrix).
        """
        if self.closed_form:
            return 0
        return self.params_bytes

    def gpu_resident_bytes(self) -> int:
        """VRAM resident outside of grad/optimizer (weights + buffers)."""
        return self.params_bytes + self.persistent_buffers_bytes


@dataclass
class DataPrepEstimate:
    """Resources needed by `warprec.pipelines.remotes.data.remote_data_preparation`.

    Data prep runs once per train / swarm pipeline invocation on a Ray worker.
    It is purely CPU + RAM (polars read → k-core filter → split → CSR build +
    evaluation dataloader construction). No GPU is used.
    """

    cpu: int
    ram_bytes: int
    vram_bytes: int  # always 0 — included only to surface the config field explicitly
    notes: list[str] = field(default_factory=list)


@dataclass
class PipelineEstimate:
    """Full RAM/VRAM estimate for one trial configuration."""

    model: str
    trial_id: str
    hyperparameters: dict
    dataset: DatasetStats
    breakdown: ModelBreakdown
    optimizer: str
    optimizer_state_bytes: int
    gradient_bytes: int
    activation_peak_bytes: int
    vram_total_bytes: int
    ram_total_bytes: int

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "trial_id": self.trial_id,
            "hyperparameters": self.hyperparameters,
            "dataset": asdict(self.dataset),
            "optimizer": self.optimizer,
            "bytes": {
                "params": self.breakdown.params_bytes,
                "persistent_buffers": self.breakdown.persistent_buffers_bytes,
                "fwd_activation": self.breakdown.fwd_activation_bytes,
                "gradient": self.gradient_bytes,
                "optimizer_state": self.optimizer_state_bytes,
                "activation_peak": self.activation_peak_bytes,
                "vram_total": self.vram_total_bytes,
                "ram_total": self.ram_total_bytes,
            },
            "notes": self.breakdown.notes,
        }


# -----------------------------------------------------------------------------
# Structural helpers — shared building blocks across paradigms
# -----------------------------------------------------------------------------

def _embedding_table(rows: int, dim: int, dtype_bytes: int = FP32) -> int:
    return int(rows) * int(dim) * dtype_bytes


def _mlp_params(dims: list[int], with_bn: bool = False, dtype_bytes: int = FP32) -> int:
    """nn.Linear(in_i, out_i) = in*out + out ; BatchNorm1d(out) = 2*out params."""
    total = 0
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        total += in_dim * out_dim + out_dim  # weight + bias
        if with_bn:
            total += 2 * out_dim  # BN gamma + beta
    return total * dtype_bytes


def _mlp_bn_buffers(dims: list[int]) -> int:
    """BatchNorm1d non-learnable buffers: running_mean (N×fp32) + running_var (N×fp32)
    + num_batches_tracked (int64) per layer after Linear."""
    total = 0
    for out_dim in dims[1:]:
        total += 2 * out_dim * FP32 + INT64
    return total


def _transformer_encoder_params(
    d: int, inner: int, heads: int, n_layers: int, dtype_bytes: int = FP32
) -> int:
    """nn.TransformerEncoderLayer with batch_first=True, norm_first=False.

    Per-layer:
      - MultiheadAttention: (4 × d²) + (4 × d) for Q/K/V/out projections
      - FFN: Linear(d → inner) + Linear(inner → d) = 2 × (d*inner + (inner + d))
      - 2 × LayerNorm: 2 × (2 × d)
    """
    per_layer_params = (4 * d * d + 4 * d) + (2 * d * inner + inner + d) + (4 * d)
    return per_layer_params * n_layers * dtype_bytes


def _gru_params(input_size: int, hidden: int, n_layers: int, dtype_bytes: int = FP32) -> int:
    """nn.GRU params: per layer, 3 × (input_size*hidden + hidden*hidden + 2*hidden)
    Input size for layer 0 = input_size, for layer > 0 = hidden."""
    total = 0
    for layer in range(n_layers):
        in_size = input_size if layer == 0 else hidden
        total += 3 * (in_size * hidden + hidden * hidden + 2 * hidden)
    return total * dtype_bytes


def _sparse_adj_bytes(n_edges_directed: int) -> int:
    """torch_sparse.SparseTensor stores coo (row, col, value) + perm.
    Bidirectional adjacency: 2 × n_interactions edges.
    row/col = int64 × 2*E ; value = fp32 × 2*E (only present if normalized).
    """
    rowcol = 2 * (2 * n_edges_directed) * INT64
    value = (2 * n_edges_directed) * FP32
    return rowcol + value


def _activation_prop_layer(n_users: int, n_items_plus_pad: int, d: int, n_layers: int) -> int:
    """Peak activation during GNN forward: for autograd, all layer outputs are kept.
    Each layer output: (n_users + n_items+1) × d × fp32.
    Plus the stacked tensor (n_layers+1) × ... for aggregation (LightGCN uses mean).
    """
    per_layer = (n_users + n_items_plus_pad) * d * FP32
    stacked = (n_layers + 1) * per_layer
    return stacked


def _full_logits_activation(batch: int, n_items: int) -> int:
    """Batch × n_items float32 logits matrix — dominates in autoencoder / full-softmax losses."""
    return batch * n_items * FP32


def _xavier_2d_weight(in_dim: int, out_dim: int, dtype_bytes: int = FP32) -> int:
    return in_dim * out_dim * dtype_bytes


# -----------------------------------------------------------------------------
# Per-model estimators
#
# Each function takes (n_users, n_items, dataset_edges, hp: dict, dtype_bytes)
# and returns a ModelBreakdown.
#
# Rationale: formulas are derived from reading each model's __init__ in
# experiments/warprec/recommenders/**.  Notes indicate the source lines.
# -----------------------------------------------------------------------------

def _pad(n_items: int) -> int:
    """Most WarpRec item embeddings have +1 padding row (padding_idx=n_items)."""
    return n_items + 1


# ---------- Latent-factor paradigm -----------------------------------------

def _est_bpr(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    batch = int(hp["batch_size"])
    params = _embedding_table(n_users, d, dtype) + _embedding_table(_pad(n_items), d, dtype)
    # Activation: (batch, d) × 3 (user, pos, neg) + scalar dot products
    fwd = 3 * batch * d * dtype
    return ModelBreakdown("BPR", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=["bpr.py: user_embedding + item_embedding (with pad)"])


def _est_fism(n_users, n_items, n_edges, hp, dtype):
    # FISM: item-to-item factorization with two item embedding tables.
    d = int(hp["embedding_size"])
    batch = int(hp["batch_size"])
    # Two item embedding tables (history + target)
    params = 2 * _embedding_table(_pad(n_items), d, dtype)
    # Activation: (batch, max_history_len, d) — approx max_history ≈ n_edges/n_users
    max_hist = min(int(hp.get("max_history_len", max(1, n_edges // max(1, n_users)))), 200)
    fwd = batch * max_hist * d * dtype + batch * d * dtype
    return ModelBreakdown("FISM", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"FISM: 2 item tables, est. max_history≈{max_hist}"])


def _est_slim(n_users, n_items, n_edges, hp, dtype):
    # SLIM (slim.py):
    #   X = self.train_matrix.tolil()                      # LIL ~ |X| × large overhead
    #   model = ElasticNet(..., precompute=True)           # caches Gram G = X.T @ X FP64 (n_items²)
    #   for j in range(n_items):                           # per-item sequential loop
    #       model.fit(X, X[:, j].todense().getA1())        # dense (n_users,) RHS
    #       item_coeffs.append(model.sparse_coef_)
    #   self.item_similarity = sp.vstack(item_coeffs).T.todense()   # DENSE FP64 n_items²
    #
    # Final artifact is DENSE FP64 (not sparse as the 2011 paper suggests — the
    # WarpRec implementation calls .todense() on the stacked result).
    dense_fp64 = n_items * n_items * 8

    # Gram matrix cached by sklearn's precompute=True (FP64)
    gram_precompute = dense_fp64

    # LIL conversion of train_matrix: LIL uses two Python lists per row → ~5× CSR RAM
    lil_peak = 5 * (n_edges * (FP32 + INT32) + (n_users + 1) * INT32)

    # Per-iteration: current dense RHS (n_users × FP64) + internal ElasticNet state.
    # Not the dominant peak; subsumed by gram_precompute.

    return ModelBreakdown(
        "SLIM",
        params_bytes=dense_fp64,           # final item_similarity DENSE FP64
        persistent_buffers_bytes=0,
        fwd_activation_bytes=gram_precompute + lil_peak,
        closed_form=True,
        notes=[
            "SLIM: sklearn ElasticNet per-item; final item_similarity via .todense() → DENSE FP64 n_items²",
            f"Gram matrix cached by precompute=True: {_human(gram_precompute)} (FP64)",
            f"X.tolil() expansion: ~{_human(lil_peak)} (LIL overhead ~5× CSR)",
            "CAUTION: Sequential per-item ElasticNet loop — fit time O(n_items) but memory stays constant",
        ],
    )


def _est_admmslim(n_users, n_items, n_edges, hp, dtype):
    # ADMMSlim (admmslim.py):
    #   if center_columns:
    #       zero_mean_X = X.toarray() - item_means         # DENSE n_users × n_items FP32/64 (!!!)
    #       G = zero_mean_X.T @ zero_mean_X                # DENSE n_items² FP64
    #       del zero_mean_X
    #   else:
    #       G = (X.T @ X).toarray()                        # DENSE n_items² FP64
    #   diag = lambda_2*diag(item_means^alpha) + rho*I     # DENSE n_items² FP64
    #   P = np.linalg.inv(G + diag).astype(FP32)           # DENSE n_items² FP32 (+ fp64 peak in inv)
    #   B_aux = (P @ G).astype(FP32)                       # DENSE n_items² FP32
    #   Gamma = np.zeros_like(G, dtype=FP32)               # DENSE n_items² FP32
    #   C = np.zeros_like(G, dtype=FP32)                   # DENSE n_items² FP32
    #   del diag, G
    #   for _ in range(it):                                # ADMM iterations
    #       B_tilde = B_aux + P @ (rho*C - Gamma)          # DENSE n_items² FP32
    #       ...
    #   self.item_similarity = C                           # DENSE n_items² FP32
    center_columns = bool(hp.get("center_columns", False))

    dense_fp64 = n_items * n_items * 8    # G before del
    dense_fp32 = n_items * n_items * FP32

    if center_columns:
        # zero_mean_X: full dense FP32 n_users × n_items (!!!). For Netflix = 31.8 GiB.
        zero_mean_X = n_users * n_items * FP32
        # Peak during center_columns branch: zero_mean_X + G (fp64) + X CSR
        pre_inv_peak = zero_mean_X + dense_fp64
        center_note = (
            f"center_columns=True → zero_mean_X DENSE FP32 ({_human(zero_mean_X)})! "
            f"For Netflix-100M this is ~32 GiB just for the centered matrix"
        )
    else:
        # Peak during non-centered branch: sparse X.T@X (near-dense) + G densified
        pre_inv_peak = n_items * n_items * (FP32 + INT32) + dense_fp64
        center_note = "center_columns=False → sparse X.T @ X densified via .toarray()"

    # During np.linalg.inv(G + diag) both G and diag (fp64) are alive + LAPACK copy + result
    # but P is cast to fp32. Peak: G + diag + inv_copy + result_fp32 = 3×fp64 + fp32
    inv_peak = 3 * dense_fp64 + dense_fp32

    # Resident during ADMM loop after del G: P, B_aux, Gamma, C (all fp32) + B_tilde transient
    admm_resident = 4 * dense_fp32    # P + B_aux + Gamma + C
    admm_transient = 2 * dense_fp32   # B_tilde + (rho*C - Gamma) intermediate

    # Final: C (fp32 dense) — the item_similarity
    final = dense_fp32

    return ModelBreakdown(
        "ADMMSlim",
        params_bytes=final,
        persistent_buffers_bytes=admm_resident - final,  # P, B_aux, Gamma still resident with C
        fwd_activation_bytes=max(pre_inv_peak + inv_peak, admm_resident + admm_transient),
        closed_form=True,
        notes=[
            center_note,
            f"Peak during inv: G(fp64) + diag(fp64) + LAPACK copy + P(fp32) = {_human(inv_peak)}",
            f"ADMM loop: P + B_aux + Gamma + C all DENSE FP32 n_items² = {_human(admm_resident)}",
            f"Final item_similarity: {_human(final)} DENSE FP32",
        ],
    )


# ---------- Graph paradigm --------------------------------------------------

def _est_lightgcn(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    n_layers = int(hp["n_layers"])
    batch = int(hp["batch_size"])
    n_items_pad = _pad(n_items)

    params = _embedding_table(n_users, d, dtype) + _embedding_table(n_items_pad, d, dtype)
    # LGConv has NO learnable params. propagation_network is stateless.

    adj = _sparse_adj_bytes(n_edges)
    # All layer activations kept for autograd — mean over layers.
    activation = _activation_prop_layer(n_users, n_items_pad, d, n_layers)
    # Batch-dependent embeddings lookup: (batch, d) × 3
    activation += 3 * batch * d * dtype
    return ModelBreakdown("LightGCN", params_bytes=params,
                          persistent_buffers_bytes=adj,
                          fwd_activation_bytes=activation,
                          notes=[f"LGConv stateless; {n_layers} layer activations kept for autograd",
                                 f"Sparse adj (bidirectional, 2×{n_edges} edges)"])


def _est_ngcf(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    weight_size = hp.get("weight_size", [d, d, d])
    if isinstance(weight_size, int):
        weight_size = [weight_size]
    batch = int(hp["batch_size"])
    n_items_pad = _pad(n_items)

    params = _embedding_table(n_users, d, dtype) + _embedding_table(n_items_pad, d, dtype)
    # NGCFLayer: W1 (in × out), W2 (in × out), b1 (out), b2 (out)
    dims = [d] + list(weight_size)
    for in_d, out_d in zip(dims[:-1], dims[1:]):
        params += 2 * (in_d * out_d + out_d) * dtype

    adj = _sparse_adj_bytes(n_edges)
    # NGCF concatenates layer outputs → final emb width = d + sum(weight_size)
    layer_activations = sum((n_users + n_items_pad) * w * dtype for w in dims)
    activation = layer_activations + 3 * batch * sum(dims) * dtype
    return ModelBreakdown("NGCF", params_bytes=params,
                          persistent_buffers_bytes=adj,
                          fwd_activation_bytes=activation,
                          notes=[f"NGCF layers: {weight_size} (W1+W2 per layer)"])


def _est_sgl(n_users, n_items, n_edges, hp, dtype):
    # SGL = LightGCN + 2 augmented views + InfoNCE per-batch
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "SGL"
    # Contrastive loss keeps 2 additional augmented propagated views.
    d = int(hp["embedding_size"])
    n_layers = int(hp["n_layers"])
    n_items_pad = _pad(n_items)
    extra_views = 2 * _activation_prop_layer(n_users, n_items_pad, d, n_layers)
    # InfoNCE similarity (batch × batch)
    batch = int(hp["batch_size"])
    nce = 2 * batch * batch * FP32
    bd.fwd_activation_bytes += extra_views + nce
    bd.notes.append("SGL: +2 augmented views + InfoNCE [B×B] similarity")
    return bd


def _est_lightgcl(n_users, n_items, n_edges, hp, dtype):
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "LightGCL"
    # LightGCL uses SVD-based low-rank contrastive signal; rank q
    q = int(hp.get("svd_q", 5))
    n_items_pad = _pad(n_items)
    low_rank_buffer = (n_users + n_items_pad) * q * FP32 * 2  # U, V
    bd.persistent_buffers_bytes += low_rank_buffer
    batch = int(hp["batch_size"])
    bd.fwd_activation_bytes += 2 * batch * batch * FP32
    bd.notes.append(f"LightGCL: SVD rank-{q} buffers + InfoNCE")
    return bd


def _est_sgcl(n_users, n_items, n_edges, hp, dtype):
    bd = _est_sgl(n_users, n_items, n_edges, hp, dtype)
    bd.model = "SGCL"
    bd.notes[-1] = "SGCL: SGL variant with simplified contrastive head"
    return bd


def _est_ultragcn(n_users, n_items, n_edges, hp, dtype):
    """UltraGCN (Mao et al. CIKM 2021). No message passing at training —
    the graph structure enters via constraint losses (user-item + item-item).
    """
    d = int(hp["embedding_size"])
    batch = int(hp["batch_size"])
    ii_k = int(hp.get("ii_k", 10))

    # Same as BPR: user + item embeddings only (no layer-wise propagation).
    params = _embedding_table(n_users, d, dtype) + _embedding_table(_pad(n_items), d, dtype)

    # Item-item k-NN graph: n_items × k (indices + values).
    ii_buffer = n_items * ii_k * (FP32 + INT32)

    # Training-time activation: (batch, d) × 3 (user, pos, neg) + constraint scores.
    fwd = 3 * batch * d * dtype

    return ModelBreakdown(
        "UltraGCN",
        params_bytes=params,
        persistent_buffers_bytes=ii_buffer,
        fwd_activation_bytes=fwd,
        notes=[
            f"UltraGCN: user/item embeddings + item-item kNN (k={ii_k})",
            "No graph propagation at training time (direct embedding constraint loss)",
        ],
    )


def _est_dgcf(n_users, n_items, n_edges, hp, dtype):
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "DGCF"
    # DGCF disentangles into K latent factors; each factor has its own embedding path
    n_factors = int(hp.get("n_factors", 4))
    d = int(hp["embedding_size"])
    n_items_pad = _pad(n_items)
    # Factor-specific edge weights (learned): K × n_edges
    factor_weights = n_factors * (2 * n_edges) * FP32
    bd.persistent_buffers_bytes += factor_weights
    # Each layer activation scales by K
    bd.fwd_activation_bytes = (bd.fwd_activation_bytes - _activation_prop_layer(
        n_users, n_items_pad, d, int(hp["n_layers"])
    )) + n_factors * _activation_prop_layer(n_users, n_items_pad, d // max(1, n_factors),
                                             int(hp["n_layers"]))
    bd.notes.append(f"DGCF: {n_factors} disentangled factor channels")
    return bd


def _est_lightgcnpp(n_users, n_items, n_edges, hp, dtype):
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "LightGCNpp"
    d = int(hp["embedding_size"])
    # Hypernetwork: additional (d × d) weight per layer
    n_layers = int(hp["n_layers"])
    bd.params_bytes += n_layers * d * d * dtype
    bd.notes.append("LightGCN++: hypernet weights per layer")
    return bd


def _est_gcmc(n_users, n_items, n_edges, hp, dtype):
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "GCMC"
    d = int(hp["embedding_size"])
    # Bipartite GCN has a decoder mapping (d → n_ratings) + side feature encoder
    n_ratings = int(hp.get("n_ratings", 5))
    bd.params_bytes += d * n_ratings * dtype
    bd.notes.append("GCMC: bipartite decoder layer")
    return bd


def _est_lightccf(n_users, n_items, n_edges, hp, dtype):
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "LightCCF"
    bd.notes.append("LightCCF: same memory profile as LightGCN")
    return bd


def _est_egcf(n_users, n_items, n_edges, hp, dtype):
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "EGCF"
    bd.notes.append("EGCF: LightGCN backbone with explicit edge features")
    return bd


def _est_esigcf(n_users, n_items, n_edges, hp, dtype):
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "ESIGCF"
    # Signed edges → 2× adjacency storage
    bd.persistent_buffers_bytes += _sparse_adj_bytes(n_edges)
    bd.notes.append("ESIGCF: signed adjacency (2× storage)")
    return bd


def _est_lightgode(n_users, n_items, n_edges, hp, dtype):
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "LightGODE"
    # ODE solver keeps solver state per step; approximated as 3× activation
    bd.fwd_activation_bytes *= 3
    bd.notes.append("LightGODE: ODE solver keeps 3× activation for backprop through solver")
    return bd


def _est_mixrec(n_users, n_items, n_edges, hp, dtype):
    bd = _est_lightgcn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "MixRec"
    batch = int(hp["batch_size"])
    d = int(hp["embedding_size"])
    # Mixup augmentation: 2× per-batch user/item mixed embeddings
    bd.fwd_activation_bytes += 2 * batch * d * FP32
    bd.notes.append("MixRec: +mixup buffers per batch")
    return bd


def _est_xsimgcl(n_users, n_items, n_edges, hp, dtype):
    bd = _est_sgl(n_users, n_items, n_edges, hp, dtype)
    bd.model = "XSimGCL"
    bd.notes[-1] = "XSimGCL: cross-layer simple contrastive (SGL-like memory)"
    return bd


def _est_rp3beta(n_users, n_items, n_edges, hp, dtype):
    # RP3β: closed-form graph-based item similarity via 3-step random walk.
    # Pipeline (rp3beta.py):
    #   Pui = normalize(X, "l1", axis=1)                 # sparse CSR FP32 ~ |X|
    #   X_bool = X.T with ones                           # sparse CSR FP32 ~ |X|
    #   Piu = normalize(X_bool, "l1", axis=1)            # sparse CSR FP32 ~ |X|
    #   degree = np.power(item_pop, -beta)               # n_items FP32 (tiny)
    #   # Blockwise similarity accumulation in rows/cols/values int32/int32/fp32 arrays
    #   # For Netflix-scale these grow to near n_items² nnz = ~316M × (4+4+4) ≈ 3.8 GiB
    #   similarity = csr_matrix((values, (rows, cols)), ...)
    #   filtered = _apply_sparse_topk(similarity, k)     # lil_matrix (per-row python loop!)
    #   self.item_similarity = filtered.toarray()        # DENSE FP32 n_items²
    #
    # Final resident: DENSE n_items² FP32 (not sparse!).
    # Peak during fit: 3 sparse matrices (Piu/Pui/X_bool ≈ 3×|X|) + (rows, cols, values)
    # accumulator (near n_items² nnz in worst case) + lil_matrix build + toarray()
    # + dense target = at least 2× n_items² FP32 during toarray().

    X_sparse = n_edges * (FP32 + INT32) + (n_users + 1) * INT32  # Pui
    all_sparse = 3 * X_sparse                                    # Pui + Piu + X_bool

    # Worst-case accumulator arrays (rows, cols, values) for near-dense XTX-like similarity:
    # Each full n_items² triple costs (INT32 + INT32 + FP32) = 12 bytes per entry.
    # Bound by a fraction of n_items²; RP3Beta's degree weighting keeps it dense.
    accumulator = n_items * n_items * (2 * INT32 + FP32)

    # lil_matrix during topk is per-row inefficient but temporarily holds only top-k values.
    # The .toarray() at the end allocates a DENSE n_items² FP32 matrix.
    dense_final = n_items * n_items * FP32

    return ModelBreakdown(
        "RP3Beta",
        params_bytes=dense_final,                      # final item_similarity (FP32 DENSE)
        persistent_buffers_bytes=0,
        fwd_activation_bytes=all_sparse + accumulator, # peak before .toarray()
        closed_form=True,
        notes=[
            "RP3β: closed-form 3-step random walk, dense FP32 n_items² output via .toarray()",
            f"Peak fit: 3 sparse norm matrices ({_human(all_sparse)}) + dense accumulator ({_human(accumulator)})",
            "CAUTION: _apply_sparse_topk uses lil_matrix + Python loop — slow and RAM-hungry for large n_items",
        ],
    )


# ---------- Autoencoder paradigm -------------------------------------------

def _est_multivae(n_users, n_items, n_edges, hp, dtype):
    inter = int(hp["intermediate_dim"])
    latent = int(hp["latent_dim"])
    batch = int(hp["batch_size"])
    # Encoder: Linear(n_items → inter) + Linear(inter → latent) × 2 (mean + log_var)
    enc = _xavier_2d_weight(n_items, inter, dtype) + inter * dtype  # bias
    enc += 2 * (_xavier_2d_weight(inter, latent, dtype) + latent * dtype)
    # Decoder: Linear(latent → inter) + Linear(inter → n_items)
    dec = _xavier_2d_weight(latent, inter, dtype) + inter * dtype
    dec += _xavier_2d_weight(inter, n_items, dtype) + n_items * dtype
    params = enc + dec
    # Activation: (batch, n_items) input + (batch, inter) hidden + (batch, latent) × 3 + (batch, n_items) recon
    fwd = 2 * batch * n_items * FP32 + batch * inter * FP32 + 3 * batch * latent * FP32
    return ModelBreakdown("MultiVAE", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=["Encoder(n_items→inter→latent×2) + Decoder(latent→inter→n_items)"])


def _est_multidae(n_users, n_items, n_edges, hp, dtype):
    bd = _est_multivae(n_users, n_items, n_edges, hp, dtype)
    bd.model = "MultiDAE"
    # No log_var branch: subtract (inter × latent + latent) bytes
    inter = int(hp["intermediate_dim"])
    latent = int(hp["latent_dim"])
    bd.params_bytes -= (inter * latent + latent) * dtype
    bd.notes = ["MultiDAE: encoder(n_items→inter→latent) + decoder(latent→inter→n_items)"]
    return bd


def _est_cdae(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    batch = int(hp["batch_size"])
    # CDAE: user embedding (n_users × d) + encoder Linear(n_items → d) + decoder Linear(d → n_items)
    params = _embedding_table(n_users, d, dtype)
    params += _xavier_2d_weight(n_items, d, dtype) + d * dtype
    params += _xavier_2d_weight(d, n_items, dtype) + n_items * dtype
    # Activation: (batch, n_items) input + (batch, d) hidden + (batch, n_items) recon
    fwd = 2 * batch * n_items * FP32 + batch * d * FP32
    return ModelBreakdown("CDAE", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=["CDAE: user emb + encoder(n_items→d) + decoder(d→n_items)"])


def _est_ease(n_users, n_items, n_edges, hp, dtype):
    # EASE closed-form: B = -(G⁻¹ - diag) where G = X^T X + λI
    # Implementation uses np.linalg.inv → FP64. self.item_similarity stored FP64.
    # During `G = X.T @ X + l2 * np.identity(n_items)` the following are alive
    # simultaneously (measured empirically on Netflix-100M):
    #   - sparse X.T @ X CSC near-dense (~n_items² nnz, FP32 data + INT32 cols)
    #   - np.identity(n_items) FP64
    #   - l2 * np.identity (new allocation) FP64
    #   - densified G from sparse+dense addition FP64
    # Plus the training CSR X resident throughout (user's responsibility).
    # Then np.linalg.inv(G) adds ~1× FP64 for B on top of G still alive.
    #
    # Peak accounting (FP64 dominates):
    #   params_bytes         = B final resident (FP64)
    #   persistent_buffers   = G during inversion (FP64)
    #   fwd_activation       = transient peak during G = XTX + l2·I
    #                          (XTX sparse data + np.identity + l2·I copy) ~ 3×FP64
    dense_fp64 = n_items * n_items * 8
    xtx_sparse = n_items * n_items * (FP32 + INT32)  # near-dense CSC
    transient = xtx_sparse + 2 * dense_fp64          # identity + l2·I copy
    return ModelBreakdown(
        "EASE",
        params_bytes=dense_fp64,               # B final (FP64)
        persistent_buffers_bytes=dense_fp64,   # G during inv (FP64)
        fwd_activation_bytes=transient,        # XTX sparse + identity + l2·I
        closed_form=True,
        notes=[
            "EASE: closed-form, np.linalg.inv(G) in FP64",
            "Peak ≈ 4× n_items²·FP64 during G = X.T@X + l2·I (identity, l2·I copy, XTX densified, G)",
            f"item_similarity final: {_human(dense_fp64)} (FP64 dense)",
        ],
    )


def _est_elsa(n_users, n_items, n_edges, hp, dtype):
    # ELSA ≠ EASE. It's an IterativeRecommender with nn.Parameter W of shape
    # [n_items, n_dims] trained via SGD/Adam. Memory is DOMINATED by activations
    # (batch × n_items FP32, as train_batch is densified in predict/forward).
    #
    # Params: W = n_items × n_dims (FP32)
    # Activations (forward):
    #   - rating_matrix (batch, n_items): FP32 dense (densified from sparse batch)
    #   - latent = x @ A: (batch, n_dims) FP32
    #   - reconstruction = latent @ A.t(): (batch, n_items) FP32
    #   - return reconstruction - x: (batch, n_items) FP32
    # Backward saves these → 2× activation peak.
    n_dims = int(hp.get("n_dims", 256))
    batch = int(hp.get("batch_size", 1024))
    params = _embedding_table(n_items, n_dims, dtype)
    # Forward dense activations (saved for backward):
    fwd = (2 * batch * n_items + 2 * batch * n_dims) * dtype
    return ModelBreakdown(
        "ELSA",
        params_bytes=params,
        fwd_activation_bytes=fwd,
        notes=[
            f"ELSA: gradient-trained embedding W[{n_items}×{n_dims}] (FP32)",
            f"Activation dominated by dense batch×n_items densification (batch={batch})",
        ],
    )


def _est_macridvae(n_users, n_items, n_edges, hp, dtype):
    bd = _est_multivae(n_users, n_items, n_edges, hp, dtype)
    bd.model = "MacridVAE"
    # Macrid-VAE adds K prototype categories
    K = int(hp.get("kfac", 7))
    latent = int(hp["latent_dim"])
    bd.params_bytes += K * latent * FP32
    bd.notes.append(f"MacridVAE: +{K} prototype categories")
    return bd


def _est_sansa(n_users, n_items, n_edges, hp, dtype):
    # SANSA: closed-form sparse approximate inverse of G = X^T X + l2·I.
    # Pipeline (sansa.py):
    #   G = (X.T @ X).tocsc()                            # FP32 CSC near-dense
    #   G[diag, diag] += l2                              # in-place
    #   P = cholesky(G).solve_A(sp.eye(n_items))         # CHOLMOD factor + solve
    #     # fallback: P = sp.linalg.inv(G)  — sparse inverse with significant fill-in
    #   B = P.multiply(inv_diag); B.setdiag(0); B.eliminate_zeros()
    #   B = _sparsify_matrix(B, target_density)          # globally keeps top-k entries
    #   self.item_similarity = B.tocsr()                 # FINAL: sparse CSR
    #
    # Peak memory: Cholesky fill-in on X.T@X for dense datasets can be very large.
    # For Netflix-100M scale (n_items ≈ 18k, X.T@X near-fully dense), CHOLMOD's
    # L factor is essentially dense lower triangular ≈ n_items² × FP64 / 2.
    # Plus sp.eye CSC RHS (O(n_items)) and the solve output P (~near-dense).
    #
    # Final resident: sparse CSR with nnz = target_density × n_items².
    target_density = float(hp.get("target_density", 0.02))
    sparse_nnz = int(n_items * n_items * target_density)

    # CHOLMOD peak: near-dense lower-triangular L (FP64), plus G (FP32 CSC),
    # plus P output (approximately as dense as G).
    cholmod_L = n_items * n_items * 8 // 2       # dense lower triangular FP64
    G_sparse = n_items * n_items * (FP32 + INT32)  # near-dense CSC of X.T@X
    P_peak = n_items * n_items * (FP32 + INT32)    # P as sparse result of solve

    # Final item_similarity (CSR): nnz × (FP32 data + INT32 col_idx) + row_ptr
    final = sparse_nnz * (FP32 + INT32) + (n_items + 1) * INT32

    return ModelBreakdown(
        "SANSA",
        params_bytes=final,
        persistent_buffers_bytes=0,
        # Peak during fit is transient (dominated by CHOLMOD factor + G + P)
        fwd_activation_bytes=cholmod_L + G_sparse + P_peak,
        closed_form=True,
        notes=[
            f"SANSA: closed-form sparse inverse, target_density={target_density}",
            f"Final item_similarity CSR nnz={sparse_nnz:,} ({_human(final)})",
            f"Peak during CHOLMOD: ~dense triang FP64 ({_human(cholmod_L)}) + G ({_human(G_sparse)})",
            "WARNING: For dense near-full X.T@X the L factor can approach n_items²·FP64/2",
        ],
    )


# ---------- Neural CF paradigm ---------------------------------------------

def _est_neumf(n_users, n_items, n_edges, hp, dtype):
    mf_d = int(hp["mf_embedding_size"])
    mlp_d = int(hp["mlp_embedding_size"])
    mlp_hidden = hp.get("mlp_hidden_size", [64, 32, 16, 8])
    if isinstance(mlp_hidden, int):
        mlp_hidden = [mlp_hidden]
    mlp_hidden = list(mlp_hidden)
    batch = int(hp["batch_size"])
    n_items_pad = _pad(n_items)

    params = _embedding_table(n_users, mf_d, dtype) + _embedding_table(n_items_pad, mf_d, dtype)
    params += _embedding_table(n_users, mlp_d, dtype) + _embedding_table(n_items_pad, mlp_d, dtype)
    params += _mlp_params([2 * mlp_d] + mlp_hidden, with_bn=False, dtype_bytes=dtype)
    # predict_layer: Linear(mf_d + mlp_hidden[-1] → 1)
    params += ((mf_d + mlp_hidden[-1]) + 1) * dtype

    # Activation: (batch, mf_d) + (batch, 2*mlp_d) + (batch, h) for each hidden
    fwd = batch * mf_d * FP32 + batch * 2 * mlp_d * FP32
    for h in mlp_hidden:
        fwd += batch * h * FP32
    return ModelBreakdown("NeuMF", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"NeuMF: MF(d={mf_d}) + MLP(d={mlp_d}, hidden={mlp_hidden})"])


def _est_convncf(n_users, n_items, n_edges, hp, dtype):
    d = int(hp.get("embedding_size", 64))
    batch = int(hp["batch_size"])
    params = _embedding_table(n_users, d, dtype) + _embedding_table(_pad(n_items), d, dtype)
    # CNN over the outer-product map (d × d) — 4 Conv2d layers, 32 channels typical
    channels = hp.get("cnn_channels", [32, 32, 32, 32])
    if isinstance(channels, int):
        channels = [channels]
    in_ch = 1
    for out_ch in channels:
        params += (in_ch * out_ch * 4 * 4 + out_ch) * dtype  # 4x4 kernel, stride
        in_ch = out_ch
    # Final FC
    params += (channels[-1] + 1) * dtype
    # Activation: outer-product (batch, 1, d, d) + conv maps
    fwd = batch * d * d * FP32
    for c in channels:
        fwd += batch * c * d * d * FP32  # rough (ignoring pooling)
        d = max(1, d // 2)
    return ModelBreakdown("ConvNCF", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=["ConvNCF: outer-product (d×d) fed through CNN"])


# ---------- KNN paradigm ---------------------------------------------------

def _est_itemknn(n_users, n_items, n_edges, hp, dtype):
    # ItemKNN (itemknn.py):
    #   sim_matrix = torch.from_numpy(similarity.compute(X.T))  # DENSE n_items² FP32
    #   filtered = _apply_topk_filtering(sim_matrix, k)
    #     → values, indices = torch.topk(sim_matrix, k=k, dim=1)
    #     → return torch.zeros_like(sim_matrix).scatter_(1, indices, values)
    #   self.item_similarity = filtered.numpy()                 # DENSE n_items² FP32 (mostly 0)
    #
    # Despite "top-k", torch.zeros_like + scatter_ yields a DENSE n_items² FP32
    # tensor. Final item_similarity is dense, not sparse. During fit, both the
    # similarity dense and the zeros_like dense are alive simultaneously (~2×).
    k = int(hp.get("k", 50))
    dense = n_items * n_items * FP32
    # Peak during _apply_topk_filtering: sim_matrix + zeros_like output + topk values/indices
    peak_during_fit = 2 * dense + n_items * k * (FP32 + INT64)
    return ModelBreakdown(
        "ItemKNN",
        params_bytes=dense,                                # final DENSE FP32 n_items²
        persistent_buffers_bytes=0,
        fwd_activation_bytes=peak_during_fit - dense,      # transient (sim_matrix + topk bufs)
        closed_form=True,
        notes=[
            f"ItemKNN: similarity.compute(X.T).toarray() → DENSE FP32 n_items²",
            f"top-{k} filter via torch.zeros_like + scatter: final storage is DENSE, NOT sparse",
            f"Final item_similarity: {_human(dense)} dense FP32",
        ],
    )


def _est_userknn(n_users, n_items, n_edges, hp, dtype):
    # UserKNN (userknn.py):
    #   sim_matrix = torch.from_numpy(similarity.compute(X))    # DENSE n_users² FP32
    #   filtered = _apply_topk_filtering(sim_matrix, k)         # DENSE via zeros_like+scatter
    #   self.user_similarity = filtered.numpy()                 # DENSE n_users² FP32
    #
    # This scales with n_users², which is catastrophic for datasets with many users.
    # Netflix-100M: 480k users → 480k² × FP32 = 858 GiB. Cannot run.
    k = int(hp.get("k", 50))
    dense = n_users * n_users * FP32
    peak_during_fit = 2 * dense + n_users * k * (FP32 + INT64)

    notes = [
        f"UserKNN: similarity.compute(X).toarray() → DENSE FP32 n_users²",
        f"top-{k} filter via torch.zeros_like + scatter: final storage is DENSE, NOT sparse",
        f"Final user_similarity: {_human(dense)} dense FP32",
    ]
    # Hard warning when the dense user-user matrix exceeds typical VRAM / host RAM.
    if dense > 20 * GIB:
        notes.append(
            f"[!!] UserKNN on n_users={n_users:,} requires {_human(dense)} for user_similarity alone; "
            f"will almost certainly OOM on any commodity node"
        )

    return ModelBreakdown(
        "UserKNN",
        params_bytes=dense,
        persistent_buffers_bytes=0,
        fwd_activation_bytes=peak_during_fit - dense,
        closed_form=True,
        notes=notes,
    )


# ---------- Sequential paradigm --------------------------------------------

def _est_sasrec(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    n_layers = int(hp["n_layers"])
    n_heads = int(hp["n_heads"])
    inner = int(hp["inner_size"])
    max_seq_len = int(hp["max_seq_len"])
    batch = int(hp["batch_size"])
    neg = int(hp.get("neg_samples", 1))

    n_items_pad = _pad(n_items)
    params = _embedding_table(n_items_pad, d, dtype) + _embedding_table(max_seq_len, d, dtype)
    params += _transformer_encoder_params(d, inner, n_heads, n_layers, dtype)
    # LayerNorm on embedding
    params += 2 * d * dtype

    # Causal mask buffer (bool, seq_len × seq_len)
    causal_mask = max_seq_len * max_seq_len  # bool = 1 byte

    # Activation:
    #   embeddings (batch, seq, d)
    #   attention scores (batch, heads, seq, seq) per layer
    #   FFN (batch, seq, inner) per layer
    emb = batch * max_seq_len * d * FP32
    attn = batch * n_heads * max_seq_len * max_seq_len * FP32 * n_layers
    ffn = batch * max_seq_len * inner * FP32 * n_layers
    # Neg samples: (batch, neg, d)
    neg_buf = batch * neg * d * FP32
    fwd = emb + attn + ffn + neg_buf
    return ModelBreakdown("SASRec", params_bytes=params,
                          persistent_buffers_bytes=causal_mask,
                          fwd_activation_bytes=fwd,
                          notes=[f"SASRec: Transformer(layers={n_layers}, heads={n_heads}, "
                                 f"d={d}, inner={inner}, seq={max_seq_len})",
                                 f"Attention activation: B×H×L²={batch}×{n_heads}×{max_seq_len}²"])


def _est_bert4rec(n_users, n_items, n_edges, hp, dtype):
    bd = _est_sasrec(n_users, n_items, n_edges, hp, dtype)
    bd.model = "BERT4Rec"
    d = int(hp["embedding_size"])
    # BERT4Rec adds [MASK] token → item vocab is n_items + 2
    bd.params_bytes += d * dtype  # +1 mask token
    # Position embedding is +1 longer (for [CLS]-like)
    max_seq_len = int(hp["max_seq_len"])
    bd.params_bytes += d * dtype
    bd.notes = [f"BERT4Rec: +[MASK] token, bidirectional (no causal mask)",
                *bd.notes[1:]]
    return bd


def _est_gru4rec(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    gru_size = int(hp.get("gru_size", d))
    n_layers = int(hp.get("n_layers", 1))
    max_seq_len = int(hp["max_seq_len"])
    batch = int(hp["batch_size"])
    neg = int(hp.get("neg_samples", 1))

    params = _embedding_table(_pad(n_items), d, dtype)
    params += _gru_params(d, gru_size, n_layers, dtype)
    # Output projection: gru_size → d
    params += (gru_size * d + d) * dtype

    # Activation: (batch, seq, d) + (batch, seq, gru_size) + hidden (n_layers, batch, gru_size)
    fwd = batch * max_seq_len * d * FP32
    fwd += batch * max_seq_len * gru_size * FP32
    fwd += n_layers * batch * gru_size * FP32
    fwd += batch * neg * d * FP32
    return ModelBreakdown("GRU4Rec", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"GRU4Rec: GRU({d}→{gru_size}, layers={n_layers}, seq={max_seq_len})"])


def _est_caser(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    max_seq_len = int(hp["max_seq_len"])
    batch = int(hp["batch_size"])
    n_h = int(hp.get("nh", 16))  # horizontal filters
    n_v = int(hp.get("nv", 4))   # vertical filters

    params = _embedding_table(_pad(n_items), d, dtype) + _embedding_table(n_users, d, dtype)
    # Horizontal conv: n_h filters of (h, d) for h in [1..max_seq_len]
    # Approx: sum_h (n_h × h × d) ≈ n_h × max_seq_len × d (upper bound)
    params += (n_h * max_seq_len * d + n_h) * dtype
    # Vertical conv: n_v filters of (max_seq_len, 1)
    params += (n_v * max_seq_len + n_v) * dtype
    # FC layer: (n_h + n_v*d) → d
    fc_in = n_h + n_v * d
    params += (fc_in * d + d) * dtype
    # Predict layer: (d + d) → n_items  (concat user emb)
    params += (2 * d * n_items + n_items) * dtype

    fwd = batch * max_seq_len * d * FP32 + batch * fc_in * FP32 + batch * n_items * FP32
    return ModelBreakdown("Caser", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"Caser: horiz={n_h}, vert={n_v}, seq={max_seq_len}"])


def _est_narm(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    hidden = int(hp.get("hidden_size", d))
    max_seq_len = int(hp["max_seq_len"])
    batch = int(hp["batch_size"])
    n_layers = int(hp.get("n_layers", 1))
    neg = int(hp.get("neg_samples", 1))

    params = _embedding_table(_pad(n_items), d, dtype)
    params += _gru_params(d, hidden, n_layers, dtype)
    # Attention: A1 (hidden, hidden), A2 (hidden, hidden), v (hidden)
    params += (2 * hidden * hidden + hidden) * dtype
    # Output projection: 2*hidden → d
    params += (2 * hidden * d + d) * dtype

    fwd = batch * max_seq_len * d * FP32
    fwd += batch * max_seq_len * hidden * FP32
    fwd += batch * max_seq_len * FP32  # attention weights
    fwd += batch * neg * d * FP32
    return ModelBreakdown("NARM", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"NARM: GRU({d}→{hidden}) + attention, seq={max_seq_len}"])


def _est_fossil(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    max_seq_len = int(hp["max_seq_len"])
    batch = int(hp["batch_size"])
    n_order = int(hp.get("n_order", 2))

    params = _embedding_table(_pad(n_items), d, dtype)
    # User-specific temporal bias
    params += _embedding_table(n_users, d, dtype)
    # Order coefficients
    params += n_users * n_order * dtype

    fwd = batch * max_seq_len * d * FP32
    return ModelBreakdown("FOSSIL", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"FOSSIL: temporal weighting, order={n_order}"])


def _est_lightsans(n_users, n_items, n_edges, hp, dtype):
    bd = _est_sasrec(n_users, n_items, n_edges, hp, dtype)
    bd.model = "LightSANs"
    # LightSANs: low-rank attention → attention activation is B × k × L (k=n_latent)
    k = int(hp.get("k_interests", 5))
    d = int(hp["embedding_size"])
    n_layers = int(hp["n_layers"])
    max_seq_len = int(hp["max_seq_len"])
    batch = int(hp["batch_size"])
    # Replace B×H×L² attention with B×k×L (much smaller)
    n_heads = int(hp["n_heads"])
    attn_orig = batch * n_heads * max_seq_len * max_seq_len * FP32 * n_layers
    attn_new = batch * k * max_seq_len * FP32 * n_layers
    bd.fwd_activation_bytes = bd.fwd_activation_bytes - attn_orig + attn_new
    bd.notes.append(f"LightSANs: low-rank attention (k={k})")
    return bd


def _est_linrec(n_users, n_items, n_edges, hp, dtype):
    d = int(hp["embedding_size"])
    max_seq_len = int(hp["max_seq_len"])
    batch = int(hp["batch_size"])
    # Linear-attention sequential — similar param count to SASRec but linear-in-L activation
    n_layers = int(hp["n_layers"])
    params = _embedding_table(_pad(n_items), d, dtype) + _embedding_table(max_seq_len, d, dtype)
    params += _transformer_encoder_params(d, 4 * d, 1, n_layers, dtype)
    fwd = batch * max_seq_len * d * FP32 * (n_layers + 1)
    return ModelBreakdown("LinRec", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=["LinRec: linear-attention Transformer (O(L) memory)"])


def _est_gsasrec(n_users, n_items, n_edges, hp, dtype):
    bd = _est_sasrec(n_users, n_items, n_edges, hp, dtype)
    bd.model = "gSASRec"
    # gSASRec adds gBCE loss — no new parameters, same memory as SASRec
    bd.notes.append("gSASRec: SASRec + gBCE loss (same memory as SASRec)")
    return bd


def _est_core(n_users, n_items, n_edges, hp, dtype):
    bd = _est_sasrec(n_users, n_items, n_edges, hp, dtype)
    bd.model = "CORE"
    bd.notes.append("CORE: Transformer-style with consistent representations")
    return bd


# ---------- Context-aware paradigm -----------------------------------------

def _context_base_params(n_users, n_items, hp, feat_dims, ctx_dims, dtype):
    """Parameters shared across all ContextRecommenderUtils subclasses."""
    d = int(hp["embedding_size"])
    total = 0
    total += _embedding_table(n_users, d, dtype)
    total += _embedding_table(_pad(n_items), d, dtype)
    total += dtype  # global_bias
    total += _embedding_table(n_users, 1, dtype)  # user_bias
    total += _embedding_table(_pad(n_items), 1, dtype)  # item_bias
    if feat_dims:
        total_feat = sum(feat_dims)
        total += _embedding_table(total_feat, d, dtype)
        total += _embedding_table(total_feat, 1, dtype)
    if ctx_dims:
        total_ctx = sum(ctx_dims)
        total += _embedding_table(total_ctx, d, dtype)
        total += _embedding_table(total_ctx, 1, dtype)
    return total


def _num_fields(feat_dims, ctx_dims):
    return 2 + len(feat_dims or []) + len(ctx_dims or [])


def _est_fm(n_users, n_items, n_edges, hp, dtype):
    feat = hp.get("feature_dims", [])
    ctx = hp.get("context_dims", [])
    batch = int(hp["batch_size"])
    d = int(hp["embedding_size"])
    params = _context_base_params(n_users, n_items, hp, feat, ctx, dtype)
    nf = _num_fields(feat, ctx)
    fwd = batch * nf * d * FP32
    return ModelBreakdown("FM", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"FM: {nf} fields × d={d}"])


def _est_deepfm(n_users, n_items, n_edges, hp, dtype):
    feat = hp.get("feature_dims", [])
    ctx = hp.get("context_dims", [])
    batch = int(hp["batch_size"])
    d = int(hp["embedding_size"])
    mlp_hidden = hp.get("mlp_hidden_size", [64, 32])
    if isinstance(mlp_hidden, int):
        mlp_hidden = [mlp_hidden]
    mlp_hidden = list(mlp_hidden)
    nf = _num_fields(feat, ctx)

    params = _context_base_params(n_users, n_items, hp, feat, ctx, dtype)
    params += _mlp_params([nf * d] + mlp_hidden, with_bn=False, dtype_bytes=dtype)
    params += (mlp_hidden[-1] + 1) * dtype  # deep_predict_layer

    fwd = batch * nf * d * FP32
    for h in mlp_hidden:
        fwd += batch * h * FP32
    return ModelBreakdown("DeepFM", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"DeepFM: {nf} fields × d={d}, MLP={mlp_hidden}"])


def _est_nfm(n_users, n_items, n_edges, hp, dtype):
    feat = hp.get("feature_dims", [])
    ctx = hp.get("context_dims", [])
    batch = int(hp["batch_size"])
    d = int(hp["embedding_size"])
    mlp_hidden = hp.get("mlp_hidden_size", [64, 32])
    if isinstance(mlp_hidden, int):
        mlp_hidden = [mlp_hidden]
    mlp_hidden = list(mlp_hidden)
    nf = _num_fields(feat, ctx)

    params = _context_base_params(n_users, n_items, hp, feat, ctx, dtype)
    # NFM pools FM output to (d,) and feeds into MLP
    params += _mlp_params([d] + mlp_hidden, with_bn=False, dtype_bytes=dtype)
    params += (mlp_hidden[-1] + 1) * dtype

    fwd = batch * nf * d * FP32 + batch * d * FP32
    for h in mlp_hidden:
        fwd += batch * h * FP32
    return ModelBreakdown("NFM", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"NFM: bilinear pooling + MLP={mlp_hidden}"])


def _est_afm(n_users, n_items, n_edges, hp, dtype):
    feat = hp.get("feature_dims", [])
    ctx = hp.get("context_dims", [])
    batch = int(hp["batch_size"])
    d = int(hp["embedding_size"])
    attn_size = int(hp.get("attention_size", 8))
    nf = _num_fields(feat, ctx)

    params = _context_base_params(n_users, n_items, hp, feat, ctx, dtype)
    # Attention net: Linear(d → attn_size) + attention h vector
    params += (d * attn_size + attn_size) * dtype + attn_size * dtype
    # Output projection: d → 1
    params += (d + 1) * dtype

    # Pairwise interactions: (batch, nf*(nf-1)/2, d)
    n_pairs = nf * (nf - 1) // 2
    fwd = batch * n_pairs * d * FP32
    return ModelBreakdown("AFM", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"AFM: pairwise attention over {n_pairs} interactions"])


def _est_dcn(n_users, n_items, n_edges, hp, dtype):
    feat = hp.get("feature_dims", [])
    ctx = hp.get("context_dims", [])
    batch = int(hp["batch_size"])
    d = int(hp["embedding_size"])
    cross_layers = int(hp.get("cross_layers", 3))
    mlp_hidden = hp.get("mlp_hidden_size", [64, 32])
    if isinstance(mlp_hidden, int):
        mlp_hidden = [mlp_hidden]
    mlp_hidden = list(mlp_hidden)
    nf = _num_fields(feat, ctx)
    input_dim = nf * d

    params = _context_base_params(n_users, n_items, hp, feat, ctx, dtype)
    # Cross network: per layer (input_dim, ) weight + (input_dim,) bias
    params += cross_layers * 2 * input_dim * dtype
    # Deep network
    params += _mlp_params([input_dim] + mlp_hidden, with_bn=False, dtype_bytes=dtype)
    # Final projection
    params += (input_dim + mlp_hidden[-1] + 1) * dtype

    fwd = batch * input_dim * FP32 * (cross_layers + 1)
    for h in mlp_hidden:
        fwd += batch * h * FP32
    return ModelBreakdown("DCN", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"DCN: cross={cross_layers}, deep={mlp_hidden}"])


def _est_dcnv2(n_users, n_items, n_edges, hp, dtype):
    feat = hp.get("feature_dims", [])
    ctx = hp.get("context_dims", [])
    batch = int(hp["batch_size"])
    d = int(hp["embedding_size"])
    cross_layers = int(hp.get("cross_layers", 3))
    mlp_hidden = hp.get("mlp_hidden_size", [64, 32])
    if isinstance(mlp_hidden, int):
        mlp_hidden = [mlp_hidden]
    mlp_hidden = list(mlp_hidden)
    nf = _num_fields(feat, ctx)
    input_dim = nf * d

    params = _context_base_params(n_users, n_items, hp, feat, ctx, dtype)
    # DCNv2 cross: per layer (input_dim × input_dim) matrix + bias
    params += cross_layers * (input_dim * input_dim + input_dim) * dtype
    params += _mlp_params([input_dim] + mlp_hidden, with_bn=False, dtype_bytes=dtype)
    params += (input_dim + mlp_hidden[-1] + 1) * dtype

    fwd = batch * input_dim * FP32 * (cross_layers + 1)
    for h in mlp_hidden:
        fwd += batch * h * FP32
    return ModelBreakdown("DCNv2", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"DCNv2: full matrix cross={cross_layers}, deep={mlp_hidden}"])


def _est_wide_and_deep(n_users, n_items, n_edges, hp, dtype):
    # Wide part = linear combination (FM first-order); Deep part = MLP
    feat = hp.get("feature_dims", [])
    ctx = hp.get("context_dims", [])
    batch = int(hp["batch_size"])
    d = int(hp["embedding_size"])
    mlp_hidden = hp.get("mlp_hidden_size", [64, 32])
    if isinstance(mlp_hidden, int):
        mlp_hidden = [mlp_hidden]
    mlp_hidden = list(mlp_hidden)
    nf = _num_fields(feat, ctx)

    params = _context_base_params(n_users, n_items, hp, feat, ctx, dtype)
    params += _mlp_params([nf * d] + mlp_hidden, with_bn=False, dtype_bytes=dtype)
    params += (mlp_hidden[-1] + 1) * dtype

    fwd = batch * nf * d * FP32
    for h in mlp_hidden:
        fwd += batch * h * FP32
    return ModelBreakdown("WideAndDeep", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"Wide&Deep: linear + MLP={mlp_hidden}"])


def _est_xdeepfm(n_users, n_items, n_edges, hp, dtype):
    feat = hp.get("feature_dims", [])
    ctx = hp.get("context_dims", [])
    batch = int(hp["batch_size"])
    d = int(hp["embedding_size"])
    mlp_hidden = hp.get("mlp_hidden_size", [64, 32])
    if isinstance(mlp_hidden, int):
        mlp_hidden = [mlp_hidden]
    mlp_hidden = list(mlp_hidden)
    cin_layers = hp.get("cin_layer_size", [64, 64])
    if isinstance(cin_layers, int):
        cin_layers = [cin_layers]
    cin_layers = list(cin_layers)
    nf = _num_fields(feat, ctx)

    params = _context_base_params(n_users, n_items, hp, feat, ctx, dtype)
    # CIN: per layer (h_prev × h_0, h_curr) 1D conv kernels
    prev = nf
    for h in cin_layers:
        params += prev * nf * h * dtype
        prev = h
    # Deep part
    params += _mlp_params([nf * d] + mlp_hidden, with_bn=False, dtype_bytes=dtype)
    # Prediction layer
    params += (sum(cin_layers) + mlp_hidden[-1] + 1 + 1) * dtype

    fwd = batch * nf * d * FP32
    for h in cin_layers:
        fwd += batch * h * d * FP32
    for h in mlp_hidden:
        fwd += batch * h * FP32
    return ModelBreakdown("xDeepFM", params_bytes=params, fwd_activation_bytes=fwd,
                          notes=[f"xDeepFM: CIN={cin_layers}, DNN={mlp_hidden}"])


# ---------- Content-based / Hybrid -----------------------------------------

def _est_vsm(n_users, n_items, n_edges, hp, dtype):
    # VSM: TF-IDF-like item-feature vectors → cosine similarity
    n_feat = int(hp.get("n_features", 1000))
    sparse_item_feat = int(n_items * n_feat * 0.01) * (FP32 + INT32)
    return ModelBreakdown("VSM", params_bytes=sparse_item_feat,
                          notes=[f"VSM: sparse item×feature matrix ({n_feat} features)"])


def _est_addease(n_users, n_items, n_edges, hp, dtype):
    bd = _est_ease(n_users, n_items, n_edges, hp, dtype)
    bd.model = "AddEASE"
    d = int(hp.get("embedding_size", 64))
    bd.params_bytes += _embedding_table(n_users, d, dtype)
    bd.notes.append(f"AddEASE: EASE + user embedding(d={d})")
    return bd


def _est_cease(n_users, n_items, n_edges, hp, dtype):
    bd = _est_ease(n_users, n_items, n_edges, hp, dtype)
    bd.model = "CEASE"
    bd.notes[0] = "CEASE: collaborative EASE (same memory as EASE)"
    return bd


def _est_attributeitemknn(n_users, n_items, n_edges, hp, dtype):
    bd = _est_itemknn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "AttributeItemKNN"
    n_feat = int(hp.get("n_features", 1000))
    bd.params_bytes += int(n_items * n_feat * 0.01) * (FP32 + INT32)
    bd.notes.append(f"AttributeItemKNN: + item×feature({n_feat}) sparse matrix")
    return bd


def _est_attributeuserknn(n_users, n_items, n_edges, hp, dtype):
    bd = _est_userknn(n_users, n_items, n_edges, hp, dtype)
    bd.model = "AttributeUserKNN"
    n_feat = int(hp.get("n_features", 1000))
    bd.params_bytes += int(n_users * n_feat * 0.01) * (FP32 + INT32)
    bd.notes.append(f"AttributeUserKNN: + user×feature({n_feat}) sparse matrix")
    return bd


# ---------- Unpersonalized / Utility ---------------------------------------

def _est_pop(n_users, n_items, n_edges, hp, dtype):
    return ModelBreakdown("Pop", params_bytes=n_items * FP32,
                          notes=["Pop: item popularity vector"])


def _est_random(n_users, n_items, n_edges, hp, dtype):
    return ModelBreakdown("Random", params_bytes=0,
                          notes=["Random: stateless"])


def _est_proxy(n_users, n_items, n_edges, hp, dtype):
    return ModelBreakdown("ProxyRecommender", params_bytes=0,
                          notes=["ProxyRecommender: wraps external model — unknown cost"])


# ---------- Project model --------------------------------------------------

def _est_causal_ecl_rec(n_users, n_items, n_edges, hp, dtype):
    """CausalECLRec (experiments/model/v1/causal_ecl_rec.py).

    Architecture:
        - user_embedding (n_users × d)  [learnable]
        - student MLP([d] + mlp_hidden_layers + [d]) with BN + ReLU + dropout
        - teacher_item_emb (n_items × d)  [FROZEN buffer, not a parameter]
        - teacher_item_norm (n_items × d)  [buffer: F.normalize copy]
        - ips_item_weights (n_items,)  [buffer]
    """
    d = int(hp["embedding_size"])
    mlp_hidden = hp["mlp_hidden_layers"]
    # Config uses [[512, 256]] (list of list) or [512, 256].
    if isinstance(mlp_hidden, list) and mlp_hidden and isinstance(mlp_hidden[0], list):
        mlp_hidden = mlp_hidden[0]
    mlp_hidden = list(mlp_hidden)
    batch = int(hp["batch_size"])

    mlp_layers = [d] + mlp_hidden + [d]
    params = _embedding_table(n_users, d, dtype)
    params += _mlp_params(mlp_layers, with_bn=True, dtype_bytes=dtype)

    # Buffers (not trainable — no grad, no Adam state)
    teacher = _embedding_table(n_items, d, dtype)
    teacher_norm = _embedding_table(n_items, d, dtype)
    ips_weights = n_items * FP32
    bn_buffers = _mlp_bn_buffers(mlp_layers)
    buffers = teacher + teacher_norm + ips_weights + bn_buffers

    # Forward activation:
    #   (batch, d) user emb
    #   (batch, hidden_i) per MLP layer
    #   (batch, n_items) logits matrix (dominates!)
    fwd = batch * d * FP32
    for h in mlp_hidden:
        fwd += batch * h * FP32
    fwd += batch * n_items * FP32  # logits
    # logsumexp keeps another copy
    fwd += batch * FP32

    return ModelBreakdown("CausalECLRec",
                          params_bytes=params,
                          persistent_buffers_bytes=buffers,
                          fwd_activation_bytes=fwd,
                          notes=[f"CausalECLRec: user_emb + MLP({mlp_layers})",
                                 f"Buffers: teacher({_human(teacher)}) + "
                                 f"teacher_norm({_human(teacher_norm)}) + "
                                 f"ips({_human(ips_weights)})",
                                 f"Full-batch logits: {batch}×{n_items}={_human(batch * n_items * FP32)}"])


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

MODEL_ESTIMATORS: dict[str, Callable] = {
    # Latent factor
    "BPR": _est_bpr,
    "FISM": _est_fism,
    "Slim": _est_slim,
    "SLIM": _est_slim,
    "ADMMSlim": _est_admmslim,

    # Graph-based
    "LightGCN": _est_lightgcn,
    "NGCF": _est_ngcf,
    "SGL": _est_sgl,
    "LightGCL": _est_lightgcl,
    "SGCL": _est_sgcl,
    "UltraGCN": _est_ultragcn,
    "DGCF": _est_dgcf,
    "LightGCNpp": _est_lightgcnpp,
    "GCMC": _est_gcmc,
    "LightCCF": _est_lightccf,
    "EGCF": _est_egcf,
    "ESIGCF": _est_esigcf,
    "LightGODE": _est_lightgode,
    "MixRec": _est_mixrec,
    "XSimGCL": _est_xsimgcl,
    "RP3Beta": _est_rp3beta,

    # Autoencoder
    "MultiVAE": _est_multivae,
    "MultiDAE": _est_multidae,
    "CDAE": _est_cdae,
    "EASE": _est_ease,
    "ELSA": _est_elsa,
    "MacridVAE": _est_macridvae,
    "SANSA": _est_sansa,

    # Neural CF
    "NeuMF": _est_neumf,
    "ConvNCF": _est_convncf,

    # KNN
    "ItemKNN": _est_itemknn,
    "UserKNN": _est_userknn,

    # Sequential
    "SASRec": _est_sasrec,
    "BERT4Rec": _est_bert4rec,
    "GRU4Rec": _est_gru4rec,
    "Caser": _est_caser,
    "NARM": _est_narm,
    "FOSSIL": _est_fossil,
    "LightSANs": _est_lightsans,
    "LinRec": _est_linrec,
    "gSASRec": _est_gsasrec,
    "CORE": _est_core,

    # Context-aware
    "FM": _est_fm,
    "DeepFM": _est_deepfm,
    "NFM": _est_nfm,
    "AFM": _est_afm,
    "DCN": _est_dcn,
    "DCNv2": _est_dcnv2,
    "WideAndDeep": _est_wide_and_deep,
    "xDeepFM": _est_xdeepfm,

    # Content-based
    "VSM": _est_vsm,

    # Hybrid
    "AddEASE": _est_addease,
    "CEASE": _est_cease,
    "AttributeItemKNN": _est_attributeitemknn,
    "AttributeUserKNN": _est_attributeuserknn,

    # Unpersonalized
    "Pop": _est_pop,
    "Random": _est_random,

    # Utility
    "ProxyRecommender": _est_proxy,

    # Project
    "CausalECLRec": _est_causal_ecl_rec,
}


# -----------------------------------------------------------------------------
# Pipeline overhead
# -----------------------------------------------------------------------------

_OPTIMIZER_STATE_MULTIPLIER = {
    "adam": 2.0, "adamw": 2.0, "nadam": 2.0, "radam": 2.0,
    "sgd": 1.0,  # momentum by default
    "sgd_no_momentum": 0.0,
    "rmsprop": 1.0, "adagrad": 1.0, "adadelta": 2.0,
}


def _resolve_optimizer(hp: dict) -> tuple[str, float]:
    """Read optimizer name from the config; default to Adam."""
    name = str(hp.get("optimizer", "Adam")).lower()
    opt_cfg = hp.get("optimization") or {}
    if isinstance(opt_cfg, dict):
        name = str(opt_cfg.get("optimizer", name)).lower()
    multiplier = _OPTIMIZER_STATE_MULTIPLIER.get(name, 2.0)
    return name, multiplier


def estimate_data_prep(dataset: DatasetStats, cfg: dict) -> DataPrepEstimate:
    """Estimate CPU + RAM for the remote data-preparation Ray task.

    The task in experiments/warprec/pipelines/remotes/data.py::remote_data_preparation
    runs: raw CSV read (polars) → k-core filter → splitting → Dataset build (CSR
    + ID maps) → evaluation dataloader construction. It is CPU-bound; VRAM = 0.

    CPU heuristic: more threads help polars on larger files but saturate around 8.
    RAM heuristic: peak ≈ 3× raw row tensor + CSR + polars working copy + process baseline.
    """
    E = dataset.n_interactions
    n_users = dataset.n_users
    n_items = dataset.n_items

    # CPU: logarithmic bracket on n_interactions
    if E < 100_000:
        cpu = 1
    elif E < 1_000_000:
        cpu = 2
    elif E < 10_000_000:
        cpu = 3
    elif E < 50_000_000:
        cpu = 4
    elif E < 100_000_000:
        cpu = 6
    else:
        cpu = 8

    # RAM components
    # Raw polars DataFrame: user_id + item_id + timestamp? + rating → ~24 B/row
    row_bytes_per_interaction = 24
    raw_df = E * row_bytes_per_interaction
    # Peak: polars keeps a working copy during k-core iterations (x2) + splitting (x1.5)
    polars_peak = int(raw_df * 3)
    # CSR interaction matrix (what Interactions.get_sparse() returns)
    csr = E * (INT32 + FP32) + (n_users + 1) * INT32
    # Evaluation dataloader pre-compute (tensor of test user-item pairs)
    eval_buf = int(E * 0.2) * 2 * INT64  # 20% held out, 2 columns (user, item)
    # ID remap dicts (Python dict overhead is high)
    id_maps = (n_users + n_items) * 100
    # Process baseline (Python + polars + PyTorch + Ray worker)
    baseline = 600 * MIB

    ram = polars_peak + csr + eval_buf + id_maps + baseline

    # Check if a label_selector_data_prep is set → note that the cluster's
    # `vram_gb` custom resource is typically used as a scheduling hint, not
    # an actual VRAM requirement.
    general = (cfg.get("general") or {}) if cfg else {}
    user_requested_custom = general.get("custom_resources_data_prep") or {}

    notes = [
        f"Polars scan + k-core + split: peak ~{_human(polars_peak)}",
        f"CSR interaction matrix: {_human(csr)}",
        f"Eval dataloader pre-compute (20% holdout × 2 cols × int64): {_human(eval_buf)}",
        f"Process baseline (polars + Ray worker): {_human(baseline)}",
    ]
    if user_requested_custom:
        notes.append(
            f"Config currently sets custom_resources_data_prep={user_requested_custom} — "
            f"data prep uses no GPU, so `vram_gb` (if set) is a host-pinning hint only"
        )

    return DataPrepEstimate(cpu=cpu, ram_bytes=ram, vram_bytes=0, notes=notes)


def _pipeline_ram_bytes(cfg: dict, hp: dict, dataset: DatasetStats) -> int:
    """Process-side RAM overhead (CPU).

    Components:
      - CSR interaction matrix (indices + data + indptr)
      - ID remap dicts (user_id_map + item_id_map)
      - Dataloader staging (batch × (1 + 1 + neg_samples) × int64)
      - Fixed process overhead (Python + PyTorch + Ray worker)
    """
    E = dataset.n_interactions
    csr = E * (INT32 + FP32) + (dataset.n_users + 1) * INT32
    id_maps = (dataset.n_users + dataset.n_items) * 100  # rough Python-dict overhead
    batch = int(hp.get("batch_size", 1024))
    neg = int(hp.get("neg_samples", 1))
    loader = batch * (2 + neg) * INT64 * 4  # pos+neg + workers prefetch
    fixed = 500 * MIB
    return csr + id_maps + loader + fixed


def estimate_trial(model: str, hp: dict, dataset: DatasetStats,
                   dtype_bytes: int = FP32) -> PipelineEstimate:
    estimator = MODEL_ESTIMATORS.get(model)
    if estimator is None:
        warnings.warn(f"[UNSUPPORTED_MODEL] '{model}' — falling back to embedding-only estimate")
        d = int(hp.get("embedding_size", 64))
        params = _embedding_table(dataset.n_users, d, dtype_bytes) + \
                 _embedding_table(_pad(dataset.n_items), d, dtype_bytes)
        breakdown = ModelBreakdown(model, params_bytes=params,
                                   notes=[f"[UNSUPPORTED_MODEL] embedding-only fallback"])
    else:
        breakdown = estimator(dataset.n_users, dataset.n_items,
                              dataset.n_interactions, hp, dtype_bytes)

    opt_name, opt_multiplier = _resolve_optimizer(hp)
    gradient_bytes = breakdown.trainable_bytes()
    optimizer_state_bytes = int(breakdown.trainable_bytes() * opt_multiplier)
    # Backward keeps saved activations (~= forward for most models).
    # Closed-form models have no backward pass → no activation doubling.
    activation_multiplier = 1 if breakdown.closed_form else 2
    activation_peak_bytes = activation_multiplier * breakdown.fwd_activation_bytes

    # Closed-form models fit on CPU (numpy/scipy/sklearn), so their "model memory"
    # is actually RAM, not VRAM. Route it accordingly.
    model_memory = (breakdown.gpu_resident_bytes()
                    + gradient_bytes
                    + optimizer_state_bytes
                    + activation_peak_bytes)
    pipeline_ram = _pipeline_ram_bytes({}, hp, dataset)

    if breakdown.closed_form:
        vram_total = 0
        ram_total = model_memory + pipeline_ram
    else:
        vram_total = model_memory
        ram_total = pipeline_ram

    trial_id = "_".join(f"{k}={v}" for k, v in hp.items()
                        if k in ("embedding_size", "n_layers", "batch_size",
                                 "max_seq_len", "n_heads", "inner_size"))
    return PipelineEstimate(
        model=model,
        trial_id=trial_id or "default",
        hyperparameters=hp,
        dataset=dataset,
        breakdown=breakdown,
        optimizer=opt_name,
        optimizer_state_bytes=optimizer_state_bytes,
        gradient_bytes=gradient_bytes,
        activation_peak_bytes=activation_peak_bytes,
        vram_total_bytes=vram_total,
        ram_total_bytes=ram_total,
    )


# -----------------------------------------------------------------------------
# HPO enumeration
# -----------------------------------------------------------------------------

_MEMORY_AFFECTING = {
    "embedding_size", "batch_size", "n_layers", "n_heads", "inner_size",
    "max_seq_len", "mlp_hidden_layers", "mlp_hidden_size", "weight_size",
    "gru_size", "hidden_size", "neg_samples", "intermediate_dim", "latent_dim",
    "mf_embedding_size", "mlp_embedding_size", "cross_layers", "cin_layer_size",
    "k", "n_factors", "n_order", "k_interests", "kfac",
}


def _expand_value(value: Any) -> list[Any]:
    """Expand one hyperparameter value into the list of trial values it implies.

    Supports:
      - scalar → [scalar]
      - list of scalars → itself (grid)
      - ['uniform', lo, hi] / ['loguniform', lo, hi] → [lo, hi] (envelope)
      - ['choice', a, b, ...] → [a, b, ...]
      - list of lists (e.g. mlp_hidden_layers: [[512,256],[256,128]]) → itself
    """
    if isinstance(value, list):
        if not value:
            return [value]
        # Search-space annotation
        if isinstance(value[0], str) and value[0] in ("uniform", "loguniform",
                                                       "quniform", "qloguniform"):
            # Sample at min and max (numeric values follow)
            numeric = [v for v in value[1:] if isinstance(v, (int, float))]
            if len(numeric) >= 2:
                return [min(numeric), max(numeric)]
            return [value]
        if isinstance(value[0], str) and value[0] == "choice":
            return list(value[1:])
        # If all elements are lists (e.g. mlp_hidden_layers: [[512,256]]) → list of configs
        if all(isinstance(v, list) for v in value):
            return value
        # Plain grid of scalars
        return value
    return [value]


def _enumerate_trials(model_hp: dict) -> Iterable[dict]:
    """Given a (possibly HPO-annotated) hyperparameter dict, yield all trial configs.

    Only memory-affecting axes are expanded. Non-memory axes are collapsed to a
    single representative value (the first element if list, else the scalar).
    """
    expanded_axes: dict[str, list] = {}
    fixed_axes: dict[str, Any] = {}
    for key, value in model_hp.items():
        if key in _MEMORY_AFFECTING:
            expanded_axes[key] = _expand_value(value)
        else:
            # Collapse non-memory axes to a representative
            vals = _expand_value(value)
            fixed_axes[key] = vals[0] if vals else None

    axis_names = list(expanded_axes.keys())
    axis_values = [expanded_axes[k] for k in axis_names]

    if not axis_names:
        yield dict(fixed_axes)
        return

    for combo in itertools.product(*axis_values):
        trial = dict(fixed_axes)
        trial.update(dict(zip(axis_names, combo)))
        yield trial


# -----------------------------------------------------------------------------
# Dataset resolution
# -----------------------------------------------------------------------------

def _resolve_dataset_stats(cfg: dict, config_path: Path) -> DatasetStats:
    reader = cfg.get("reader", {}) or {}
    writer = cfg.get("writer", {}) or {}
    dataset_name = writer.get("dataset_name", "unknown")

    # Priority 1: file scan
    loading_strategy = reader.get("loading_strategy", "dataset")
    labels = reader.get("labels", {}) or {}
    user_col = labels.get("user_id_label", "user_id")
    item_col = labels.get("item_id_label", "item_id")
    sep = reader.get("sep", "\t")

    candidate_paths: list[Path] = []
    cfg_dir = config_path.parent.parent  # configs/<dataset>/file.yml → configs/
    experiments_dir = cfg_dir.parent  # experiments/

    if loading_strategy == "dataset":
        local_path = reader.get("local_path")
        if local_path:
            candidate_paths = [Path(local_path), experiments_dir / local_path]
    else:  # split
        split = reader.get("split", {}) or {}
        local_path = split.get("local_path")
        ext = split.get("ext", ".tsv")
        if local_path:
            base = Path(local_path)
            alt_base = experiments_dir / local_path
            candidate_paths = [base / f"train{ext}", alt_base / f"train{ext}"]

    for p in candidate_paths:
        if p.exists():
            try:
                stats = _polars_scan(p, user_col, item_col, sep)
                if stats is not None:
                    return DatasetStats(*stats, source=f"scan:{p}")
            except Exception as exc:
                warnings.warn(f"Failed to scan {p}: {exc}")

    # Priority 2: baked-in registry
    if dataset_name in _KNOWN_DATASETS:
        return _KNOWN_DATASETS[dataset_name]

    raise RuntimeError(
        f"Could not resolve dataset stats for '{dataset_name}'. "
        f"Tried file scan at: {[str(p) for p in candidate_paths] or 'no paths'}. "
        f"Known datasets: {sorted(_KNOWN_DATASETS.keys())}."
    )


def _polars_scan(path: Path, user_col: str, item_col: str, sep: str) -> tuple[int, int, int] | None:
    try:
        import polars as pl
    except ImportError:
        return None
    # scan_csv is lazy
    frame = pl.scan_csv(str(path), separator=sep, has_header=True)
    result = frame.select(
        pl.col(user_col).n_unique().alias("n_users"),
        pl.col(item_col).n_unique().alias("n_items"),
        pl.len().alias("n_interactions"),
    ).collect()
    row = result.row(0)
    return int(row[0]), int(row[1]), int(row[2])


# -----------------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------------

def _format_table(estimates: list[PipelineEstimate],
                  data_prep: DataPrepEstimate | None = None) -> str:
    lines = []
    first = estimates[0]
    ds = first.dataset
    lines.append("=" * 90)
    lines.append(f"WarpRec Memory Estimator")
    lines.append("=" * 90)
    lines.append(f"Dataset: {_format_dataset(ds)}")
    lines.append(f"Source : {ds.source}")
    lines.append("")

    # Data-prep section — maps to general.cpu_data_prep / general.custom_resources_data_prep
    if data_prep is not None:
        lines.append("─" * 90)
        lines.append("Data preparation (one-shot Ray task, CPU-bound)")
        lines.append("─" * 90)
        lines.append(f"  CPU                 : {data_prep.cpu} thread(s)")
        lines.append(f"  RAM peak            : {_human(data_prep.ram_bytes)}"
                     f"   → request ≥{_gb_round_up(data_prep.ram_bytes)} GiB")
        lines.append(f"  VRAM                : {_human(data_prep.vram_bytes)}  (no GPU used)")
        for note in data_prep.notes:
            lines.append(f"    · {note}")
        lines.append("")
        lines.append(f"  Suggested config values for general:")
        lines.append(f"    cpu_data_prep: {data_prep.cpu}")
        if data_prep.vram_bytes == 0:
            lines.append(f"    custom_resources_data_prep: {{}}  "
                         f"# or {{vram_gb: 0}} — data prep needs no GPU")
        else:
            lines.append(f"    custom_resources_data_prep: "
                         f"{{vram_gb: {max(1, math.ceil(data_prep.ram_bytes / GIB))}}}")
        lines.append("")

    current_model = None
    for est in estimates:
        if est.model != current_model:
            current_model = est.model
            lines.append(f"{'─' * 90}")
            lines.append(f"Model: {est.model}  ({len(_trials_for_model(estimates, est.model))} trial(s))")
            lines.append(f"{'─' * 90}")
        lines.append(f"  Trial [{est.trial_id}]  optimizer={est.optimizer}")
        bd = est.breakdown
        lines.append(f"    Parameters          : {_human(bd.params_bytes):>12}")
        lines.append(f"    Persistent buffers  : {_human(bd.persistent_buffers_bytes):>12}")
        lines.append(f"    Forward activation  : {_human(bd.fwd_activation_bytes):>12}")
        lines.append(f"    Gradient buffer     : {_human(est.gradient_bytes):>12}")
        lines.append(f"    Optimizer state     : {_human(est.optimizer_state_bytes):>12}")
        lines.append(f"    Activation peak(2x) : {_human(est.activation_peak_bytes):>12}")
        lines.append(f"    {'─' * 40}")
        if bd.closed_form:
            lines.append(f"    VRAM total          : {_human(est.vram_total_bytes):>12}"
                         f"   (closed-form fit on CPU — no GPU needed)")
        else:
            lines.append(f"    VRAM total          : {_human(est.vram_total_bytes):>12}"
                         f"   → request ≥{_gb_round_up(est.vram_total_bytes)} GiB")
        lines.append(f"    RAM total           : {_human(est.ram_total_bytes):>12}"
                     f"   → request ≥{_gb_round_up(est.ram_total_bytes)} GiB")
        for note in bd.notes:
            lines.append(f"    · {note}")
        lines.append("")

    # Summary across all trials
    lines.append("=" * 90)
    lines.append("Summary")
    lines.append("=" * 90)
    peak_vram = max(e.vram_total_bytes for e in estimates)
    peak_train_ram = max(e.ram_total_bytes for e in estimates)
    any_closed_form = any(e.breakdown.closed_form for e in estimates)
    all_closed_form = all(e.breakdown.closed_form for e in estimates)
    lines.append(f"  Training ({len(estimates)} trial(s))")
    if all_closed_form or peak_vram == 0:
        lines.append(f"    Peak VRAM per trial  : {_human(peak_vram)}  "
                     f"(closed-form fit on CPU — no GPU needed)")
    else:
        lines.append(f"    Peak VRAM per trial  : {_human(peak_vram)}  "
                     f"→ GPU must have ≥ {_gb_round_up(peak_vram)} GiB")
    lines.append(f"    Peak RAM per trial   : {_human(peak_train_ram)}  "
                 f"→ request ≥{_gb_round_up(peak_train_ram)} GiB")
    if data_prep is not None:
        lines.append(f"  Data preparation (one-shot, runs before training)")
        lines.append(f"    CPU                  : {data_prep.cpu} thread(s)")
        lines.append(f"    RAM peak             : {_human(data_prep.ram_bytes)}  "
                     f"→ request ≥{_gb_round_up(data_prep.ram_bytes)} GiB")
        lines.append(f"    VRAM                 : {_human(data_prep.vram_bytes)}")
    lines.append("=" * 90)
    return "\n".join(lines)


def _trials_for_model(estimates: list[PipelineEstimate], model: str) -> list[PipelineEstimate]:
    return [e for e in estimates if e.model == model]


def _format_dataset(ds: DatasetStats) -> str:
    return (f"{ds.n_users:,} users × {ds.n_items:,} items × "
            f"{ds.n_interactions:,} interactions")


def _gb_round_up(nbytes: int) -> int:
    return max(1, math.ceil(nbytes / GIB * 1.5))  # 50% safety margin


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def run(config_path: Path, output_format: str = "table") -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = _resolve_dataset_stats(cfg, config_path)
    data_prep = estimate_data_prep(dataset, cfg)

    models_cfg = cfg.get("models") or {}
    if not models_cfg:
        raise RuntimeError("Config has no 'models' section.")

    estimates: list[PipelineEstimate] = []
    for model_name, model_hp in models_cfg.items():
        if model_hp is None:
            model_hp = {}
        for trial_hp in _enumerate_trials(model_hp):
            est = estimate_trial(model_name, trial_hp, dataset)
            estimates.append(est)

    if output_format == "json":
        return json.dumps({
            "config": str(config_path),
            "dataset": asdict(dataset),
            "data_prep": asdict(data_prep),
            "trials": [e.to_dict() for e in estimates],
            "summary": {
                "peak_vram_bytes": max(e.vram_total_bytes for e in estimates),
                "n_trials": len(estimates),
                "suggested_general": {
                    "cpu_data_prep": data_prep.cpu,
                    "custom_resources_data_prep": (
                        {"vram_gb": 0} if data_prep.vram_bytes == 0
                        else {"vram_gb": max(1, math.ceil(data_prep.vram_bytes / GIB))}
                    ),
                },
            },
        }, indent=2)
    return _format_table(estimates, data_prep)


# -----------------------------------------------------------------------------
# --self-test
# -----------------------------------------------------------------------------

def _self_test_closed_form() -> list[str]:
    """Tier A: hand-computed ground-truth assertions."""
    results = []

    # BPR: n_users=1000, n_items=500, d=32 → params = (1000 + 501) × 32 × 4
    bd = _est_bpr(1000, 500, 10_000,
                  {"embedding_size": 32, "batch_size": 128}, FP32)
    expected = (1000 + 501) * 32 * 4
    assert bd.params_bytes == expected, f"BPR: expected {expected}, got {bd.params_bytes}"
    results.append(f"OK  BPR params (1000 users × 500 items × d=32) = {_human(expected)}")

    # LightGCN: LGConv has no params, only embeddings
    bd = _est_lightgcn(1000, 500, 10_000,
                       {"embedding_size": 32, "n_layers": 3, "batch_size": 128}, FP32)
    expected = (1000 + 501) * 32 * 4
    assert bd.params_bytes == expected, f"LightGCN: expected {expected}, got {bd.params_bytes}"
    results.append(f"OK  LightGCN params (same as BPR since LGConv has no params) = {_human(expected)}")

    # SASRec: item_emb + pos_emb + transformer + layernorm
    bd = _est_sasrec(100, 50, 1000, {
        "embedding_size": 16, "n_layers": 2, "n_heads": 2, "inner_size": 32,
        "max_seq_len": 10, "batch_size": 8, "neg_samples": 1,
    }, FP32)
    # item_emb = 51*16*4=3264, pos_emb=10*16*4=640, ln=2*16*4=128
    # per transformer layer:
    #   attn: 4*16² + 4*16 = 1024+64 = 1088
    #   ffn:  2*16*32 + 32 + 16 = 1024 + 48 = 1072
    #   2 LN: 4*16 = 64
    # per-layer = 2224; 2 layers = 4448; × 4 bytes = 17792
    expected = 51 * 16 * 4 + 10 * 16 * 4 + 2 * 16 * 4 + 2224 * 2 * 4
    assert bd.params_bytes == expected, f"SASRec: expected {expected}, got {bd.params_bytes}"
    results.append(f"OK  SASRec params ({_human(expected)})")

    # CausalECLRec: user_emb + MLP([16, 32, 16, 16]) with BN
    bd = _est_causal_ecl_rec(100, 50, 1000, {
        "embedding_size": 16, "mlp_hidden_layers": [[32]], "batch_size": 8,
    }, FP32)
    # user_emb = 100*16*4 = 6400
    # MLP [16 → 32 → 16]: (16*32+32) + (2*32) + (32*16+16) + (2*16) = 544 + 64 + 528 + 32 = 1168
    # × 4 bytes = 4672
    # teacher=3200, teacher_norm=3200, ips=200,
    # BN buffers (for 2 Linear outputs 32,16): 2*(32*4) + 2*(16*4) + 2*8 = 256+128+16 = 400
    expected_params = 6400 + 1168 * 4
    assert bd.params_bytes == expected_params, \
        f"CausalECLRec params: expected {expected_params}, got {bd.params_bytes}"
    expected_buffers = 3200 + 3200 + 200 + 400
    assert bd.persistent_buffers_bytes == expected_buffers, \
        f"CausalECLRec buffers: expected {expected_buffers}, got {bd.persistent_buffers_bytes}"
    results.append(f"OK  CausalECLRec params ({_human(expected_params)}) + "
                   f"buffers ({_human(expected_buffers)})")

    return results


def _self_test_live() -> list[str]:
    """Tier B: instantiate WarpRec models and compare exact parameter counts."""
    results: list[str] = []
    try:
        import torch
        from torch import nn
    except Exception as exc:
        results.append(f"SKIP Tier B (torch not importable: {exc})")
        return results

    def _sum_params(mod) -> int:
        return sum(p.numel() * p.element_size() for p in mod.parameters())

    def _sum_buffers(mod) -> int:
        return sum(b.numel() * b.element_size() for _, b in mod.named_buffers() if b is not None)

    def _check(name: str, live_p: int, est_p: int, live_b: int = None, est_b: int = None):
        assert live_p == est_p, f"{name} params: live={live_p}, est={est_p}"
        if live_b is not None and est_b is not None:
            # Some models have extra buffers (e.g. cache, causal mask with int64 vs bool);
            # allow exact match only for persistent buffers we explicitly model.
            assert live_b == est_b, f"{name} buffers: live={live_b}, est={est_b}"
        buf_str = f", buffers={_human(live_b)}" if live_b is not None else ""
        results.append(f"OK  {name}: params={_human(live_p)}{buf_str}")

    # ---- BPR ----
    from warprec.recommenders.collaborative_filtering_recommender.latent_factor.bpr import BPR
    info = {"n_users": 100, "n_items": 50}
    bpr_hp = {"embedding_size": 16, "reg_weight": 0.0, "batch_size": 8,
              "epochs": 1, "learning_rate": 1e-3}
    bpr = BPR(bpr_hp, info)
    _check("BPR", _sum_params(bpr), _est_bpr(100, 50, 1000, bpr_hp, FP32).params_bytes)

    # ---- NeuMF ----
    try:
        from warprec.recommenders.collaborative_filtering_recommender.neural.neumf import NeuMF
        neumf_hp = {
            "mf_embedding_size": 8, "mlp_embedding_size": 8,
            "mlp_hidden_size": [16, 8], "mf_train": True, "mlp_train": True,
            "dropout": 0.0, "reg_weight": 0.0, "weight_decay": 0.0,
            "batch_size": 8, "epochs": 1, "learning_rate": 1e-3, "neg_samples": 1,
        }
        neumf = NeuMF(neumf_hp, info)
        _check("NeuMF", _sum_params(neumf),
               _est_neumf(100, 50, 1000, neumf_hp, FP32).params_bytes)
    except Exception as exc:
        results.append(f"SKIP NeuMF: {exc}")

    # ---- MultiVAE ----
    try:
        from warprec.recommenders.collaborative_filtering_recommender.autoencoder.multivae import MultiVAE
        # MultiVAE needs interactions — pass a mocked sparse matrix
        from warprec.data.entities import Interactions
        import numpy as np
        import scipy.sparse as sp
        # Build a tiny Interactions-like object by passing a csr to the MultiVAE init path;
        # but MultiVAE calls interactions.get_sparse() — so we need a real Interactions.
        # Create a small dense interaction matrix → Interactions.
        rows = np.array([0, 1, 2, 0, 1])
        cols = np.array([0, 1, 2, 3, 4])
        data = np.ones(len(rows), dtype=np.float32)
        sp_mat = sp.csr_matrix((data, (rows, cols)), shape=(100, 50))

        class _MockInteractions:
            def get_sparse(self):
                return sp_mat
        mvae_hp = {
            "intermediate_dim": 32, "latent_dim": 16, "corruption": 0.0,
            "weight_decay": 0.0, "batch_size": 8, "epochs": 1,
            "learning_rate": 1e-3, "anneal_cap": 0.2, "anneal_step": 10,
        }
        mvae = MultiVAE(mvae_hp, info, _MockInteractions())
        _check("MultiVAE", _sum_params(mvae),
               _est_multivae(100, 50, 1000, mvae_hp, FP32).params_bytes)
    except Exception as exc:
        results.append(f"SKIP MultiVAE: {exc}")

    # ---- SASRec ----
    try:
        from warprec.recommenders.sequential_recommender.sasrec import SASRec
        sasrec_hp = {
            "embedding_size": 16, "n_layers": 2, "n_heads": 2, "inner_size": 32,
            "dropout_prob": 0.0, "attn_dropout_prob": 0.0,
            "reg_weight": 0.0, "weight_decay": 0.0, "batch_size": 8,
            "epochs": 1, "learning_rate": 1e-3, "neg_samples": 1, "max_seq_len": 10,
        }
        sasrec = SASRec(sasrec_hp, info)
        _check("SASRec", _sum_params(sasrec),
               _est_sasrec(100, 50, 1000, sasrec_hp, FP32).params_bytes)
    except Exception as exc:
        results.append(f"SKIP SASRec: {exc}")

    # ---- LightGCN (needs torch_geometric + Interactions) ----
    try:
        from warprec.recommenders.collaborative_filtering_recommender.graph_based.lightgcn import LightGCN
        import scipy.sparse as sp
        import numpy as np
        rows = np.array([0, 1, 2, 0, 1, 3])
        cols = np.array([0, 1, 2, 3, 4, 5])
        data = np.ones(len(rows), dtype=np.float32)
        sp_mat = sp.csr_matrix((data, (rows, cols)), shape=(100, 50))

        class _MockInteractions2:
            def get_sparse(self):
                return sp_mat
        lgcn_hp = {
            "embedding_size": 16, "n_layers": 3, "reg_weight": 0.0,
            "batch_size": 8, "epochs": 1, "learning_rate": 1e-3,
        }
        lgcn = LightGCN(lgcn_hp, info, _MockInteractions2())
        _check("LightGCN", _sum_params(lgcn),
               _est_lightgcn(100, 50, 6, lgcn_hp, FP32).params_bytes)
    except Exception as exc:
        results.append(f"SKIP LightGCN: {exc}")

    # ---- CausalECLRec ----
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "causal_ecl_rec",
            Path(__file__).resolve().parent.parent / "model" / "v1" / "causal_ecl_rec.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError("spec not found")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        CausalECLRec = mod.CausalECLRec
        ce_hp = {
            "embedding_size": 16, "mlp_hidden_layers": [32], "dropout": 0.0,
            "temperature": 0.2, "gamma": 0.0, "lambda_bpr": 0.0, "reg_weight": 0.0,
            "freeze_teacher": True, "neg_samples": 1, "batch_size": 8,
            "epochs": 1, "learning_rate": 1e-3,
        }
        ce_model = CausalECLRec(ce_hp, info)
        est_bd = _est_causal_ecl_rec(100, 50, 1000, ce_hp, FP32)
        _check("CausalECLRec", _sum_params(ce_model), est_bd.params_bytes,
               _sum_buffers(ce_model), est_bd.persistent_buffers_bytes)
    except Exception as exc:
        results.append(f"SKIP CausalECLRec: {exc}")

    return results


def self_test() -> int:
    print("=" * 90)
    print("Tier A — closed-form consistency")
    print("=" * 90)
    try:
        for line in _self_test_closed_form():
            print(f"  {line}")
    except AssertionError as exc:
        print(f"  FAIL {exc}")
        return 1

    print()
    print("=" * 90)
    print("Tier B — live PyTorch comparison")
    print("=" * 90)
    try:
        for line in _self_test_live():
            print(f"  {line}")
    except AssertionError as exc:
        print(f"  FAIL {exc}")
        return 1

    print()
    print(f"Registered estimators: {len(MODEL_ESTIMATORS)} models")
    print("All checks passed.")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate RAM/VRAM for a WarpRec configuration.",
    )
    parser.add_argument("-c", "--config", type=Path,
                        help="Path to WarpRec YAML config.")
    parser.add_argument("--format", choices=("table", "json"), default="table",
                        help="Output format (default: table).")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-tests and exit.")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.config:
        parser.error("--config is required (or use --self-test)")
    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")

    output = run(args.config, args.format)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
