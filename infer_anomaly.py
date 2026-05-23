#!/usr/bin/env python3
"""
Anomaly Detection Inference for BraTS2021.

Pipeline:
  1. Reverse ODE  : encode image from t=1 → t_start  (unconditional velocity)
  2. Forward ODE  : reconstruct with CFG label=0,    t_start → t=1
  3. Anomaly map  : |input - recon| per modality, Otsu only inside brain region
  4. Combined     : max-diff across 4 modalities → Otsu → final binary mask
  5. Unhealthy    : DICE / IOU / AUROC
     Healthy      : PSNR / SSIM
  6. Visualisation: 5 rows (T1 T1ce T2 FLAIR Combined) × 5 cols
                    (Original | Reconstruction | Anomaly-jet | Binary | GT Mask)
                    Metrics annotated on each row.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from flow_matching.solver.ode_solver import ODESolver
from models.model_configs import instantiate_model
from training.eval_loop import CFGScaledModel

MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="BraTS2021 anomaly detection with reverse+forward ODE")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--arch",          type=str, default=None,
                   help="Override model architecture key in MODEL_CONFIGS "
                        "(e.g. 'bratsv2'). If unset, infer from args.json's "
                        "'dataset' field.")
    p.add_argument("--data_path",     type=str, required=True,
                   help="BraTS2021 root containing healthy/ unhealthy/ and preprocessed_split.json")
    p.add_argument("--split_file",    type=str, default=None,
                   help="Path to preprocessed_split.json. "
                        "Defaults to <data_path>/preprocessed_split.json")
    p.add_argument("--t",             type=float, default=0.6,
                   help="Encoding depth: fraction along ODE from data (1) toward noise (0).")
    p.add_argument("--step_size",     type=float, default=0.02)
    p.add_argument("--cfg_scale",     type=float, default=3.0)
    p.add_argument("--num_unhealthy", type=int,   default=50,
                   help="Cap on unhealthy samples to evaluate. -1 = all in val.")
    p.add_argument("--num_healthy",   type=int,   default=20,
                   help="Cap on healthy samples to evaluate. -1 = all in val.")
    p.add_argument("--output_dir",    type=str,   default="anomaly_results")
    p.add_argument("--device",        type=str,   default="cuda")
    p.add_argument("--seed",          type=int,   default=42)
    # CFG-Zero* (Fan et al.) — improves CFG by (1) projecting v_uncond onto
    # v_cond before mixing, and (2) skipping the first K ODE steps where the
    # learned velocity is most error-prone.
    p.add_argument("--cfg_zero_star",    action="store_true",
                   help="Enable CFG-Zero*: optimized scale + zero-init.")
    p.add_argument("--zero_init_steps",  type=int, default=1,
                   help="K — number of leading ODE steps to skip (zero-init).")
    # --encode_label : conditional reverse encoding (Wolleb-style oracle).
    #   -1 = unconditional (default, class-agnostic, deployable)
    #    0 = encode with healthy label
    #    1 = encode with unhealthy label (uses GT — oracle for unhealthy samples;
    #        for healthy samples falls back to encode_label=0 to avoid
    #        forcing them through the unhealthy distribution)
    p.add_argument("--encode_label",     type=int, default=-1, choices=[-1, 0, 1],
                   help="Reverse-encode conditioning. -1=uncond (default), "
                        "0=healthy, 1=unhealthy.")
    p.add_argument("--encode_cfg_scale", type=float, default=0.0,
                   help="CFG scale for reverse encoding (only with encode_label>=0). "
                        "0 = plain conditional. >0 = (1+w)v_cond - w·v_uncond. "
                        "With encode_label=1, pushes latent deeper into unhealthy "
                        "manifold (class-agnostic, no GT needed).")
    # Mask post-processing — denoise per-modality binary maps and build the
    # combined map as a union (no second Otsu pass on the max-diff).
    p.add_argument("--no_postprocess",      action="store_true",
                   help="Disable small-blob removal post-processing.")
    p.add_argument("--min_component_size",  type=int, default=50,
                   help="Drop connected components smaller than this (pixels).")
    p.add_argument("--hysteresis",          action="store_true",
                   help="Use hysteresis threshold (high+low percentile) instead of Otsu.")
    p.add_argument("--hyst_high_pct",       type=float, default=99.5,
                   help="High percentile for hysteresis seed threshold.")
    p.add_argument("--hyst_low_pct",        type=float, default=95.0,
                   help="Low percentile for hysteresis candidate threshold.")
    # Border-noise suppression — discard CCs that hug the brain boundary
    # (typical rim artefacts from |input - recon| at the skull/CSF edge),
    # while preserving tumours that touch the cortex because they still have
    # substantial mass inside the eroded brain interior.
    p.add_argument("--border_erosion",      type=int, default=3,
                   help="Erode brain mask by this many pixels to define a "
                        "'border zone' used for rim-noise filtering. 0 disables.")
    p.add_argument("--border_overlap_thr",  type=float, default=0.6,
                   help="A connected component is considered rim noise (and "
                        "removed) if more than this fraction of its pixels "
                        "fall inside the border zone.")
    # --best : per unhealthy sample, pick the candidate (T1 / T1ce / T2 / FLAIR
    # / combined) with the highest DICE vs GT (IoU as tie-breaker) and report
    # it as 'best'. NOTE: this is an *oracle* selection — it requires GT and
    # cannot be used at deploy time. Useful as an upper bound for benchmarking.
    p.add_argument("--best",                action="store_true",
                   help="Add a 'best' row that picks the per-sample mask with "
                        "the highest DICE vs GT (oracle selection — leaks GT).")
    p.add_argument("--combined_modalities", type=str, default="T1,T1ce,T2,FLAIR",
                   help="Comma-separated subset of {T1,T1ce,T2,FLAIR} used to "
                        "build the 'combined' mask (union). Per-modality masks "
                        "are still computed for all 4. Default: all four.")
    p.add_argument("--save_clean_idx",  type=int, default=-1,
                   help="Val-set index of a single sample for which to save "
                        "individual clean images (no axes / text / labels) into "
                        "<output_dir>/clean/<idx>/fm_*.png.  -1 disables (default).")
    p.add_argument("--target_idx",      type=str, default=None,
                   help="Comma-separated list of val-set indices to evaluate. "
                        "If set, ignores --num_unhealthy/--num_healthy.")
    p.add_argument("--split",           type=str, default="val", choices=["val", "test"],
                   help="Which split to evaluate on: 'val' (default) or 'test'.")
    p.add_argument("--clean_base_dir",  type=str, default=None,
                   help="Base directory to save clean images. If not set, "
                        "saves to <output_dir>/clean/")
    p.add_argument("--no_images",       action="store_true",
                   help="Skip all image saving (visualizations + clean images). "
                        "Faster when only metrics/CSV are needed.")
    return p.parse_args()


# ─── Data loading from .npy ───────────────────────────────────────────────────

def load_sample(data_path: str, entry: dict):
    """
    Load one val entry from preprocessed_split.json.

    Returns:
        image_np  : (4, 256, 256) float32 [0,1]  — from .npy
        label     : int  0=healthy  1=unhealthy
        mask_np   : (1, 256, 256) float32         — from _seg.npy (zeros if missing)
    """
    rel_path = entry["path"]
    label    = int(entry["label"])

    image_np = np.load(os.path.join(data_path, rel_path)).astype(np.float32)

    seg_path = os.path.join(data_path, rel_path.replace(".npy", "_seg.npy"))
    if os.path.exists(seg_path):
        mask_np = np.load(seg_path).astype(np.float32)
    else:
        mask_np = np.zeros((1, 256, 256), dtype=np.float32)

    return image_np, label, mask_np          # (4,H,W), int, (1,H,W)


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str, arch_override: str = None):
    checkpoint_path = Path(checkpoint_path)
    args_path = checkpoint_path.parent / "args.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found: {args_path}")

    with open(args_path) as f:
        train_args = json.load(f)

    if arch_override:
        arch = arch_override
    else:
        arch = "brats_healthy" if train_args.get("healthy_only", False) else train_args["dataset"]
    model = instantiate_model(
        architechture=arch,
        is_discrete=train_args.get("discrete_flow_matching", False),
        use_ema=train_args.get("use_ema", False),
    )
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded [{arch}] epoch={ckpt.get('epoch', '?')}")

    cfg_model = CFGScaledModel(model=model)
    cfg_model.to(device).eval()
    return cfg_model, train_args


# ─── Velocity wrappers ────────────────────────────────────────────────────────

class _UncondVelocity(nn.Module):
    """Unconditional velocity for the reverse-ODE encoding step.

    At inference time the true label is unknown (that is the point of anomaly
    detection), so the reverse trajectory must be class-neutral.

    Note: CFGScaledModel with cfg_scale=0.0 ignores any label passed and runs
    the model with extra={} — the UNet then substitutes padding_idx (= zero
    embedding). This is the correct unconditional behaviour.
    """
    def __init__(self, cfg_model: CFGScaledModel):
        super().__init__()
        self.cfg = cfg_model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        dummy_label = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.cfg(x=x, t=t, cfg_scale=0.0, label=dummy_label)


class _CondVelocity(nn.Module):
    """Conditional velocity with a fixed class label for reverse-ODE encoding.

    Used by --encode_label option: encoding with a specific class label
    produces a latent that more strongly encodes that class's signal.
    """
    def __init__(self, cfg_model: CFGScaledModel, label_value: int):
        super().__init__()
        self.cfg = cfg_model
        self.label_value = label_value

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        t_batch = torch.zeros(x.shape[0], device=x.device) + t
        label   = torch.full((x.shape[0],), self.label_value,
                             dtype=torch.long, device=x.device)
        with torch.cuda.amp.autocast():
            out = self.cfg.model(x, t_batch, extra={"label": label})
        return out.to(dtype=torch.float32)


class _CFGCondVelocity(nn.Module):
    """CFG-style conditional velocity for reverse-ODE encoding (class-agnostic).

        v_hat = (1 + w) * v(x, t, y=label_value) - w * v(x, t, uncond)

    With label_value=1 and w>0: pushes the reverse trajectory deeper into the
    unhealthy manifold so the latent at t_start retains tumor signal more
    strongly. The forward decode with CFG toward healthy then subtracts that
    signal more effectively.

    Class-agnostic at deploy: no GT label needed — every sample is treated as
    if it were unhealthy.
    """
    def __init__(self, cfg_model: CFGScaledModel,
                 label_value: int, cfg_scale: float):
        super().__init__()
        self.cfg = cfg_model
        self.label_value = label_value
        self.cfg_scale = cfg_scale

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        t_batch = torch.zeros(x.shape[0], device=x.device) + t
        label   = torch.full((x.shape[0],), self.label_value,
                             dtype=torch.long, device=x.device)
        with torch.cuda.amp.autocast():
            v_cond   = self.cfg.model(x, t_batch, extra={"label": label})
            v_uncond = self.cfg.model(x, t_batch, extra={})
        v_cond   = v_cond.to(dtype=torch.float32)
        v_uncond = v_uncond.to(dtype=torch.float32)
        self.cfg.nfe_counter += 2
        return (1.0 + self.cfg_scale) * v_cond - self.cfg_scale * v_uncond


class CFGZeroStarModel(nn.Module):
    """
    CFG-Zero* (Fan et al.): optimized-scale variant of classifier-free guidance.

        s*       = <v_cond, v_uncond> / (||v_uncond||² + eps)   (per-sample)
        v_hat    = (1 + w) · v_cond − w · s* · v_uncond

    Bypasses CFGScaledModel for the two model evaluations because that wrapper
    drops the label whenever cfg_scale=0.0 (eval_loop.py:106), which would
    make v_cond and v_uncond identical here. We call the underlying UNet
    directly with extra={"label": ...} for conditional and extra={} for
    unconditional — matching how training conditioning works.
    """
    def __init__(self, cfg_model: CFGScaledModel, eps: float = 1e-8):
        super().__init__()
        self.cfg = cfg_model
        self.eps = eps

    def _eval_model(self, x: torch.Tensor, t_batch: torch.Tensor,
                    extra: dict) -> torch.Tensor:
        with torch.cuda.amp.autocast():
            out = self.cfg.model(x, t_batch, extra=extra)
        return out.to(dtype=torch.float32)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cfg_scale: float, label: torch.Tensor) -> torch.Tensor:
        # If guidance is off, just defer to the regular path.
        if cfg_scale == 0.0:
            return self.cfg(x=x, t=t, cfg_scale=0.0, label=label)

        t_batch = torch.zeros(x.shape[0], device=x.device) + t
        v_cond   = self._eval_model(x, t_batch, extra={"label": label})   # conditional
        v_uncond = self._eval_model(x, t_batch, extra={})                 # unconditional
        # NFE bookkeeping (mirrors CFGScaledModel doing two forwards per step).
        self.cfg.nfe_counter += 2

        flat_c = v_cond.reshape(v_cond.shape[0], -1)
        flat_u = v_uncond.reshape(v_uncond.shape[0], -1)
        dot     = (flat_c * flat_u).sum(dim=1)
        sq_norm = (flat_u * flat_u).sum(dim=1)
        s_star  = dot / (sq_norm + self.eps)                                   # (B,)
        s_star  = s_star.view(-1, *([1] * (v_cond.ndim - 1)))                  # broadcast

        return (1.0 + cfg_scale) * v_cond - cfg_scale * s_star * v_uncond


# ─── ODE encode / decode ──────────────────────────────────────────────────────

@torch.no_grad()
def encode(cfg_model: CFGScaledModel,
           x_image: torch.Tensor,
           t_start: float,
           step_size: float,
           device: str,
           encode_label: int = -1,
           encode_cfg_scale: float = 0.0) -> torch.Tensor:
    """
    Reverse ODE: data (t=1) → latent at t_start.

    encode_label = -1  : unconditional encoding (default; class-agnostic).
    encode_label =  0  : conditional on healthy.
    encode_label =  1  : conditional on unhealthy.

    encode_cfg_scale > 0 (with encode_label >= 0): use CFG-style velocity in the
    reverse pass: v = (1+w)·v_cond - w·v_uncond. With encode_label=1, this
    pushes the latent further into the unhealthy manifold so the forward decode
    toward healthy "subtracts" tumor signal more effectively. Class-agnostic.

    encode_cfg_scale = 0: plain conditional (or unconditional if label=-1).

    Returns x_t in [-1, 1].
    """
    x = x_image.to(device)
    x_scaled = x * 2.0 - 1.0                          # [0,1] → [-1,1]
    if encode_label < 0:
        wrapper = _UncondVelocity(cfg_model)
    elif encode_cfg_scale > 0.0:
        wrapper = _CFGCondVelocity(cfg_model, label_value=encode_label,
                                   cfg_scale=encode_cfg_scale)
    else:
        wrapper = _CondVelocity(cfg_model, label_value=encode_label)
    solver  = ODESolver(velocity_model=wrapper)
    # descending time_grid → torchdiffeq integrates backward
    time_grid = torch.tensor([1.0, t_start], device=device)
    x_t = solver.sample(x_init=x_scaled, time_grid=time_grid,
                        step_size=step_size, method="euler")
    return x_t                                         # (B,4,H,W) in [-1,1]


@torch.no_grad()
def decode(cfg_model: CFGScaledModel,
           x_t: torch.Tensor,
           t_start: float,
           step_size: float,
           device: str,
           cfg_scale: float,
           cfg_zero_star: bool = False,
           zero_init_steps: int = 0) -> torch.Tensor:
    """
    Forward ODE: latent at t_start → healthy reconstruction (t=1).
    Uses CFG with label=0 (healthy).
    Returns x_recon in [0, 1].

    CFG-Zero* options:
      - cfg_zero_star: use optimized-scale CFG instead of vanilla.
      - zero_init_steps (K): skip the first K Euler steps (x stays put).
        Implemented by advancing the integration start to t_start + K·step_size,
        which is exactly equivalent to taking K "do nothing" steps.
    """
    x_t = x_t.to(device)

    velocity = CFGZeroStarModel(cfg_model) if cfg_zero_star else cfg_model
    solver   = ODESolver(velocity_model=velocity)

    # Zero-init: advance the start time so the first K steps are no-ops on x_t.
    # Clamp to leave at least one real step before t=1.
    t_eff = t_start
    if cfg_zero_star and zero_init_steps > 0:
        t_eff = min(t_start + zero_init_steps * step_size, 1.0 - step_size)
        t_eff = max(t_eff, t_start)   # safety: never go backwards

    if t_eff >= 1.0:
        # Nothing left to integrate — return x_t directly.
        return torch.clamp(x_t * 0.5 + 0.5, 0.0, 1.0)

    time_grid = torch.tensor([t_eff, 1.0], device=device)
    healthy_label = torch.zeros(x_t.shape[0], dtype=torch.long, device=device)
    x_recon_scaled = solver.sample(
        x_init=x_t,
        time_grid=time_grid,
        step_size=step_size,
        method="euler",
        label=healthy_label,
        cfg_scale=cfg_scale,
    )
    return torch.clamp(x_recon_scaled * 0.5 + 0.5, 0.0, 1.0)   # (B,4,H,W)


# ─── Brain mask ───────────────────────────────────────────────────────────────

def brain_mask(image_np: np.ndarray) -> np.ndarray:
    """(H,W) bool: True where at least one modality is non-zero."""
    return np.any(image_np > 1e-4, axis=0)


# ─── Anomaly analysis ─────────────────────────────────────────────────────────

def _otsu_on_region(diff_2d: np.ndarray, region: np.ndarray):
    """Compute Otsu threshold using only pixels inside `region` (bool H×W).
    Returns (threshold_value, binary_mask_HW)."""
    pixels = diff_2d[region]
    if pixels.size == 0 or pixels.max() - pixels.min() < 1e-8:
        return 0.0, np.zeros_like(diff_2d, dtype=np.float32)
    pmax = pixels.max()
    pixels_u8 = (pixels / pmax * 255.0).astype(np.uint8)
    thresh_u8, _ = cv2.threshold(pixels_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = float(thresh_u8) / 255.0 * pmax
    binary = np.zeros_like(diff_2d, dtype=np.float32)
    binary[region & (diff_2d > thresh)] = 1.0
    return thresh, binary


def _hysteresis_on_region(diff_2d: np.ndarray, region: np.ndarray,
                          high_pct: float = 99.5,
                          low_pct: float = 95.0):
    """Hysteresis threshold inside `region`: a pixel is positive iff its diff
    exceeds the LOW threshold AND its connected component contains at least one
    pixel above the HIGH threshold. Filters scattered noise effectively.
    Returns (high_thresh, binary_mask_HW)."""
    pixels = diff_2d[region]
    if pixels.size == 0 or pixels.max() - pixels.min() < 1e-8:
        return 0.0, np.zeros_like(diff_2d, dtype=np.float32)

    t_high = float(np.percentile(pixels, high_pct))
    t_low  = float(np.percentile(pixels, low_pct))
    # Guard against degenerate thresholds (e.g. when most in-region pixels are
    # zero so the percentile collapses to 0). Without this, every non-zero
    # pixel would be a "seed" and the mask would explode.
    eps = 1e-8
    if t_high <= eps or t_low <= eps:
        return t_high, np.zeros_like(diff_2d, dtype=np.float32)
    if t_low > t_high:
        t_low = t_high

    high_mask = (region & (diff_2d >= t_high)).astype(np.uint8)
    low_mask  = (region & (diff_2d >= t_low)).astype(np.uint8)
    if low_mask.sum() == 0 or high_mask.sum() == 0:
        return t_high, np.zeros_like(diff_2d, dtype=np.float32)

    n_labels, labels = cv2.connectedComponents(low_mask, connectivity=8)
    keep = np.zeros(n_labels, dtype=bool)
    seeded_labels = labels[high_mask.astype(bool)]
    keep[np.unique(seeded_labels)] = True
    keep[0] = False
    binary = keep[labels].astype(np.float32)
    return t_high, binary


def _clean_mask(binary: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_size pixels."""
    if binary.sum() == 0:
        return binary.astype(np.float32)
    bin_u8 = (binary > 0).astype(np.uint8)
    if min_size > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
        cleaned = np.zeros_like(bin_u8)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == lbl] = 1
        bin_u8 = cleaned
    return bin_u8.astype(np.float32)


def _brain_border_zone(b_mask: np.ndarray, erosion: int) -> np.ndarray:
    """Return the thin rim along the inside of the brain mask.

    border_zone = b_mask AND NOT eroded(b_mask). Pixels here are within
    `erosion` pixels of the brain boundary — exactly where reconstruction
    artefacts at the skull/CSF interface tend to appear.
    """
    if erosion <= 0:
        return np.zeros_like(b_mask, dtype=bool)
    b_u8 = b_mask.astype(np.uint8)
    k = 2 * erosion + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(b_u8, kernel, iterations=1).astype(bool)
    return b_mask & (~eroded)


def _remove_border_noise(binary: np.ndarray,
                         border_zone: np.ndarray,
                         overlap_thr: float = 0.6) -> np.ndarray:
    """Drop connected components that are mostly tucked into the brain rim.

    Why: |input - recon| spikes along the cortical edge produce thin
    crescent-shaped artefacts that hug the boundary. Tumours touching the
    cortex still have a body extending inwards, so their overlap with the
    border zone is well below 1. Removing CCs whose border-zone overlap
    exceeds `overlap_thr` strips rim noise without erasing peripheral
    tumours.
    """
    if binary.sum() == 0 or border_zone.sum() == 0:
        return binary.astype(np.float32)
    bin_u8 = (binary > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    kept = np.zeros_like(bin_u8)
    for lbl in range(1, n_labels):
        cc = labels == lbl
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area == 0:
            continue
        border_frac = float((cc & border_zone).sum()) / float(area)
        if border_frac <= overlap_thr:
            kept[cc] = 1
    return kept.astype(np.float32)


def anomaly_analysis(input_np: np.ndarray,
                     recon_np: np.ndarray,
                     b_mask: np.ndarray,
                     postprocess: bool = True,
                     min_component_size: int = 50,
                     use_hysteresis: bool = False,
                     high_pct: float = 99.5,
                     low_pct: float = 95.0,
                     border_erosion: int = 0,
                     border_overlap_thr: float = 0.6,
                     combined_idx: tuple = (0, 1, 2, 3)):
    """
    Args:
        input_np            : (4,H,W) float32 [0,1]
        recon_np            : (4,H,W) float32 [0,1]
        b_mask              : (H,W) bool brain mask
        postprocess         : if True, remove small connected components.
        min_component_size  : connected components smaller than this (in pixels)
                              are discarded.
        use_hysteresis      : if True, use hysteresis threshold instead of Otsu.
        high_pct, low_pct   : percentiles for hysteresis (only if use_hysteresis).
    Returns:
        diff_maps      : (4,H,W) |input - recon|
        binary_masks   : (4,H,W) per-modality binary anomaly maps (post-processed)
        combined_diff  : (H,W)   max over modalities (for visualisation / AUROC)
        combined_binary: (H,W)   UNION of the 4 cleaned per-modality masks.
    """
    diff_maps    = np.abs(input_np - recon_np) * b_mask[None, ...]   # (4,H,W)
    binary_masks = np.zeros_like(diff_maps)
    border_zone  = _brain_border_zone(b_mask, border_erosion)
    for i in range(4):
        if use_hysteresis:
            _, raw = _hysteresis_on_region(diff_maps[i], b_mask,
                                           high_pct=high_pct, low_pct=low_pct)
        else:
            _, raw = _otsu_on_region(diff_maps[i], b_mask)
        if border_erosion > 0:
            raw = _remove_border_noise(raw, border_zone,
                                       overlap_thr=border_overlap_thr)
        if postprocess:
            raw = _clean_mask(raw, min_size=min_component_size)
        binary_masks[i] = raw

    idx = list(combined_idx)
    combined_diff   = diff_maps[idx].max(axis=0)
    combined_binary = (binary_masks[idx].sum(axis=0) > 0).astype(np.float32)
    return diff_maps, binary_masks, combined_diff, combined_binary


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _dice_iou(pred: np.ndarray, gt: np.ndarray):
    inter = (pred * gt).sum()
    total = pred.sum() + gt.sum()
    union = ((pred + gt) > 0).sum()
    dice  = float(2 * inter / total) if total > 0 else 1.0
    iou   = float(inter / union)     if union  > 0 else 1.0
    return dice, iou


def _auroc(score_2d: np.ndarray, gt_2d: np.ndarray, b_mask: np.ndarray):
    scores = score_2d[b_mask].ravel()
    labels = gt_2d[b_mask].ravel().astype(int)
    if labels.max() == 0 or labels.min() == 1:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def compute_metrics_unhealthy(binary_masks: np.ndarray,
                              combined_binary: np.ndarray,
                              gt_mask_np: np.ndarray,
                              diff_maps: np.ndarray,
                              combined_diff: np.ndarray,
                              b_mask: np.ndarray,
                              add_best: bool = False) -> dict:
    gt = (gt_mask_np.squeeze(0) > 0).astype(np.float32)   # (H,W)
    results = {}
    for i, name in enumerate(MODALITY_NAMES):
        d, iou = _dice_iou(binary_masks[i], gt)
        auc    = _auroc(diff_maps[i], gt, b_mask)
        results[name] = {"dice": d, "iou": iou, "auroc": auc}
    d, iou = _dice_iou(combined_binary, gt)
    auc    = _auroc(combined_diff, gt, b_mask)
    results["combined"] = {"dice": d, "iou": iou, "auroc": auc}

    if add_best:
        # Oracle selection per metric: independently pick, for each of dice /
        # iou / auroc, the candidate (per-modality or combined) that scores
        # highest on that metric for THIS sample. The aggregate row "best" at
        # the end of the run is the mean of these per-sample maxima — i.e.
        # the upper bound achievable if you could route each sample to its
        # best-performing candidate for each metric independently.
        candidates = MODALITY_NAMES + ["combined"]
        def _best_key(metric: str) -> str:
            return max(candidates,
                       key=lambda k: (results[k][metric]
                                      if not np.isnan(results[k][metric])
                                      else -np.inf))
        dice_key  = _best_key("dice")
        iou_key   = _best_key("iou")
        auroc_key = _best_key("auroc")
        results["best"] = {
            "dice":   results[dice_key]["dice"],
            "iou":    results[iou_key]["iou"],
            "auroc":  results[auroc_key]["auroc"],
            "source": dice_key,
            "source_iou":   iou_key,
            "source_auroc": auroc_key,
        }
    return results


def compute_metrics_healthy(input_np: np.ndarray, recon_np: np.ndarray) -> dict:
    results = {}
    for i, name in enumerate(MODALITY_NAMES):
        p = float(peak_signal_noise_ratio(input_np[i], recon_np[i], data_range=1.0))
        s = float(structural_similarity(input_np[i], recon_np[i], data_range=1.0))
        results[name] = {"psnr": p, "ssim": s}
    avg_in = input_np.mean(axis=0)
    avg_re = recon_np.mean(axis=0)
    p = float(peak_signal_noise_ratio(avg_in, avg_re, data_range=1.0))
    s = float(structural_similarity(avg_in, avg_re, data_range=1.0))
    results["combined"] = {"psnr": p, "ssim": s}
    return results


def _metric_str(label_int: int, mdict: dict, key: str) -> str:
    m = mdict.get(key, {})
    if label_int == 1:
        auroc_v = m.get("auroc", float("nan"))
        auroc_s = f"{auroc_v:.3f}" if not (isinstance(auroc_v, float) and np.isnan(auroc_v)) else "nan"
        return f"DICE={m.get('dice', 0):.3f}  IOU={m.get('iou', 0):.3f}  AUROC={auroc_s}"
    else:
        return f"PSNR={m.get('psnr', 0):.2f}dB  SSIM={m.get('ssim', 0):.3f}"


# ─── Visualisation ────────────────────────────────────────────────────────────

_COL_TITLES = ["Original", "Reconstruction", "Anomaly Map", "Binary (Otsu)", "GT Mask"]
_ROW_KEYS   = MODALITY_NAMES + ["combined"]


def visualize_sample(input_np, recon_np, diff_maps, binary_masks,
                     combined_diff, combined_binary, gt_mask_np,
                     label_int, mdict, idx, save_path,
                     elapsed: float = 0.0):
    """
    Grid: 5 rows × 5 cols.
    Rows  : T1 | T1ce | T2 | FLAIR | Combined
    Cols  : Original | Reconstruction | Anomaly-jet | Binary | GT Mask
    Metrics annotated as text overlay on the first column of each row.
    """
    n_rows, n_cols = 5, 5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.2, n_rows * 4.0 + 0.6),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.04})

    gt_2d = gt_mask_np.squeeze(0).astype(np.float32)          # (H,W)

    # Build per-row data: (orig_2d, recon_2d, diff_2d, binary_2d)
    row_data = [(input_np[i], recon_np[i], diff_maps[i], binary_masks[i])
                for i in range(4)]
    row_data.append((input_np.mean(0), recon_np.mean(0), combined_diff, combined_binary))

    for r, (row_key, (orig, recon, diff, binary)) in enumerate(zip(_ROW_KEYS, row_data)):
        vmax = float(diff.max()) if diff.max() > 0 else 1.0

        axes[r, 0].imshow(orig,   cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(recon,  cmap="gray", vmin=0, vmax=1)
        im_diff = axes[r, 2].imshow(diff, cmap="jet", vmin=0, vmax=vmax)
        axes[r, 3].imshow(binary, cmap="gray", vmin=0, vmax=1)
        axes[r, 4].imshow(gt_2d,  cmap="gray", vmin=0, vmax=1)

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)

        # Row name
        axes[r, 0].set_ylabel(row_key.upper(), fontsize=9, fontweight="bold",
                               rotation=0, labelpad=40, va="center")
        axes[r, 0].yaxis.set_visible(True)

        # Metric overlay on the diff cell (col 2)
        ms = _metric_str(label_int, mdict, row_key)
        axes[r, 2].text(0.01, 0.01, ms, transform=axes[r, 2].transAxes,
                        fontsize=6.5, va="bottom", ha="left", color="white",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="black", alpha=0.55))

    # Column headers (only top row)
    for c, title in enumerate(_COL_TITLES):
        axes[0, c].set_title(title, fontsize=9, fontweight="bold", pad=4)

    label_str = "Unhealthy" if label_int == 1 else "Healthy"
    fig.suptitle(f"Sample idx={idx}  |  {label_str}  |  infer_time={elapsed:.2f}s",
                 fontsize=12, fontweight="bold")

    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_clean_images(input_np, recon_np, diff_maps, binary_masks,
                      combined_diff, combined_binary, gt_mask_np,
                      clean_dir: Path) -> None:
    """
    Save individual images for a single sample with NO axes, text, or labels.
    Each image is written pixel-exact (native array size) via cv2.imwrite.

    Files written (prefix = fm_):
      fm_original_{T1,T1ce,T2,FLAIR}.png
      fm_recon_{T1,T1ce,T2,FLAIR}.png
      fm_anomaly_map_{T1,T1ce,T2,FLAIR}.png  (jet colormap)
      fm_anomaly_map_combined.png              (jet)
      fm_binary_mask_{T1,T1ce,T2,FLAIR}.png
      fm_binary_mask_combined.png
      fm_gt_mask.png
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    gt_2d = gt_mask_np.squeeze(0).astype(np.float32)

    def _imsave(arr, path, cmap, vmin=None, vmax=None):
        lo = float(arr.min()) if vmin is None else vmin
        hi = float(arr.max()) if vmax is None else vmax
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((arr - lo) / (hi - lo), 0, 1)
        if cmap == "gray":
            img_u8 = (norm * 255).astype(np.uint8)
            cv2.imwrite(str(path), img_u8)
        else:
            import matplotlib.cm as mcm
            rgba = (mcm.get_cmap(cmap)(norm) * 255).astype(np.uint8)
            cv2.imwrite(str(path), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))
        print(f"    clean → {path}")

    for i, name in enumerate(MODALITY_NAMES):
        _imsave(input_np[i],     clean_dir / f"fm_original_{name}.png",    "gray", 0, 1)
        _imsave(recon_np[i],     clean_dir / f"fm_recon_{name}.png",       "gray", 0, 1)
        dm = diff_maps[i]
        _imsave(dm,              clean_dir / f"fm_anomaly_map_{name}.png",  "jet",
                0, float(dm.max()) if dm.max() > 0 else 1.0)
        _imsave(binary_masks[i], clean_dir / f"fm_binary_mask_{name}.png", "gray", 0, 1)

    vmax_c = float(combined_diff.max()) if combined_diff.max() > 0 else 1.0
    _imsave(combined_diff,   clean_dir / "fm_anomaly_map_combined.png",  "jet",  0, vmax_c)
    _imsave(combined_binary, clean_dir / "fm_binary_mask_combined.png",  "gray", 0, 1)
    _imsave(gt_2d,           clean_dir / "fm_gt_mask.png",               "gray", 0, 1)
    # combined input/recon: mean of T2+FLAIR (indices 2,3)
    _imsave(input_np[[2,3]].mean(0),  clean_dir / "fm_original_combined.png", "gray", 0, 1)
    _imsave(recon_np[[2,3]].mean(0),  clean_dir / "fm_recon_combined.png",    "gray", 0, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("FM: Entering main()", flush=True)
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device        = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    split_file    = args.split_file    or os.path.join(args.data_path, "preprocessed_split.json")

    name_to_idx = {n: i for i, n in enumerate(MODALITY_NAMES)}
    requested = [m.strip() for m in args.combined_modalities.split(",") if m.strip()]
    bad = [m for m in requested if m not in name_to_idx]
    if bad:
        raise ValueError(f"Unknown modality names {bad}. Allowed: {MODALITY_NAMES}")
    combined_idx = tuple(name_to_idx[m] for m in requested)
    if not combined_idx:
        raise ValueError("--combined_modalities must list at least one modality")
    print(f"Combined mask uses modalities: {requested} (indices {combined_idx})")

    cfg_model, train_args = load_model(args.checkpoint, str(device), arch_override=args.arch)

    with open(split_file) as f:
        split = json.load(f)
    split_key = getattr(args, "split", "val")
    val_entries = split[split_key]
    print(f"{split_key.capitalize()} set: {len(val_entries)} slices  (from {split_file})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_uh_total = sum(1 for e in val_entries if e["label"] == 1)
    n_h_total  = sum(1 for e in val_entries if e["label"] == 0)

    if args.target_idx:
        order = [int(i.strip()) for i in args.target_idx.split(",") if i.strip()]
        cap_uh = n_uh_total
        cap_h  = n_h_total
    else:
        order = np.random.permutation(len(val_entries))
        cap_uh = n_uh_total if args.num_unhealthy < 0 else args.num_unhealthy
        cap_h = n_h_total if args.num_healthy < 0 else args.num_healthy

    unhealthy_count = healthy_count = 0
    all_uh_metrics, all_h_metrics, all_times = [], [], []

    csv_file = open(output_dir / "metrics.csv", "w")
    csv_file.write("idx,label,modality,dice,iou,auroc,psnr,ssim,time_s\n")

    print(f"Targeting {cap_uh} unhealthy + {cap_h} healthy samples "
          f"({split_key} pool: {n_uh_total} unhealthy, {n_h_total} healthy)")

    pbar = tqdm(total=1, dynamic_ncols=True)
    # Use the 'combined' row (union over T2/FLAIR / etc) for the live display.
    # The final summary still reports every row including 'best'.
    live_key = "combined"

    def _running_mean(vals):
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(clean)) if clean else float("nan")

    def _running_means():
        """Return current running means for both unhealthy and healthy pools."""
        out = {"uh": None, "h": None}
        if all_uh_metrics:
            out["uh"] = {
                "dice":  _running_mean([m[live_key]["dice"]  for m in all_uh_metrics if live_key in m]),
                "iou":   _running_mean([m[live_key]["iou"]   for m in all_uh_metrics if live_key in m]),
                "auroc": _running_mean([m[live_key]["auroc"] for m in all_uh_metrics if live_key in m]),
                "n":     unhealthy_count,
            }
        if all_h_metrics:
            out["h"] = {
                "psnr": _running_mean([m["combined"]["psnr"] for m in all_h_metrics if "combined" in m]),
                "ssim": _running_mean([m["combined"]["ssim"] for m in all_h_metrics if "combined" in m]),
                "n":    healthy_count,
            }
        return out

    def _update_postfix():
        means = _running_means()
        post = {}
        if means["uh"]:
            post["dice"]  = f"{means['uh']['dice']:.3f}"
            post["auroc"] = f"{means['uh']['auroc']:.3f}"
        if means["h"]:
            post["psnr"] = f"{means['h']['psnr']:.2f}"
            post["ssim"] = f"{means['h']['ssim']:.3f}"
        pbar.set_postfix(post, refresh=True)

    for raw_idx in order:
        print(f"Processing raw_idx={raw_idx} (unhealthy_count={unhealthy_count}, cap_uh={cap_uh})", flush=True)
        if unhealthy_count >= cap_uh and healthy_count >= cap_h:
            break
        
        idx   = int(raw_idx)
        entry = val_entries[idx]
        try:
            image_np, label_int, gt_np = load_sample(
                args.data_path, entry)                  # (4,H,W), int, (1,H,W)

            if label_int == 1 and unhealthy_count >= cap_uh:
                continue
            if label_int == 0 and healthy_count >= cap_h:
                continue

            t0 = time.time()

            x_in = torch.from_numpy(image_np).unsqueeze(0).to(device)  # (1,4,H,W)

            # 1. Reverse ODE: encode (t=1 → t_start)
            x_t    = encode(cfg_model, x_in, args.t, args.step_size, str(device),
                            encode_label=args.encode_label,
                            encode_cfg_scale=args.encode_cfg_scale)

            # 2. Forward ODE: decode healthy (t_start → t=1)
            x_recon = decode(cfg_model, x_t, args.t, args.step_size, str(device),
                             args.cfg_scale,
                             cfg_zero_star=args.cfg_zero_star,
                             zero_init_steps=args.zero_init_steps)

            elapsed = time.time() - t0

            recon_np = x_recon[0].cpu().numpy()         # (4,H,W) [0,1]
            input_np = image_np                          # (4,H,W) [0,1]

            # Force background to 0 for both input and recon. The model can
            # hallucinate non-zero intensities outside the brain (especially
            # near the skull edge), and that leaks into |input - recon| as
            # spurious anomalies. The brain mask is defined from the INPUT
            # (where any modality is non-zero) — the assumption is that any
            # signal outside the input's brain mask is non-anatomical noise
            # and must be zeroed before computing the diff.
            b_mask   = brain_mask(input_np)
            input_np = input_np * b_mask[None, ...]
            recon_np = recon_np * b_mask[None, ...]
            diff_maps, binary_masks, combined_diff, combined_binary \
                                                            = anomaly_analysis(
                                                                  input_np, recon_np, b_mask,
                                                                  postprocess=not args.no_postprocess,
                                                                  min_component_size=args.min_component_size,
                                                                  use_hysteresis=args.hysteresis,
                                                                  high_pct=args.hyst_high_pct,
                                                                  low_pct=args.hyst_low_pct,
                                                                  border_erosion=args.border_erosion,
                                                                  border_overlap_thr=args.border_overlap_thr,
                                                                  combined_idx=combined_idx)

            if label_int == 1:
                mdict = compute_metrics_unhealthy(
                    binary_masks, combined_binary, gt_np,
                    diff_maps, combined_diff, b_mask,
                    add_best=args.best)
                all_uh_metrics.append(mdict)
                row_keys = _ROW_KEYS + (["best"] if args.best else [])
                for key in row_keys:
                    m = mdict.get(key, {})
                    csv_file.write(
                        f"{idx},{label_int},{key},"
                        f"{m.get('dice','')},{m.get('iou','')},{m.get('auroc','')}"
                        f",,,{elapsed:.3f}\n")
                unhealthy_count += 1
            else:
                mdict = compute_metrics_healthy(input_np, recon_np)
                all_h_metrics.append(mdict)
                for key in _ROW_KEYS:
                    m = mdict.get(key, {})
                    csv_file.write(
                        f"{idx},{label_int},{key}"
                        f",,,,"
                        f"{m.get('psnr','')},{m.get('ssim','')},{elapsed:.3f}\n")
                healthy_count += 1

            csv_file.flush()

            tag = "unhealthy" if label_int == 1 else "healthy"
            if not getattr(args, "no_images", False) and args.save_clean_idx != idx:
                visualize_sample(
                    input_np, recon_np, diff_maps, binary_masks,
                    combined_diff, combined_binary, gt_np,
                    label_int, mdict, idx,
                    output_dir / f"sample_{idx}_{tag}.png",
                    elapsed=elapsed,
                )
            # ── Clean individual images for the target sample ──────────────
            if not getattr(args, "no_images", False) and args.save_clean_idx >= 0 and idx == args.save_clean_idx:
                print(f"  [clean] Saving individual images for idx={idx}")
                base_d = Path(args.clean_base_dir) if args.clean_base_dir else (output_dir / "clean")
                save_clean_images(
                    input_np, recon_np, diff_maps, binary_masks,
                    combined_diff, combined_binary, gt_np,
                    base_d / str(idx),
                )
            # Per-sample one-liner: 'combined' row metric for THIS sample +
            # running mean for the matching pool (UH or H — not both).
            means = _running_means()
            if label_int == 1:
                m = mdict.get("combined", {})
                uh = means["uh"]
                running = (
                    f"mean(n={uh['n']}) DICE={uh['dice']:.3f} AUROC={uh['auroc']:.3f}"
                    if uh else "mean(n=0)"
                )
                this_sample = (
                    f"DICE={m.get('dice', float('nan')):.3f} "
                    f"AUROC={m.get('auroc', float('nan')):.3f}"
                )
            else:
                m = mdict.get("combined", {})
                h = means["h"]
                running = (
                    f"mean(n={h['n']}) PSNR={h['psnr']:.2f} SSIM={h['ssim']:.3f}"
                    if h else "mean(n=0)"
                )
                this_sample = (
                    f"PSNR={m.get('psnr', float('nan')):.2f} "
                    f"SSIM={m.get('ssim', float('nan')):.3f}"
                )
            all_times.append(elapsed)
            pbar.update(1)
            _update_postfix()
            tqdm.write(
                f"[idx={idx:>5}] {tag:9s}  {this_sample}  t={elapsed:.2f}s  || {running}"
            )

        except Exception as e:
            import traceback
            print(f"\nError at idx={idx}: {e}")
            traceback.print_exc()
            continue

    pbar.close()
    csv_file.close()

    # ── Summary ──────────────────────────────────────────────────────────────

    def _nan_mean(vals):
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return np.mean(clean) if clean else float("nan")

    cfg_tag = (f"  cfg_zero_star=True  zero_init_steps={args.zero_init_steps}"
               if args.cfg_zero_star else "")
    lines = [f"t={args.t}  step={args.step_size}  cfg_scale={args.cfg_scale}{cfg_tag}\n"]

    if all_uh_metrics:
        lines.append("\n=== Unhealthy (DICE / IOU / AUROC) ===")
        print(lines[-1])
        uh_keys = _ROW_KEYS + (["best"] if args.best else [])
        per_key_means = {}
        for key in uh_keys:
            dice_v  = _nan_mean([m[key]["dice"]  for m in all_uh_metrics if key in m])
            iou_v   = _nan_mean([m[key]["iou"]   for m in all_uh_metrics if key in m])
            auroc_v = _nan_mean([m[key]["auroc"] for m in all_uh_metrics if key in m])
            per_key_means[key] = (dice_v, iou_v, auroc_v)
            s = f"  {key:10s}  DICE={dice_v:.4f}  IOU={iou_v:.4f}  AUROC={auroc_v:.4f}"
            lines.append(s); print(s)
        if args.best:
            # Show which modality was picked most often per metric.
            from collections import Counter
            for src_field, label in [("source", "dice"),
                                     ("source_iou", "iou"),
                                     ("source_auroc", "auroc")]:
                srcs = Counter(m["best"][src_field] for m in all_uh_metrics
                               if "best" in m and src_field in m["best"])
                picks = "  ".join(f"{k}={v}" for k, v in srcs.most_common())
                s = f"  best.{label} picks: {picks}"
                lines.append(s); print(s)
        # Best across all keys (per metric independently)
        best_dice_key  = max(per_key_means, key=lambda k: per_key_means[k][0])
        best_iou_key   = max(per_key_means, key=lambda k: per_key_means[k][1])
        best_auroc_key = max(per_key_means, key=lambda k: per_key_means[k][2])
        bd, _, _   = per_key_means[best_dice_key]
        _, bi, _   = per_key_means[best_iou_key]
        _, _, ba   = per_key_means[best_auroc_key]
        s = (f"\n  BEST        DICE={bd:.4f} ({best_dice_key})"
             f"  IOU={bi:.4f} ({best_iou_key})"
             f"  AUROC={ba:.4f} ({best_auroc_key})")
        lines.append(s); print(s)

    if all_h_metrics:
        lines.append("\n=== Healthy (PSNR / SSIM) ===")
        print(lines[-1])
        for key in _ROW_KEYS:
            psnr_v = _nan_mean([m[key]["psnr"] for m in all_h_metrics if key in m])
            ssim_v = _nan_mean([m[key]["ssim"] for m in all_h_metrics if key in m])
            s = f"  {key:10s}  PSNR={psnr_v:.2f}dB  SSIM={ssim_v:.4f}"
            lines.append(s); print(s)

    if all_times:
        t_arr = np.array(all_times)
        lines.append("\n=== Inference time per sample (seconds) ===")
        print(lines[-1])
        s = (f"  n={len(t_arr)}  mean={t_arr.mean():.2f}  median={np.median(t_arr):.2f}"
             f"  min={t_arr.min():.2f}  max={t_arr.max():.2f}  total={t_arr.sum():.1f}s")
        lines.append(s); print(s)

    with open(output_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
