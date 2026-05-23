#!/usr/bin/env python3
"""
Clean single-sample inference for Flow Matching model on BraTS sample idx=11487.

Saves each image as a separate file with no axes, labels, colorbars, or text.
All images are saved at full resolution.

Output layout (inside clean_images_11487/fm/):
  fm_original_T1.png
  fm_original_T1ce.png
  fm_original_T2.png
  fm_original_FLAIR.png
  fm_recon_T1.png
  fm_recon_T1ce.png
  fm_recon_T2.png
  fm_recon_FLAIR.png
  fm_anomaly_map_T1.png
  fm_anomaly_map_T1ce.png
  fm_anomaly_map_T2.png
  fm_anomaly_map_FLAIR.png
  fm_anomaly_map_combined.png
  fm_binary_mask_T1.png
  fm_binary_mask_T1ce.png
  fm_binary_mask_T2.png
  fm_binary_mask_FLAIR.png
  fm_binary_mask_combined.png
  fm_gt_mask.png
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sure imports resolve from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer_anomaly import (
    anomaly_analysis,
    brain_mask,
    decode,
    encode,
    load_model,
    load_sample,
)

# ─── Configuration ────────────────────────────────────────────────────────────

IDX_TARGET   = 11487
CHECKPOINT   = "/mnt/apple/k66/minhdd/flow-matching-main/brats-perbatch-e11/checkpoint_epoch0011.pth"
ARCH         = "bratsv2"          # override the args.json value ("brats")
DATA_PATH    = "/mnt/apple/k66/minhdd/data/brats2021"
OUTPUT_DIR   = "/mnt/apple/k66/minhdd/clean_samples/fm_old/11487"

# Best hyperparams from grid sweep (see grid_sweep_results_ckpt_e11.md)
T_START      = 0.2
STEP_SIZE    = 0.02
CFG_SCALE    = 0.5
COMBINED_IDX = (2, 3)             # T2=2, FLAIR=3 → best combined mask

MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_clean(arr: np.ndarray, path: str, cmap: str = "gray",
               vmin: float | None = None, vmax: float | None = None) -> None:
    """Save a 2-D numpy array as a plain image (no axes, no border, no text)."""
    h, w = arr.shape
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"  saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model (arch override → bratsv2)
    cfg_model, _ = load_model(CHECKPOINT, device, arch_override=ARCH)

    # Load split and get the target entry
    split_file = "/mnt/apple/k66/minhdd/data/brats2021/preprocessed_split_old.json"
    with open(split_file) as f:
        split = json.load(f)
    entry = split["val"][IDX_TARGET]
    print(f"Processing FM for idx={IDX_TARGET}: {entry['path']}")

    # Load image (4,256,256) and GT mask (1,256,256)
    image_np, label, mask_np = load_sample(DATA_PATH, entry)
    b_mask = brain_mask(image_np)

    # Encode → latent, then decode → healthy reconstruction
    x_input = torch.from_numpy(image_np).unsqueeze(0).to(device)   # (1,4,256,256)
    x_t     = encode(cfg_model, x_input, T_START, STEP_SIZE, device)
    x_recon = decode(cfg_model, x_t, T_START, STEP_SIZE, device, CFG_SCALE)

    recon_np = x_recon.squeeze(0).cpu().numpy()   # (4,256,256)

    # Anomaly analysis
    diff_maps, binary_masks, combined_diff, combined_binary = anomaly_analysis(
        image_np, recon_np, b_mask,
        postprocess=True,
        min_component_size=50,
        border_erosion=3,
        border_overlap_thr=0.6,
        combined_idx=COMBINED_IDX,
    )

    # ── Save per-modality original images ──────────────────────────────────────
    for i, name in enumerate(MODALITY_NAMES):
        save_clean(image_np[i], os.path.join(OUTPUT_DIR, f"fm_original_{name}.png"),
                   cmap="gray", vmin=0, vmax=1)

    # ── Save per-modality reconstructions ─────────────────────────────────────
    for i, name in enumerate(MODALITY_NAMES):
        save_clean(recon_np[i], os.path.join(OUTPUT_DIR, f"fm_recon_{name}.png"),
                   cmap="gray", vmin=0, vmax=1)

    # ── Save per-modality anomaly maps (jet, normalised per-map) ──────────────
    for i, name in enumerate(MODALITY_NAMES):
        dm = diff_maps[i]
        vmax_dm = float(dm.max()) if dm.max() > 0 else 1.0
        save_clean(dm, os.path.join(OUTPUT_DIR, f"fm_anomaly_map_{name}.png"),
                   cmap="jet", vmin=0, vmax=vmax_dm)

    # Combined anomaly map
    vmax_comb = float(combined_diff.max()) if combined_diff.max() > 0 else 1.0
    save_clean(combined_diff, os.path.join(OUTPUT_DIR, "fm_anomaly_map_combined.png"),
               cmap="jet", vmin=0, vmax=vmax_comb)

    # ── Save per-modality binary masks ────────────────────────────────────────
    for i, name in enumerate(MODALITY_NAMES):
        save_clean(binary_masks[i], os.path.join(OUTPUT_DIR, f"fm_binary_mask_{name}.png"),
                   cmap="gray", vmin=0, vmax=1)

    # Combined binary mask
    save_clean(combined_binary, os.path.join(OUTPUT_DIR, "fm_binary_mask_combined.png"),
               cmap="gray", vmin=0, vmax=1)

    # ── Save GT mask ──────────────────────────────────────────────────────────
    save_clean(mask_np[0], os.path.join(OUTPUT_DIR, "fm_gt_mask.png"),
               cmap="gray", vmin=0, vmax=1)

    print(f"\n✓ Done — saved {2*4 + 5 + 5 + 1} images to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
