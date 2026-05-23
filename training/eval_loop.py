# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import math
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from training import distributed_mode
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def _save_snapshot_with_labels(
    images: torch.Tensor,
    labels: torch.Tensor,
    save_path,
    max_images: int = 64,
):
    """Save a grid of generated images with their class labels displayed above each image."""
    num_images = min(images.shape[0], max_images)
    ncols = min(8, num_images)
    nrows = math.ceil(num_images / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.8))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = axes[i, j]
            ax.axis("off")
            if idx < num_images:
                img = images[idx].detach().cpu()
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                else:
                    img = img.permute(1, 2, 0).numpy()
                    ax.imshow(img)
                label_val = labels[idx].item()
                ax.set_title(f"Label: {label_val}", fontsize=9, pad=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved snapshot with labels to {save_path}")


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        # Check if model is unconditional (num_classes is None)
        inner = module.model if isinstance(module, EMA) else module
        is_unconditional = getattr(inner, 'num_classes', None) is None

        if is_unconditional or cfg_scale == 0.0:
            # Unconditional model or no CFG: just forward with no label
            with torch.amp.autocast("cuda"):
                result = self.model(x, t, extra={})
        elif cfg_scale != 0.0:
            with torch.amp.autocast("cuda"):
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.amp.autocast("cuda"):
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def preprocess_for_fid(x):
    """
    Preprocesses images for FID calculation.
    Adapts various channel counts to 3 (Inception requirement).
    - If 1 channel: Repeat to 3.
    - If > 3 channels (e.g. 4): Average to 1 channel, then repeat to 3.
      (This allows all 4 channels to contribute to the score, vs dropping one).
    - If 3 channels: Keep as is.
    """
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        # Average all channels to 1, then repeat to 3
        # This preserves info from all channels (mixed) rather than dropping some.
        return x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    return x

def preprocess_for_snapshot(x):
    """
    Preprocesses images for visual snapshot.
    User requested: "use just 1 channel for showing".
    """
    if x.shape[1] > 1:
        # Take the first channel (e.g. T1 in BraTS)
        return x[:, 0:1, ...]
    return x

def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        solver = MixtureDiscreteEulerSolver(
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,
            source_distribution_p=p,
        )
    else:
        solver = ODESolver(velocity_model=cfg_scaled_model)
        ode_opts = args.ode_options

    fid_metric = (
        FrechetInceptionDistance(normalize=True).to(device=device, non_blocking=True)
        if args.compute_fid
        else None
    )

    num_synthetic = 0
    total_generation_time = 0.0
    total_generated_samples = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if args.compute_fid:
            samples_fid = preprocess_for_fid(samples)
            fid_metric.update(samples_fid, real=True)

        if num_synthetic < fid_samples:
            cfg_scaled_model.reset_nfe_counter()
            if args.discrete_flow_matching:
                # Discrete sampling
                x_0 = (
                    torch.zeros(samples.shape, dtype=torch.long, device=device)
                    + MASK_TOKEN
                )
                if args.sym_func:
                    sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
                else:
                    sym = args.sym
                if args.sampling_dtype == "float32":
                    dtype = torch.float32
                elif args.sampling_dtype == "float64":
                    dtype = torch.float64

                synthetic_samples = solver.sample(
                    x_init=x_0,
                    step_size=1.0 / args.discrete_fm_steps,
                    verbose=False,
                    div_free=sym,
                    dtype_categorical=dtype,
                    label=labels,
                    cfg_scale=args.cfg_scale,
                )
            else:
                # Continuous sampling
                x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)

                if args.edm_schedule:
                    time_grid = get_time_discretization(nfes=ode_opts["nfe"])
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                start_time = time.time()
                synthetic_samples = solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    method=args.ode_method,
                    return_intermediates=False,
                    atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                    rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                    step_size=ode_opts["step_size"]
                    if "step_size" in ode_opts
                    else None,
                    label=labels,
                    cfg_scale=args.cfg_scale,
                )
                end_time = time.time()
                
                batch_time = end_time - start_time
                total_generation_time += batch_time
                total_generated_samples += synthetic_samples.shape[0]

                # Scaling to [0, 1] from [-1, 1]
                synthetic_samples = torch.clamp(
                    synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
                )
                synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            logger.info(
                f"{samples.shape[0]} samples generated in {cfg_scaled_model.get_nfe()} evaluations."
            )
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            
            if args.compute_fid:
                synthetic_fid = preprocess_for_fid(synthetic_samples)
                fid_metric.update(synthetic_fid, real=False)
            
            num_synthetic += synthetic_samples.shape[0]
            if not snapshots_saved and args.output_dir:
                # Use preprocessed (1-channel) samples for visualization
                synthetic_snap = preprocess_for_snapshot(synthetic_samples)
                _save_snapshot_with_labels(
                    synthetic_snap,
                    labels,
                    save_path=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
                )
                snapshots_saved = True

            if args.save_fid_samples and args.output_dir:
                # Use preprocessed (1-channel) samples for saving to disk?
                # User said "when create snapshot, use just 1 channel".
                # "save_fid_samples" is different from snapshots, but better safe.
                # Actually, save_fid_samples usually dumps whatever was used for FID calculation.
                # But if we used average for FID, maybe dumping average is best.
                # Or maybe dumping individual channels?
                # I'll use the FID version (averaged) for consistency with the metric.
                samples_to_save = synthetic_fid

                images_np = (
                    (samples_to_save * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if not args.compute_fid:
            return {}

        if data_iter_step % PRINT_FREQUENCY == 0:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(
                f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )

        if args.test_run:
            break

    metrics = {"fid": float(fid_metric.compute().detach().cpu())}
    if total_generated_samples > 0:
        metrics["avg_sec_per_sample"] = total_generation_time / total_generated_samples
        
    return metrics
