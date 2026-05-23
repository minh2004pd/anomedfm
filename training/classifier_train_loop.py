import argparse
import logging
import math
import torch
from typing import Iterable
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount
from flow_matching.path import CondOTProbPath

logger = logging.getLogger(__name__)

def train_classifier_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    model.train(True)
    epoch_loss = MeanMetric().to(device)
    batch_loss = MeanMetric().to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    path = CondOTProbPath() # Use the same path as flow model for noise sampling
    
    accum_iter = args.accum_iter
    
    # Precision setup
    precision_dtype = torch.float32
    if args.precision == "fp16":
        precision_dtype = torch.float16
    elif args.precision == "bf16":
        precision_dtype = torch.bfloat16
    use_autocast = args.precision != "fp32"

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1, 1) # (B, 1)

        # 1. Scaling to [-1, 1] (consistent with flow model)
        samples = samples * 2.0 - 1.0
        
        # 2. Sample random timesteps and perturb samples (Time-aware classifier)
        t = torch.rand(samples.shape[0], device=device)
        noise = torch.randn_like(samples)
        path_sample = path.sample(t=t, x_0=noise, x_1=samples)
        x_t = path_sample.x_t

        # 3. Forward pass
        with torch.amp.autocast("cuda", enabled=use_autocast, dtype=precision_dtype):
            logits = model(x_t, t)
            loss = criterion(logits, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        
        epoch_loss.update(loss_value * accum_iter)
        batch_loss.update(loss_value * accum_iter)

        if (data_iter_step + 1) % accum_iter == 0:
            if (data_iter_step // accum_iter) % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: "
                    f"batch_loss={batch_loss.compute():.4f}, lr={lr:.6f}"
                )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute())}
