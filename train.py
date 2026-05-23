# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
try:
    import wandb
except ImportError:
    wandb = None
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from training import distributed_mode
from training.data_transform import get_train_transform
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch

from datasets.simple_shape import SimpleShapeDataset
from datasets.brats import BraTSDataset, BraTSPreprocessedDataset


logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()
    if args.dataset == "imagenet":
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif args.dataset == "simple_shape":
        dataset_train = SimpleShapeDataset(num_samples=args.num_samples, image_size=args.image_size)
    elif args.dataset == "mnist":
        dataset_train = datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif args.dataset in ("brats", "bratsv2"):
        healthy_only = getattr(args, 'healthy_only', False)
        if getattr(args, 'use_preprocessed', False):
            dataset_train = BraTSPreprocessedDataset(
                root=args.data_path,
                mode='train',
                healthy_only=healthy_only,
                return_label=True,
            )
            dataset_val = BraTSPreprocessedDataset(
                root=args.data_path,
                mode='val',
                healthy_only=False,
                return_label=True,
            )
            dataset_test = BraTSPreprocessedDataset(
                root=args.data_path,
                mode='test',
                healthy_only=False,
                return_label=True,
            )
        else:
            dataset_train = BraTSDataset(
                root=args.data_path,
                image_size=args.image_size,
                mode='train',
                healthy_only=healthy_only,
            )
            dataset_val = BraTSDataset(
                root=args.data_path,
                image_size=args.image_size,
                mode='valid',
                healthy_only=False,
            )
            dataset_test = dataset_val  # BraTSDataset raw mode has no separate test split
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")
    
    # For datasets other than BraTS, dataset_val/test might not be defined.
    # We fallback to dataset_train to maintain compatibility.
    if 'dataset_val' not in locals():
        dataset_val = dataset_train
    if 'dataset_test' not in locals():
        dataset_test = dataset_val

    logger.info(dataset_train)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    # setup validation loader
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # setup test loader
    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    logger.info("Initializing Model")
    model_arch = args.dataset
    if args.dataset in ("brats", "bratsv2") and getattr(args, 'healthy_only', False):
        model_arch = "brats_healthy"
        logger.info("Using brats_healthy (unconditional) architecture for healthy-only training.")
    
    model = instantiate_model(
        architechture=model_arch,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
    )

    model.to(device)

    model_without_ddp = model
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )

    # Determine scheduler type (new --lr_scheduler takes precedence over legacy --decay_lr)
    scheduler_type = getattr(args, 'lr_scheduler', 'constant')
    if scheduler_type == 'constant' and args.decay_lr:
        scheduler_type = 'linear'  # backward compatibility

    warmup_epochs = getattr(args, 'warmup_epochs', 0)
    min_lr = getattr(args, 'min_lr', 1e-6)
    train_epochs = args.epochs - args.start_epoch

    # Build the main scheduler
    if scheduler_type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, train_epochs - warmup_epochs),
            eta_min=min_lr,
        )
        logger.info(f"Using CosineAnnealingLR: T_max={max(1, train_epochs - warmup_epochs)}, eta_min={min_lr}")
    elif scheduler_type == 'linear':
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=max(1, train_epochs - warmup_epochs),
            start_factor=1.0,
            end_factor=min_lr / args.lr,
        )
        logger.info(f"Using LinearLR decay to {min_lr}")
    else:
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )
        logger.info("Using ConstantLR (no decay)")

    # Combine with warmup if requested
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=min_lr / args.lr,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        lr_schedule = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
        logger.info(f"Using warmup for {warmup_epochs} epochs (from {min_lr} to {args.lr})")
    else:
        lr_schedule = main_scheduler

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler(enabled=(getattr(args, 'precision', 'fp32') == 'fp16'))

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    if getattr(args, "wandb", False) and wandb is not None and distributed_mode.is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or Path(args.output_dir).name,
            config=vars(args),
            resume="allow",
        )

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            log_stats = {
                 **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
                hf_repo = os.environ.get("HF_REPO", "")
                hf_token = os.environ.get("HF_TOKEN", "")
                if hf_repo and hf_token and distributed_mode.is_main_process():
                    try:
                        from huggingface_hub import HfApi, login
                        login(token=hf_token, add_to_git_credential=False)
                        api = HfApi()
                        api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=True)
                        ckpt = os.path.join(args.output_dir, "checkpoint.pth")
                        if os.path.exists(ckpt):
                            api.upload_file(path_or_fileobj=ckpt, path_in_repo=f"checkpoint_epoch{epoch+1:04d}.pth", repo_id=hf_repo, repo_type="model")
                            api.upload_file(path_or_fileobj=ckpt, path_in_repo="checkpoint.pth", repo_id=hf_repo, repo_type="model")
                            logger.info(f"Uploaded checkpoint epoch {epoch+1} to HF: {hf_repo}")
                    except Exception as e:
                        logger.warning(f"HF upload failed: {e}")
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks
            eval_stats = eval_model(
                model,
                data_loader_val,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
            )
            eval_stats_processed = {f"eval_{k}": v for k, v in eval_stats.items()}
            logger.info(f"Evaluation stats: {eval_stats_processed}")
            log_stats.update(eval_stats_processed)

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            if getattr(args, "wandb", False) and wandb is not None:
                wandb.log({**log_stats, "epoch": epoch})

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")

    # Final evaluation on the held-out test set
    if getattr(args, 'run_test', False) and not args.test_run:
        logger.info("Running final evaluation on test set...")
        if args.distributed:
            data_loader_test.sampler.set_epoch(0)
        if distributed_mode.is_main_process():
            fid_samples_test = args.fid_samples - (num_tasks - 1) * (
                args.fid_samples // num_tasks
            )
        else:
            fid_samples_test = args.fid_samples // num_tasks
        test_stats = eval_model(
            model,
            data_loader_test,
            device,
            epoch=args.epochs,
            fid_samples=fid_samples_test,
            args=args,
        )
        test_stats_processed = {f"test_{k}": v for k, v in test_stats.items()}
        logger.info(f"Test stats: {test_stats_processed}")
        if args.output_dir and distributed_mode.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps({**test_stats_processed, "epoch": "test"}) + "\n")
            if getattr(args, "wandb", False) and wandb is not None:
                wandb.log(test_stats_processed)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
