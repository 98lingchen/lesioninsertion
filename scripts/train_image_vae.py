#!/usr/bin/env python3
"""
Train a 2D latent VAE for medical image generation.

Example
-------
python train_latent_embedder_2d.py \
    --input-dir data/train/vae_all \
    --target-dir data/train/vae_all \
    --output-dir runs/vae_2d \
    --batch-size 8 \
    --num-workers 8 \
    --devices 0
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import NiftiPairImageGenerator2D
from medical_diffusion.models.embedders.latent_embedders import VAE


def parse_devices(value: str) -> Union[int, List[int]]:
    """
    Parse a Lightning devices argument.

    Examples
    --------
    "1"   -> 1
    "0"   -> [0]
    "0,1" -> [0, 1]
    """
    value = value.strip()

    if "," in value:
        devices = [int(device.strip()) for device in value.split(",")]
        if not devices:
            raise argparse.ArgumentTypeError("At least one device must be specified.")
        return devices

    device = int(value)
    if device < 0:
        raise argparse.ArgumentTypeError("Device index/count must be non-negative.")

    # A single "0" means CUDA device 0 rather than zero devices.
    return [0] if device == 0 else device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 2D VAE latent embedder using medical_diffusion."
    )

    # Data
    parser.add_argument(
        "--input-dir",
        "--inputfolder",
        dest="input_dir",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--target-dir",
        "--targetfolder",
        dest="target_dir",
        type=Path,
        required=True,
        help="Directory containing target images.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Input image size.",
    )
    parser.add_argument(
        "--batch-size",
        "--batchsize",
        dest="batch_size",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data-loading workers.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pinned memory in the DataLoader.",
    )

    # Model
    parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        help="Number of input channels.",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=1,
        help="Number of output channels.",
    )
    parser.add_argument(
        "--embedding-channels",
        type=int,
        default=8,
        help="Number of latent embedding channels.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Base number of hidden channels.",
    )
    parser.add_argument(
        "--deep-supervision",
        type=int,
        default=1,
        help="Deep supervision setting expected by the VAE implementation.",
    )
    parser.add_argument(
        "--sample-every-n-steps",
        "--save_and_sample_every",
        dest="sample_every_n_steps",
        type=int,
        default=10000,
        help="Generate training samples every N optimizer steps.",
    )
    parser.add_argument(
        "--learning-rate",
        "--train_lr",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate. Applied when supported by the VAE class.",
    )

    # Training
    parser.add_argument(
        "--output-dir",
        "--savefolder",
        dest="output_dir",
        type=Path,
        default=Path("runs/vae_2d"),
        help="Root directory for training outputs.",
    )
    parser.add_argument(
        "--max-epochs",
        "--epochs",
        dest="max_epochs",
        type=int,
        default=1001,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="Minimum number of training epochs.",
    )
    parser.add_argument(
        "--checkpoint-every-n-steps",
        type=int,
        default=None,
        help="Checkpoint interval. Defaults to --sample-every-n-steps.",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="train/L1",
        help="Metric monitored for best-checkpoint selection.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Logging interval.",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=[0],
        help='Lightning devices setting, for example "0", "0,1", or "2".',
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help='Lightning precision, e.g. "32-true", "16-mixed", or "bf16-mixed".',
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used to resume the complete Lightning training state.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used only to initialize model weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic Lightning training.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    if not args.target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {args.target_dir}")

    if args.pretrained_checkpoint is not None and not args.pretrained_checkpoint.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint does not exist: {args.pretrained_checkpoint}"
        )

    if (
        args.resume_from_checkpoint is not None
        and not args.resume_from_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {args.resume_from_checkpoint}"
        )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive.")

    if args.min_epochs < 0:
        raise ValueError("--min-epochs must be non-negative.")

    if args.sample_every_n_steps <= 0:
        raise ValueError("--sample-every-n-steps must be positive.")


def build_model(args: argparse.Namespace) -> VAE:
    hidden_channels = [
        args.base_channels,
        args.base_channels * 2,
        args.base_channels * 4,
        args.base_channels * 8,
    ]

    model_kwargs = dict(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        emb_channels=args.embedding_channels,
        spatial_dims=2,
        hid_chs=hidden_channels,
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        deep_supervision=args.deep_supervision,
        use_attention="none",
        loss=torch.nn.MSELoss,
        sample_every_n_steps=args.sample_every_n_steps,
    )

    # Some medical_diffusion versions expose optimizer configuration through
    # the constructor, while others configure it internally.
    try:
        model = VAE(
            **model_kwargs,
            optimizer_kwargs={"lr": args.learning_rate},
        )
    except TypeError:
        model = VAE(**model_kwargs)

    if args.pretrained_checkpoint is not None:
        model.load_pretrained(args.pretrained_checkpoint, strict=True)

    return model


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.multiprocessing.set_sharing_strategy("file_system")

    run_name = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset = NiftiPairImageGenerator2D(
        str(args.input_dir),
        str(args.target_dir),
        input_size=args.input_size,
    )

    data_module = SimpleDataModule(
        ds_train=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = build_model(args)

    checkpoint_interval = (
        args.checkpoint_every_n_steps
        if args.checkpoint_every_n_steps is not None
        else args.sample_every_n_steps
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="epoch={epoch:04d}-step={step:08d}",
        monitor=args.monitor,
        mode="min",
        every_n_train_steps=checkpoint_interval,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.devices if accelerator == "gpu" else 1

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        default_root_dir=str(run_dir),
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        deterministic=args.deterministic,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
    )

    if hasattr(model, "save_best_checkpoint"):
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            logger_dir = (
                trainer.logger.log_dir
                if trainer.logger is not None
                else str(run_dir)
            )
            model.save_best_checkpoint(logger_dir, best_model_path)


if __name__ == "__main__":
    main()
