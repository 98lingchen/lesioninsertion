#!/usr/bin/env python3
"""
Train a conditional 2D latent diffusion model for lesion generation.

Example
-------
python train_lesion_diffusion_aisd.py \
    --input-dir data/AISD/2Dmask \
    --target-dir data/AISD/2DCT \
    --vae-checkpoint checkpoints/vae_8ch.ckpt \
    --output-dir runs/aisd_lesion_diffusion \
    --batch-size 8 \
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
from medical_diffusion.models.embedders import Latent_Embedder, TimeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE
from medical_diffusion.models.estimators import UNet_nosize_noVAE
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.pipelines import DiffusionPipeline0121


def parse_devices(value: str) -> Union[int, List[int]]:
    """
    Parse a PyTorch Lightning devices argument.

    Examples
    --------
    "0"   -> [0]
    "0,1" -> [0, 1]
    "2"   -> 2
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

    return [0] if device == 0 else device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a conditional latent diffusion model for 2D lesion generation."
    )

    # Data
    parser.add_argument(
        "--input-dir",
        "--inputfolder",
        dest="input_dir",
        type=Path,
        required=True,
        help="Directory containing lesion masks or conditioning inputs.",
    )
    parser.add_argument(
        "--target-dir",
        "--targetfolder",
        dest="target_dir",
        type=Path,
        required=True,
        help="Directory containing target CT images.",
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
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pinned memory in the DataLoader.",
    )
    parser.add_argument(
        "--full-channel-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable full-channel mask loading in the dataset.",
    )

    # Latent model
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint of the pretrained VAE latent embedder.",
    )
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=8,
        help="Number of latent channels produced by the VAE.",
    )
    parser.add_argument(
        "--condition-in-channels",
        type=int,
        default=9,
        help="Number of channels supplied to the condition embedder.",
    )
    parser.add_argument(
        "--condition-embedding-channels",
        type=int,
        default=8,
        help="Number of output channels from the condition embedder.",
    )

    # Noise estimator
    parser.add_argument(
        "--model-base-channels",
        type=int,
        default=64,
        help="Base channel count of the diffusion U-Net.",
    )
    parser.add_argument(
        "--time-embedding-dim",
        type=int,
        default=1024,
        help="Time-embedding dimension.",
    )
    parser.add_argument(
        "--masked-condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable masked conditioning in the diffusion pipeline.",
    )
    parser.add_argument(
        "--classifier-free-guidance-dropout",
        type=float,
        default=0.5,
        help="Condition dropout probability used during training.",
    )

    # Noise scheduler
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps.",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=0.002,
        help="Initial beta value.",
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=0.02,
        help="Final beta value.",
    )
    parser.add_argument(
        "--schedule-strategy",
        type=str,
        default="scaled_linear",
        choices=["linear", "scaled_linear", "cosine"],
        help="Noise schedule strategy.",
    )

    # Optimization and outputs
    parser.add_argument(
        "--learning-rate",
        "--train_lr",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--sample-every-n-steps",
        "--save_and_sample_every",
        dest="sample_every_n_steps",
        type=int,
        default=10000,
        help="Sampling interval during training.",
    )
    parser.add_argument(
        "--checkpoint-every-n-steps",
        type=int,
        default=100,
        help="Checkpoint interval.",
    )
    parser.add_argument(
        "--output-dir",
        "--savefolder",
        dest="output_dir",
        type=Path,
        default=Path("runs/aisd_lesion_diffusion"),
        help="Root output directory.",
    )

    # Trainer
    parser.add_argument(
        "--max-epochs",
        "--epochs",
        dest="max_epochs",
        type=int,
        default=500000,
        help="Maximum number of epochs.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="Minimum number of epochs.",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="train/loss",
        help="Metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=3,
        help="Number of best checkpoints to retain.",
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
        help="Checkpoint used to resume the full Lightning training state.",
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
        help="Enable deterministic training.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    if not args.target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {args.target_dir}")

    if not args.vae_checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint does not exist: {args.vae_checkpoint}")

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

    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")

    if not 0.0 <= args.classifier_free_guidance_dropout <= 1.0:
        raise ValueError(
            "--classifier-free-guidance-dropout must be between 0 and 1."
        )

    if args.checkpoint_every_n_steps <= 0:
        raise ValueError("--checkpoint-every-n-steps must be positive.")


def build_pipeline(args: argparse.Namespace) -> DiffusionPipeline0121:
    latent_channels = args.latent_channels
    model_base_channels = args.model_base_channels

    noise_estimator_kwargs = {
        "in_ch": latent_channels * 2,
        "out_ch": latent_channels,
        "spatial_dims": 2,
        "hid_chs": [
            model_base_channels,
            model_base_channels,
            model_base_channels * 2,
            model_base_channels * 4,
        ],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "time_embedder": TimeEmbbeding,
        "time_embedder_kwargs": {
            "emb_dim": args.time_embedding_dim,
        },
        "cond_embedder": Latent_Embedder,
        "cond_embedder_kwargs": {
            "in_channels": args.condition_in_channels,
            "emb_channels": args.condition_embedding_channels,
            "strides": [1, 1, 1, 1],
            "hid_chs": [32, 64, 128, 256],
        },
        "deep_supervision": False,
        "use_res_block": True,
        "use_attention": "none",
        "masked_condition": False,
    }

    noise_scheduler_kwargs = {
        "timesteps": args.timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "schedule_strategy": args.schedule_strategy,
    }

    return DiffusionPipeline0121(
        noise_estimator=UNet_nosize_noVAE,
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=GaussianNoiseScheduler,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
        latent_embedder=VAE,
        latent_embedder_checkpoint=str(args.vae_checkpoint),
        estimator_objective="x_T",
        estimate_variance=False,
        use_self_conditioning=False,
        num_samples=1,
        use_ema=False,
        classifier_free_guidance_dropout=args.classifier_free_guidance_dropout,
        optimizer_kwargs={"lr": args.learning_rate},
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=args.sample_every_n_steps,
        masked_condition=args.masked_condition,
    )


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

    dataset_kwargs = {
        "input_size": args.input_size,
    }
    if args.full_channel_mask:
        dataset_kwargs["full_channel_mask"] = True

    dataset = NiftiPairImageGenerator2D(
        str(args.input_dir),
        str(args.target_dir),
        **dataset_kwargs,
    )

    data_module = SimpleDataModule(
        ds_train=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    pipeline = build_pipeline(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="epoch={epoch:04d}-step={step:08d}",
        monitor=args.monitor,
        mode="min",
        every_n_train_steps=args.checkpoint_every_n_steps,
        save_last=True,
        save_top_k=args.save_top_k,
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
        pipeline,
        datamodule=data_module,
        ckpt_path=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
    )

    if hasattr(pipeline, "save_best_checkpoint"):
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            logger_dir = (
                trainer.logger.log_dir
                if trainer.logger is not None
                else str(run_dir)
            )
            pipeline.save_best_checkpoint(logger_dir, best_model_path)


if __name__ == "__main__":
    main()
