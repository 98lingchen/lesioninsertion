"""Train a conditional latent diffusion model for 2D lesion generation.

Example
-------
python train_lesion_diffusion.py \
    --input-dir data/train/labels_2d \
    --target-dir data/train/images_2d \
    --vae-checkpoint checkpoints/vae.ckpt \
    --output-dir runs/lesion_diffusion \
    --devices 0
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

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


def parse_devices(value: str) -> int | list[int]:
    """Parse Lightning device specification.

    Examples: ``1`` uses one GPU, while ``0,1`` uses GPU 0 and GPU 1.
    """
    value = value.strip()
    if "," in value:
        try:
            return [int(device.strip()) for device in value.split(",")]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--devices must be an integer or a comma-separated list, e.g. 1 or 0,1."
            ) from exc

    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--devices must be an integer or a comma-separated list, e.g. 1 or 0,1."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a conditional latent diffusion model for 2D lesion generation."
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--input-dir", type=Path, required=True, help="Condition/mask directory.")
    data_group.add_argument("--target-dir", type=Path, required=True, help="Target image directory.")
    data_group.add_argument("--input-size", type=int, default=256)
    data_group.add_argument("--batch-size", type=int, default=16)
    data_group.add_argument("--num-workers", type=int, default=16)
    data_group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--vae-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint of the pretrained VAE used as the latent embedder.",
    )
    model_group.add_argument("--learning-rate", type=float, default=1e-4)
    model_group.add_argument("--timesteps", type=int, default=1000)
    model_group.add_argument("--beta-start", type=float, default=0.002)
    model_group.add_argument("--beta-end", type=float, default=0.02)
    model_group.add_argument("--cfg-dropout", type=float, default=0.5)
    model_group.add_argument(
        "--masked-condition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--output-dir", type=Path, default=Path("runs/lesion_diffusion"))
    training_group.add_argument("--max-epochs", type=int, default=500000)
    training_group.add_argument("--min-epochs", type=int, default=100)
    training_group.add_argument("--devices", type=parse_devices, default=1)
    training_group.add_argument("--precision", default="32-true")
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--log-every-n-steps", type=int, default=100)
    training_group.add_argument("--checkpoint-every-n-steps", type=int, default=100)
    training_group.add_argument("--sample-every-n-steps", type=int, default=10000)
    training_group.add_argument("--save-top-k", type=int, default=3)
    training_group.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional Lightning checkpoint used to resume training.",
    )

    return parser


def validate_paths(paths: Sequence[tuple[str, Path]]) -> None:
    missing = [f"{name}: {path}" for name, path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("The following required paths do not exist:\n  " + "\n  ".join(missing))


def build_pipeline(args: argparse.Namespace) -> DiffusionPipeline0121:
    time_embedder_kwargs = {"emb_dim": 1024}

    noise_estimator_kwargs = {
        "in_ch": 16,
        "out_ch": 8,
        "spatial_dims": 2,
        "hid_chs": [64, 64, 128, 256],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "time_embedder": TimeEmbbeding,
        "time_embedder_kwargs": time_embedder_kwargs,
        "cond_embedder": Latent_Embedder,
        "cond_embedder_kwargs": {
            "in_channels": 9,
            "emb_channels": 8,
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
        "schedule_strategy": "scaled_linear",
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
        classifier_free_guidance_dropout=args.cfg_dropout,
        optimizer_kwargs={"lr": args.learning_rate},
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=args.sample_every_n_steps,
        masked_condition=args.masked_condition,
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_paths(
        [
            ("input directory", args.input_dir),
            ("target directory", args.target_dir),
            ("VAE checkpoint", args.vae_checkpoint),
        ]
    )

    seed_everything(args.seed, workers=True)
    torch.multiprocessing.set_sharing_strategy("file_system")

    dataset = NiftiPairImageGenerator2D(
        str(args.input_dir),
        str(args.target_dir),
        input_size=args.input_size,
    )
    datamodule = SimpleDataModule(
        ds_train=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    run_name = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    pipeline = build_pipeline(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="epoch={epoch}-step={step}-loss={train/loss:.6f}",
        monitor="train/loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        every_n_train_steps=args.checkpoint_every_n_steps,
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
        logger=True,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
    )

    ckpt_path = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    trainer.fit(pipeline, datamodule=datamodule, ckpt_path=ckpt_path)

    if hasattr(pipeline, "save_best_checkpoint"):
        pipeline.save_best_checkpoint(
            trainer.logger.log_dir,
            checkpoint_callback.best_model_path,
        )


if __name__ == "__main__":
    main()
