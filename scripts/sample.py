#!/usr/bin/env python3
"""Generate 3D synthetic lesions from paired NIfTI volumes.

This script is a GitHub-ready inference entry point for the two-stage pipeline:

1. Generate a 2D lesion mask on a sampled central slice.
2. Extend the mask through adjacent slices using binary erosion.
3. Generate lesion content slice by slice in latent space.
4. Blend the generated lesion with the original CT volume.

Example
-------
python scripts/generate_lesion_3d.py \
    --anatomy-dir data/test/anatomy \
    --image-dir data/test/images \
    --output-dir outputs/generated_lesions \
    --lesion-distribution configs/lesion_distribution_256.json \
    --mask-checkpoint checkpoints/mask_diffusion.ckpt \
    --lesion-checkpoint checkpoints/lesion_diffusion.ckpt \
    --vae-checkpoint checkpoints/vae.ckpt \
    --device cuda:0

The repository must expose the ``medical_diffusion`` package. Install the
project in editable mode before running the script, for example:

    pip install -e .
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion

from medical_diffusion.models.embedders import (
    Latent_Embedder,
    SizeEmbbeding,
    TimeEmbbeding,
)
from medical_diffusion.models.embedders.latent_embedders import VAE
from medical_diffusion.models.estimators import UNet, UNet_nosize_noVAE
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.pipelines import (
    DiffusionPipeline,
    DiffusionPipeline0121,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 3D lesions from paired NIfTI volumes."
    )
    parser.add_argument(
        "--anatomy-dir",
        type=Path,
        required=True,
        help="Directory containing anatomy/body-mask NIfTI volumes.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing original CT NIfTI volumes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which generated cases will be saved.",
    )
    parser.add_argument(
        "--lesion-distribution",
        type=Path,
        required=True,
        help="JSON file containing central-slice and slice-count distributions.",
    )
    parser.add_argument(
        "--mask-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint for the lesion-mask diffusion model.",
    )
    parser.add_argument(
        "--lesion-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint for the lesion-generation diffusion model.",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint for the VAE used by the lesion model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example 'cuda:0' or 'cpu'.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--lesion-size-min", type=int, default=1000)
    parser.add_argument("--lesion-size-max", type=int, default=5000)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cases whose output NIfTI file already exists.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def list_nifti_files(directory: Path) -> set[str]:
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
    }


def save_3d_volume_as_png(
    volume: np.ndarray,
    save_path: Path,
    vmin: float = 0.0,
    vmax: float = 100.0,
    columns: int = 8,
) -> None:
    """Save all axial slices of a 3D volume as one mosaic image."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, but received shape {volume.shape}.")

    num_slices = volume.shape[2]
    rows = int(np.ceil(num_slices / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
    axes_array = np.asarray(axes).reshape(rows, columns)

    for index in range(rows * columns):
        axis = axes_array[index // columns, index % columns]
        axis.axis("off")
        if index < num_slices:
            axis.imshow(volume[:, :, index], cmap="gray", vmin=vmin, vmax=vmax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tensor_to_2d_numpy(tensor: torch.Tensor, name: str) -> np.ndarray:
    array = np.squeeze(tensor.detach().cpu().numpy())
    if array.ndim != 2:
        raise ValueError(f"Expected {name} to be 2D after squeeze, got {array.shape}.")
    return array.astype(np.float32, copy=False)


def build_mask_pipeline(
    checkpoint: Path,
    device: torch.device,
) -> DiffusionPipeline:
    estimator_kwargs: dict[str, Any] = {
        "in_ch": 2,
        "out_ch": 1,
        "spatial_dims": 2,
        "hid_chs": [64, 64, 128, 256],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "time_embedder": TimeEmbbeding,
        "time_embedder_kwargs": {"emb_dim": 1024},
        "size_embedder": SizeEmbbeding,
        "size_embedder_kwargs": {"emb_dim": 1024},
        "cond_embedder": None,
        "deep_supervision": False,
        "use_res_block": True,
        "use_attention": "none",
        "masked_condition": False,
    }
    scheduler_kwargs = {
        "timesteps": 1000,
        "beta_start": 0.002,
        "beta_end": 0.02,
        "schedule_strategy": "scaled_linear",
    }

    pipeline = DiffusionPipeline(
        noise_estimator=UNet,
        noise_estimator_kwargs=estimator_kwargs,
        noise_scheduler=GaussianNoiseScheduler,
        noise_scheduler_kwargs=scheduler_kwargs,
        estimator_objective="x_T",
        estimate_variance=False,
        use_self_conditioning=False,
        num_samples=1,
        use_ema=False,
        classifier_free_guidance_dropout=0.5,
        optimizer_kwargs={"lr": 1e-4},
        do_input_centering=False,
        clip_x0=False,
        masked_condition=True,
    )
    pipeline.load_pretrained(checkpoint)
    pipeline.to(device)
    pipeline.eval()
    return pipeline


def build_lesion_pipeline(
    checkpoint: Path,
    vae_checkpoint: Path,
    device: torch.device,
) -> DiffusionPipeline0121:
    estimator_kwargs: dict[str, Any] = {
        "in_ch": 16,
        "out_ch": 8,
        "spatial_dims": 2,
        "hid_chs": [64, 64, 128, 256],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "time_embedder": TimeEmbbeding,
        "time_embedder_kwargs": {"emb_dim": 1024},
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
    scheduler_kwargs = {
        "timesteps": 1000,
        "beta_start": 0.002,
        "beta_end": 0.02,
        "schedule_strategy": "scaled_linear",
    }

    pipeline = DiffusionPipeline0121(
        noise_estimator=UNet_nosize_noVAE,
        noise_estimator_kwargs=estimator_kwargs,
        noise_scheduler=GaussianNoiseScheduler,
        noise_scheduler_kwargs=scheduler_kwargs,
        latent_embedder=VAE,
        latent_embedder_checkpoint=str(vae_checkpoint),
        estimator_objective="x_T",
        estimate_variance=False,
        use_self_conditioning=False,
        num_samples=1,
        use_ema=False,
        classifier_free_guidance_dropout=0.5,
        optimizer_kwargs={"lr": 1e-4},
        do_input_centering=False,
        clip_x0=False,
        masked_condition=True,
    )
    pipeline.load_pretrained(checkpoint)
    pipeline.to(device)
    pipeline.eval()
    return pipeline


def sample_case_parameters(
    distribution: dict[str, Any],
    lesion_size_min: int,
    lesion_size_max: int,
) -> tuple[int, int, int]:
    central_slice = int(
        np.random.choice(distribution["max_mask_slice"]["values"])
    )
    num_slices = int(
        np.random.choice(distribution["nonzero_slice_count"]["values"])
    )
    lesion_size = random.randint(lesion_size_min, lesion_size_max)
    return central_slice, num_slices, lesion_size


def is_active_slice(index: int, central_slice: int, num_slices: int) -> bool:
    """Match the slice-selection rule used by the original inference script."""
    return (
        index > central_slice - num_slices // 2
        and index < central_slice + num_slices // 2
    )


def build_3d_mask(
    central_mask: np.ndarray,
    anatomy_volume: np.ndarray,
    central_slice: int,
    num_slices: int,
) -> np.ndarray:
    image_mask = np.zeros_like(anatomy_volume, dtype=np.float32)
    structure = np.ones((3, 3), dtype=bool)

    for index in range(anatomy_volume.shape[2]):
        if is_active_slice(index, central_slice, num_slices):
            distance = abs(index - central_slice)
            if distance == 0:
                eroded_mask = central_mask
            else:
                eroded_mask = binary_erosion(
                    central_mask,
                    structure=structure,
                    iterations=distance,
                )
            image_mask[:, :, index] = eroded_mask.astype(np.float32)

        if index == central_slice:
            image_mask[:, :, index] = central_mask

        image_mask[:, :, index] *= anatomy_volume[:, :, index]

    return image_mask


def save_nifti_like(
    array: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    save_path: Path,
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    output = nib.Nifti1Image(
        array.astype(np.float32),
        affine=reference.affine,
        header=header,
    )
    nib.save(output, save_path)


def process_case(
    filename: str,
    args: argparse.Namespace,
    lesion_distribution: dict[str, Any],
    mask_pipeline: DiffusionPipeline,
    lesion_pipeline: DiffusionPipeline0121,
    device: torch.device,
) -> None:
    anatomy_path = args.anatomy_dir / filename
    image_path = args.image_dir / filename
    case_name = strip_nii_suffix(filename)
    case_dir = args.output_dir / case_name
    output_nii = case_dir / f"{case_name}.nii.gz"

    if output_nii.exists() and not args.overwrite:
        print(f"[skip] {case_name}: output already exists")
        return

    anatomy_nii = nib.load(anatomy_path)
    image_nii = nib.load(image_path)
    anatomy = anatomy_nii.get_fdata(dtype=np.float32)
    original_hu = image_nii.get_fdata(dtype=np.float32)

    expected_shape = (args.height, args.width, args.depth)
    if anatomy.shape != expected_shape or original_hu.shape != expected_shape:
        raise ValueError(
            f"{case_name}: expected paired volumes with shape {expected_shape}, "
            f"got anatomy={anatomy.shape}, image={original_hu.shape}."
        )

    original_hu = np.clip(original_hu, -100.0, 200.0)
    original_normalized = original_hu / 100.0

    central_slice, num_slices, lesion_size = sample_case_parameters(
        lesion_distribution,
        args.lesion_size_min,
        args.lesion_size_max,
    )
    if not 0 <= central_slice < args.depth:
        raise ValueError(
            f"{case_name}: sampled central slice {central_slice} is outside "
            f"the valid range [0, {args.depth - 1}]."
        )
    if num_slices <= 0:
        raise ValueError(f"{case_name}: sampled invalid slice count {num_slices}.")

    print(
        f"[run] {case_name}: central_slice={central_slice}, "
        f"lesion_size={lesion_size}, num_slices={num_slices}"
    )

    anatomy_slice = torch.from_numpy(anatomy[:, :, central_slice]).to(
        device=device, dtype=torch.float32
    )[None, None]
    size_tensor = torch.tensor(
        [lesion_size], device=device, dtype=torch.float32
    )

    with torch.inference_mode():
        mask_result = mask_pipeline.sample(
            1,
            (1, args.height, args.width),
            size_tensor,
            condition=anatomy_slice,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            use_ddim=True,
        )

    central_mask = tensor_to_2d_numpy(mask_result, "generated mask")
    central_mask = (central_mask >= args.mask_threshold).astype(np.float32)
    image_mask = build_3d_mask(
        central_mask,
        anatomy,
        central_slice,
        num_slices,
    )

    generated_normalized = original_normalized.copy()
    for index in range(args.depth):
        if not is_active_slice(index, central_slice, num_slices):
            continue

        source_slice = np.minimum(original_normalized[:, :, index], 3.0)
        source_tensor = torch.from_numpy(source_slice).to(
            device=device, dtype=torch.float32
        )[None, None]
        mask_tensor = torch.from_numpy(image_mask[:, :, index]).to(
            device=device, dtype=torch.float32
        )[None, None]

        with torch.inference_mode():
            x_0 = lesion_pipeline.latent_embedder.encode(source_tensor)
            masked_source = source_tensor * (1.0 - mask_tensor)
            masked_x_0 = lesion_pipeline.latent_embedder.encode(masked_source)

            latent_mask = F.interpolate(mask_tensor, size=(32, 32))
            condition = torch.cat([masked_x_0, latent_mask], dim=1)
            lesion_result = lesion_pipeline.sample(
                1,
                (8, 32, 32),
                x0=x_0,
                condition=condition,
                condition2=latent_mask,
                guidance_scale=args.guidance_scale,
                steps=args.steps,
                use_ddim=True,
            ).detach()

        generated_normalized[:, :, index] = tensor_to_2d_numpy(
            lesion_result, "generated lesion slice"
        )

    generated_hu = generated_normalized * 100.0
    generated_hu = (
        generated_hu * image_mask + original_hu * (1.0 - image_mask)
    ).astype(np.float32)

    case_dir.mkdir(parents=True, exist_ok=True)
    save_3d_volume_as_png(
        image_mask,
        case_dir / "generated_mask.png",
        vmin=0.0,
        vmax=1.0,
    )
    save_3d_volume_as_png(
        original_hu,
        case_dir / "original_image.png",
        vmin=0.0,
        vmax=100.0,
    )
    save_3d_volume_as_png(
        generated_hu,
        case_dir / "generated_image.png",
        vmin=0.0,
        vmax=100.0,
    )
    save_nifti_like(
        image_mask,
        image_nii,
        case_dir / f"{case_name}_mask.nii.gz",
    )
    save_nifti_like(generated_hu, image_nii, output_nii)
    print(f"[done] {case_name}: saved to {case_dir}")


def main() -> None:
    args = parse_args()

    for path, description in (
        (args.anatomy_dir, "Anatomy directory"),
        (args.image_dir, "Image directory"),
        (args.lesion_distribution, "Lesion-distribution JSON"),
        (args.mask_checkpoint, "Mask checkpoint"),
        (args.lesion_checkpoint, "Lesion checkpoint"),
        (args.vae_checkpoint, "VAE checkpoint"),
    ):
        validate_path(path, description)

    if args.lesion_size_min > args.lesion_size_max:
        raise ValueError("--lesion-size-min cannot exceed --lesion-size-max.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    set_random_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.lesion_distribution.open("r", encoding="utf-8") as file:
        lesion_distribution = json.load(file)

    anatomy_files = list_nifti_files(args.anatomy_dir)
    image_files = list_nifti_files(args.image_dir)
    common_files = sorted(anatomy_files & image_files)
    if not common_files:
        raise RuntimeError(
            "No paired .nii or .nii.gz files with matching names were found."
        )

    print(f"Found {len(common_files)} paired NIfTI volumes.")
    print(f"Using device: {device}")

    mask_pipeline = build_mask_pipeline(args.mask_checkpoint, device)
    lesion_pipeline = build_lesion_pipeline(
        args.lesion_checkpoint,
        args.vae_checkpoint,
        device,
    )

    for filename in common_files:
        process_case(
            filename,
            args,
            lesion_distribution,
            mask_pipeline,
            lesion_pipeline,
            device,
        )


if __name__ == "__main__":
    main()
