#%%
import torchvision.transforms as transforms
import sys 
import os
# sys.path.append("./medfusion_3d")
import torch.nn.functional as F 
sys.path.append('/workspace/lesion_insertion/Mask2PET3D_2')
from pathlib import Path
import torch 
from torchvision import utils 
import math 
from medical_diffusion.models.pipelines import DiffusionPipeline, DiffusionPipeline0121noVAE, DiffusionPipeline0121
import logging
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from medical_diffusion.data.datasets import NiftiPairImageGenerator, NiftiPairImageGenerator2D
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import nibabel as nib
import random

from medical_diffusion.models.estimators import UNet, UNet_nosize, UNet_nosize_noVAE
from medical_diffusion.models.embedders import Latent_Embedder, TimeEmbbeding, SizeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE, VAEGAN, VQVAE, VQGAN
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
import json
from scipy.ndimage import binary_erosion

def save_3d_volume_as_png(volume_3d, save_path, v_min =0, v_max =100):
    """
    将 256x256x32 的 3D 图像保存为一个 .png 文件。
    每行显示 8 张切片，共 4 行，组成一个 mosaic。

    Args:
        volume_3d (np.ndarray): 形状为 (256, 256, 32) 的图像。
        save_path (str): 输出文件路径，如 'output.png'。
    """
    assert volume_3d.shape == (256, 256, 32), "输入必须为 256x256x32 的 3D 图像"

    num_slices = volume_3d.shape[2]
    cols = 8
    rows = (num_slices + cols - 1) // cols  # 自动计算行数

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis('off')

        if i < num_slices:
            ax.imshow(volume_3d[:, :, i], cmap='gray', vmin = v_min, vmax = v_max)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()









def apply_transfer_to_img(img: np.array, bins: np.array, bins_mapped: np.array, reverse=False):
    if reverse:
        bins, bins_mapped = bins_mapped, bins
    mask = (img > bins[0]) & (img < bins[-1])
    img_mapped = np.interp(img.astype(np.float32), bins, bins_mapped)
    img_mapped[~mask] = img[~mask]
    return img_mapped

def reverse_histogram_equalization(img, histogram_csv):

    # 加载映射函数
    df = pd.read_csv(histogram_csv)
    bins = df['HU'].values
    bins_mapped = df['HU_mapped'].values



    img_restored = apply_transfer_to_img(img, bins, bins_mapped, reverse=True)

    return img_restored
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 忽略 TensorFlow/TensorBoard 的 info & warning
torch.manual_seed(0)
masked_condition = True

device = torch.device('cuda')
# ----------------------define the model----------------------







noise_estimator = UNet
noise_estimator_kwargs = {
    'in_ch':2, 
    'out_ch':1, 
    'spatial_dims':2,
    # 'hid_chs':  [  256, 256, 512, 1024],
    'hid_chs':  [  64, 64, 128, 256],
    # 'hid_chs':  [  32, 32, 64, 128],
    'kernel_sizes':[3, 3, 3, 3],
    'strides':     [1, 2, 2, 2],
    'time_embedder':TimeEmbbeding,
    'time_embedder_kwargs': {'emb_dim': 1024},
    'size_embedder':SizeEmbbeding,
    'size_embedder_kwargs': {'emb_dim': 1024},
    'cond_embedder':None,
    # 'cond_embedder_kwargs': {
    #     'in_channels': 1,
    #     'emb_channels': 1,
    #     'strides' : [ 1,  1,   1,   1],
    #     'hid_chs' : [32, 64, 128,  256],
    # },
    'deep_supervision': False,
    'use_res_block':True,
    'use_attention':'none',
    'masked_condition': False
}

# ------------ Initialize Noise ------------
noise_scheduler = GaussianNoiseScheduler
noise_scheduler_kwargs = {
    'timesteps': 1000,
    'beta_start': 0.002, # 0.0001, 0.0015
    'beta_end': 0.02, # 0.01, 0.0195
    'schedule_strategy': 'scaled_linear'
}


pipeline_mask = DiffusionPipeline(
    noise_estimator=noise_estimator, 
    noise_estimator_kwargs=noise_estimator_kwargs,
    noise_scheduler=noise_scheduler, 
    noise_scheduler_kwargs = noise_scheduler_kwargs,
    # latent_embedder=latent_embedder,
    # latent_embedder_checkpoint = latent_embedder_checkpoint,
    estimator_objective='x_T',
    estimate_variance=False, 
    use_self_conditioning=False, 
    num_samples = 1,
    use_ema=False,
    classifier_free_guidance_dropout=0.5, # Disable during training by setting to 0
    optimizer_kwargs={'lr':1e-4}, # stable-diffusion ~ 1e-4
    do_input_centering=False,
    clip_x0=False,
    # sample_every_n_steps=save_and_sample_every,
    masked_condition=masked_condition
)

ckpt_path = '/mnt/data/lesion_insertion/AISDruns/diffusion_generate_mask/2025_04_16_135620/epoch=874-step=790700.ckpt'
#'./medfusion_3d/runs/LDM_VQGAN/2024_06_07_175241/epoch=1079-step=53999.ckpt'
pipeline_mask.load_pretrained(Path(ckpt_path))

pipeline_mask.to(device)



device = torch.device('cuda')
# ----------------------define the model----------------------

    # ------------ Initialize Model ------------
# cond_embedder = Latent_Embedder
#%%






noise_estimator = UNet_nosize_noVAE

noise_estimator_kwargs = {
    'in_ch':16, 
    'out_ch':8, 
    'spatial_dims':2,
    'hid_chs':  [  64, 64, 128, 256],
    # 'hid_chs':  [  32, 32, 64, 128],
    'kernel_sizes':[3, 3, 3, 3],
    'strides':     [1, 2, 2, 2],
    'time_embedder':TimeEmbbeding,
    'time_embedder_kwargs': {'emb_dim': 1024},
    # 'size_embedder':size_embedder,
    # 'size_embedder_kwargs': size_embedder_kwargs,
    'cond_embedder':Latent_Embedder,
    'cond_embedder_kwargs': {
        'in_channels': 9,
        'emb_channels': 8,
        'strides' : [ 1,  1,   1,   1],
        'hid_chs' : [32, 64, 128,  256],
    },
    'deep_supervision': False,
    'use_res_block':True,
    'use_attention':'none',
    'masked_condition': False
}

# ------------ Initialize Noise ------------
noise_scheduler = GaussianNoiseScheduler
noise_scheduler_kwargs = {
    'timesteps': 1000,
    'beta_start': 0.002, # 0.0001, 0.0015
    'beta_end': 0.02, # 0.01, 0.0195
    'schedule_strategy': 'scaled_linear'
}


latent_embedder = VAE # VQVAE: "/home/local/PARTNERS/rh384/runs/VAE/epoch=114-step=23000.ckpt"
latent_embedder_checkpoint = "/mnt/data/lesion_insertion/stroke0520/VAE256/2025_05_21_132417/epoch=903-step=226000.ckpt"

# ------------ Initialize Pipeline ------------
pipeline_lesion = DiffusionPipeline0121(
    noise_estimator=noise_estimator, 
    noise_estimator_kwargs=noise_estimator_kwargs,
    noise_scheduler=noise_scheduler, 
    noise_scheduler_kwargs = noise_scheduler_kwargs,
    latent_embedder=latent_embedder,
    latent_embedder_checkpoint = latent_embedder_checkpoint,
    estimator_objective='x_T',
    estimate_variance=False, 
    use_self_conditioning=False, 
    num_samples = 1,
    use_ema=False,
    classifier_free_guidance_dropout=0.5, # Disable during training by setting to 0
    optimizer_kwargs={'lr':1e-4}, # stable-diffusion ~ 1e-4
    do_input_centering=False,
    clip_x0=False,
    # sample_every_n_steps=save_and_sample_every,
    masked_condition=masked_condition
)

# ------------ Load Model ------------
# pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)
# pipeline = DiffusionPipeline.load_from_checkpoint("./medfusion_3d/runs/LDM_VQGAN/2024_06_07_115628/epoch=199-step=9999.ckpt") #/home/local/PARTNERS/rh384/runs/LDM/epoch=119-step=24000.ckpt")

ckpt_path = '/mnt/data/lesion_insertion/stroke0520/diffusion_generate_lesion_8ch/2025_05_23_022236/epoch=2440-step=524700.ckpt'
#'./medfusion_3d/runs/LDM_VQGAN/2024_06_07_175241/epoch=1079-step=53999.ckpt'
pipeline_lesion.load_pretrained(Path(ckpt_path))

pipeline_lesion.to(device)







#%%



inputfolder = "/mnt/data/ctsinogram/nii2/test/an2"
targetfolder = "/mnt/data/ctsinogram/nii2/test/original2"
savefolder = '/workspace/lesion_insertion/Mask2PET3D_2/scripts/AISD/results_generated_lesion3D2'


stats_csv_path = "/workspace/lesion_insertion/Mask2PET3D_2/ctsinogram/normalization.csv"
histogram_csv = "/workspace/lesion_insertion/Mask2PET3D_2/AISD_preprocessing3D/histogram.csv"


input_files = set(f for f in os.listdir(inputfolder) if f.endswith(".nii") or f.endswith(".nii.gz"))
target_files = set(f for f in os.listdir(targetfolder) if f.endswith(".nii") or f.endswith(".nii.gz"))
common_files = sorted(input_files & target_files)

print(f"共找到 {len(common_files)} 对匹配的文件")
#%%


s = 5000

for fname in common_files:
    s =  random.randint(1000, 5000)
    path_input = os.path.join(inputfolder, fname)
    path_target = os.path.join(targetfolder, fname)

    img_an = nib.load(path_input).get_fdata()
    img_original = nib.load(path_target).get_fdata()
    img_original2 = img_original
    # img_original = img_original * 1000
    img_original[img_original<-100]=-100
    img_original[img_original>200]=200
    img_original = img_original/100


    with open("/workspace/lesion_insertion/Mask2PET3D_2/AISD_preprocessing3D/lesion_distribution_256.json", "r") as f:
        lesion_distribution = json.load(f)

    # 随机选择
    central_slice = np.random.choice(lesion_distribution["max_mask_slice"]["values"])
    size = np.random.choice(lesion_distribution["max_mask_size"]["values"])
    num_slice = np.random.choice(lesion_distribution["nonzero_slice_count"]["values"])
    size = s
    print('central_slice', central_slice, 'size', size, 'num_slice', num_slice)
    # size, central_slice, num_slice = 1000, 10, 5

    img_an2d = img_an[:,:,central_slice]
    img_original2d = img_original[:,:,central_slice]

    size = torch.tensor(size).unsqueeze(0).float().to(device)  # shape: [1, 1, 256, 256]
    img_an2d = torch.tensor(img_an2d).unsqueeze(0).unsqueeze(0).float().to(device) 

    results = pipeline_mask.sample(1, (1, 256, 256), size, condition=img_an2d, guidance_scale=1,  steps=250, use_ddim=True )

    print(results.shape)
    mask = results.squeeze(0).squeeze(0).detach().cpu().numpy()
    mask[mask>=0.5] = 1
    mask[mask<0.5] = 0


    # base_name = os.path.splitext(os.path.splitext(fname)[0])[0]  # 去除 .nii.gz
    # subfolder = os.path.join(savefolder, base_name)
    # os.makedirs(subfolder, exist_ok=True)
    
    # save_path = os.path.join(subfolder, f"mask.png")
    # plt.imsave(save_path, mask, cmap='gray')

    image_generate = img_original
    image_mask = np.zeros((256, 256, 32))

    for i in range(32):   
        if i > (central_slice - num_slice // 2) and i < (central_slice + num_slice // 2):
            eroded_mask = binary_erosion(mask, structure=np.ones((3, 3)), iterations=np.abs(i - central_slice))
            image_mask[:,:,i] = eroded_mask
            
            # eroded_mask = torch.tensor(eroded_mask).unsqueeze(0).unsqueeze(0).float().to(device)  # shape: [1, 1, 256, 256]
        if i == central_slice:
            image_mask[:,:,i] = mask
        image_mask[:,:,i] = image_mask[:,:,i] * img_an[:,:,i]
            # eroded_mask = torch.tensor(eroded_mask).unsqueeze(0).unsqueeze(0).float().to(device)  # shape: [1, 1, 256, 256]
    base_name = os.path.splitext(os.path.splitext(fname)[0])[0]  # 去除 .nii.gz
    subfolder = os.path.join(savefolder, base_name)
    os.makedirs(subfolder, exist_ok=True)
    save_path = os.path.join(subfolder, f"generate_mask.png")
    save_3d_volume_as_png(image_mask, save_path, v_min = 0, v_max = 1)
    # save_3d_volume_as_png(image_mask, save_path, v_min = 0, v_max = 1)

    folder_name = os.path.basename(os.path.normpath(subfolder))  # e.g., "Patient001"
    save_path = os.path.join(subfolder, f"{folder_name}_mask.nii.gz")
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(image_mask.astype(np.float32), affine=affine)
    nib.save(nii_img, save_path)



    base_name = os.path.splitext(os.path.splitext(fname)[0])[0]  # 去除 .nii.gz
    subfolder = os.path.join(savefolder, base_name)
    os.makedirs(subfolder, exist_ok=True)


    img_original_save = img_original2
    # img_original_save = reverse_histogram_equalization(img_original_save, histogram_csv)
    save_path = os.path.join(subfolder, f"image_original.png")  
    save_3d_volume_as_png(img_original_save, save_path, v_min = 0, v_max = 100)

    for i in range(32):
        img_original2d = img_original[:,:,i]
        img_an2d = img_an[:,:,i]
        eroded_mask = image_mask[:,:,i]
        if i > (central_slice - num_slice // 2) and i < (central_slice + num_slice // 2):

            img_original2d[img_original2d>3] = 3

            img_original2d = torch.tensor(img_original2d).unsqueeze(0).unsqueeze(0).float().to(device) 
            eroded_mask = torch.tensor(eroded_mask).unsqueeze(0).unsqueeze(0).float().to(device) 

            x_0 = pipeline_lesion.latent_embedder.encode(img_original2d)


            masked_x_0 = img_original2d * (1 - eroded_mask)
            masked_x_0 = pipeline_lesion.latent_embedder.encode(masked_x_0)
            condition = F.interpolate(eroded_mask, (32,32))
            condition = torch.cat([masked_x_0, condition], dim=1)

            condition2 = F.interpolate(eroded_mask, (32,32))

# pipeline_mask.sample(1, (1, 256, 256), size, condition=img_an2d, guidance_scale=1,  steps=250, use_ddim=True )
            results = pipeline_lesion.sample(1, (8, 32, 32), x0 = x_0, condition=condition, condition2 = condition2, guidance_scale=1,  steps=250, use_ddim=True).detach()
            # sample(1, (1, 256, 256), size, condition=img_an2d, guidance_scale=1,  steps=250, use_ddim=True )

            print(results.shape)
            mask = results.squeeze(0).squeeze(0).detach().cpu().numpy()
            # img_original2d2 = img_original2d.squeeze(0).squeeze(0).detach().cpu().numpy()
            # img_original2d2 = img_original2d2 * 1000

            # img_original2d2 = reverse_histogram_equalization(img_original2d2, histogram_csv)
            # mask = restore_single_nii(fname, mask, stats_csv_path)

            # mask = mask * 1000

            # mask = reverse_histogram_equalization(mask, histogram_csv)

            # mask = mask * (1 - img_an2d) + img_an2d * img_original2d2

            image_generate[:,:,i] = mask



            # mask = mask * img * 20 + mask * (1 - img)      

    # base_name = os.path.splitext(os.path.splitext(fname)[0])[0]  # 去除 .nii.gz
    # subfolder = os.path.join(savefolder, base_name)
    # os.makedirs(subfolder, exist_ok=True)
    # save_path = os.path.join(subfolder, f"generate_mask.png")

    # save_3d_volume_as_png(image_mask, save_path, v_min = 0, v_max = 1)
    image_generate = image_generate * 100

    # image_generate = reverse_histogram_equalization(image_generate, histogram_csv)
    # save_path = os.path.join(subfolder, f"generate_image1.png")
    # save_3d_volume_as_png(image_generate, save_path, v_min = 0, v_max = 100)

    for i in range(32):

        image_generate[:,:,i] = image_generate[:,:,i] * image_mask[:,:,i] + img_original_save[:,:,i] * (1 - image_mask[:,:,i])
            # mask = mask * (1 - img_an2d) + img_an2d * img_original2d2

    save_path = os.path.join(subfolder, f"generate_image2.png")
    save_3d_volume_as_png(image_generate, save_path, v_min = 0, v_max = 100)
            # plt.imsave(save_path, mask, cmap='gray', vmin = -0, vmax = 100)
    folder_name = os.path.basename(os.path.normpath(subfolder))  # e.g., "Patient001"
    save_path = os.path.join(subfolder, f"{folder_name}.nii.gz")
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(image_generate.astype(np.float32), affine=affine)
    nib.save(nii_img, save_path)
            # save_path = os.path.join(subfolder, f"original.png")
            # plt.imsave(save_path, img_original, cmap='gray', vmin = -0, vmax = 100)  






# sample_img = self.sample(num_samples=self.num_samples, img_size=x_0.shape[1:], s=s_sample, condition=sample_cond).detach()  

# %%
