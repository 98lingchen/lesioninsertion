# Synthesis of Cerebral Infarction and Hemorrhage on Brain CT via Latent Diffusion Models

This repository contains the implementation of a latent diffusion framework for synthesizing infarction and hemorrhage lesions on brain CT. The proposed method generates anatomically consistent pathological regions while preserving surrounding normal brain structures.

---

## Requirements

```bash
torch
torchvision
pytorch-lightning
numpy
nibabel
opencv-python
monai
einops
tqdm
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Datasets

The experiments were conducted using publicly available ischemic stroke and intracranial hemorrhage CT datasets.

### Ischemic Stroke

- **AISD (Acute Ischemic Stroke Dataset)**  
  https://github.com/griffinliang/aisd

### Intracranial Hemorrhage

- **INSTANCE 2022 (INtracranial Hemorrhage SegmenTAtion ChallengE)**  
  https://instance.grand-challenge.org/

- **BHSD (Brain Hemorrhage Segmentation Dataset)**  
  https://github.com/White65534/BHSD

### Normal Brain CT

- **SinoCT**  
  https://aimi.stanford.edu/datasets/sinoct

Please download the datasets and organize them according to the paths specified in the training scripts.

---

## Training

### Train the VAE

```bash
python scripts/train_image_vae.py
```

### Train the Lesion Diffusion Model

```bash
python scripts/train_image_diffusion.py
```

### Train the Mask Diffusion Model

```bash
python scripts/train_mask_diffusion.py
```

---

## Sampling

Generate synthetic lesions using trained checkpoints:

```bash
python scripts/sample.py
```

Generated images will be saved to the specified output directory.

---

## Pretrained Models

Pretrained checkpoints and related files can be downloaded from:

**GoogleDrive**

[GoogleDrive]((https://drive.google.com/drive/folders/1RTffWz4nZGbSj53-8iLRSSHc0Kj8bq6v?usp=sharing)).

Please place the downloaded checkpoints in the corresponding checkpoint directory before running inference.

---

