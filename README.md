# Synthesis of Cerebral Infarction and Hemorrhage on Brain CT with  Latent Diffusion

This repository contains the implementation of our latent diffusion framework for Cerebral Infarction and Hemorrhage on Brain CT

### Requirements

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

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

## Datasets

The experiments were conducted using publicly available stroke and hemorrhage CT datasets.

- AISD (Acute Ischemic Stroke Dataset):
  [https://github.com/griffinliang/aisd](https://github.com/griffinliang/aisd)

- INSTANCE 2022 (INtracranial Hemorrhage SegmenTAtion ChallengE):
  [https://instance.grand-challenge.org/](https://instance.grand-challenge.org/)

- BHSD (Brain Hemorrhage Segmentation Dataset):
  [https://arxiv.org/abs/2308.11298](https://github.com/White65534/BHSD/tree/2ec4322a5a644494a312df891a3089bbf74c136a)

- SinoCT:
  [https://huggingface.co/datasets/yhygao/SinoCT](https://aimi.stanford.edu/datasets/sinoct)



## Training

### Train VAE

Run:

```bash
python scripts/train_image_vae.py
```

### Train Lesion Diffusion Model

Run:

```bash
python scripts/train_image_diffusion.py
```

### Train Mask Diffusion Model

Run:

```bash
python scripts/train_mask_diffusion.py
```

## Sampling

Generate synthetic lesions using a trained model:

```bash
python scripts/sample.py
```

Generated images will be saved to the specified output directory.

## Pretrained Models

## Pretrained Model The pretrained model and related files can be downloaded from [[ OneDrive]]([https://drive.google.com/drive/folders/1vqyvFI3SB4hDb--3-PMI_hV2g75NdNz2?usp=sharing](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/chen_15048_osu_edu/IgB6iBv0ySKWRqlemRw69shfAYC0hxa3sYOoK10Y_qdvNOo?e=DhMYat)).




## Acknowledgments

This work builds upon several excellent open-source projects, including diffusion models, latent diffusion models, and medical image generation frameworks. We thank the authors for making their code publicly available.
