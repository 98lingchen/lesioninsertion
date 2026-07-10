
import torch.utils.data as data 
from pathlib import Path 
from torchvision import transforms as T

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from glob import glob
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os
import scipy.ndimage as snd
import torchio as tio 

from medical_diffusion.data.augmentation.augmentations_3d import ImageToTensor


class SimpleDataset3D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = ['nii'], # other options are ['nii.gz'],
        transform = None,
        image_resize = None,
        flip = False,
        image_crop = None,
        use_znorm=True, # Use z-Norm for MRI as scale is arbitrary, otherwise scale intensity to [-1, 1]
    ):
        super().__init__()
        self.path_root = path_root
        self.crawler_ext = crawler_ext

        if transform is None: 
            self.transform = T.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x),
                tio.CropOrPad(image_crop) if image_crop is not None else tio.Lambda(lambda x: x),
                tio.ZNormalization() if use_znorm else tio.RescaleIntensity((-1,1)),
                ImageToTensor() # [C, W, H, D] -> [C, D, H, W]
            ])
        else:
            self.transform = transform
        
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        return {'uid':rel_path_item.stem, 'source': self.transform(img)}
    
    def load_item(self, path_item):
        return tio.ScalarImage(path_item) # Consider to use this or tio.ScalarLabel over SimpleITK (sitk.ReadImage(str(path_item)))
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]
    

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 2,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        # Initialize the dataset with input folder, target folder, and various configurations
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()  # Generate list of paired input and target files
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    # def pair_file(self):
    #     # Pair input and target files based on their identifiers
    #     input_files = sorted(glob(os.path.join(self.input_folder, '*')))
    #     target_files = sorted(glob(os.path.join(self.target_folder, '*_0000.nii.gz')))
    #     pairs = []
    #     for input_file, target_file in zip(input_files, target_files):
    #         # Ensure the input and target files are correctly paired by comparing their identifiers
    #         assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file))[:-4])
    #         pairs.append((input_file, target_file))
    #     return pairs
    # def pair_file(self):
    #     # 查找输入和目标文件并进行配对
    #     input_files = sorted(glob(os.path.join(self.input_folder, '*.nii')) + 
    #            glob(os.path.join(self.input_folder, '*.nii.gz')))
    #     target_files = sorted(glob(os.path.join(self.target_folder, '*.nii')) + 
    #            glob(os.path.join(self.target_folder, '*.nii.gz')))
    #     pairs = []
    #     for input_file, target_file in zip(input_files, target_files):
    #         # print(input_file, target_file)
    #         if int("".join(re.findall("\d", input_file))[-6:]) == int("".join(re.findall("\d", target_file))[-6:]):
    #             pairs.append((input_file, target_file))
    #             # print(1)
    #     return pairs
    def pair_file(self):
        # 查找输入和目标文件并进行配对
        input_files = sorted(glob(os.path.join(self.input_folder, '*.nii')) + 
            glob(os.path.join(self.input_folder, '*.nii.gz')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*.nii')) + 
            glob(os.path.join(self.target_folder, '*.nii.gz')))
        pairs = []
        # print(input_files)
        for input_file, target_file in zip(input_files, target_files):
            # 检查文件名中的数字是否匹配
            if int("".join(re.findall("\d", input_file))[-6:]) == int("".join(re.findall("\d", target_file))[-6:]):
                pairs.append((input_file, target_file))
            
            # # 限制最多配对 20 对
            # if len(pairs) >= 350:
            #     break
                
        return pairs

    def read_image(self, file_path, pass_scaler=False):
        # Load and preprocess the image using nibabel
        img = nib.load(file_path).get_fdata()
        # img = img.clip(min=0)  # Clip values to be non-negative
        img = np.expand_dims(img, 0)  # Add channel dimension
        return img

    def plot(self, index, n_slice=30):
        # Plot a specific slice from the input and target images
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        # Resize a 3D image to match the specified input size and depth
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])  # Convert to TorchIO ScalarImage
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))  # Resize to target dimensions
            img = np.asarray(cop(img))[0]  # Convert back to numpy array
        return img

    def resize_img_4d(self, input_img):
        # Resize a 4D image (with channels) to match the specified dimensions
        c, h, w, d = input_img.shape
        scaled_img = snd.zoom(input_img, [c, self.input_size / h, self.input_size / w, self.depth_size / d])
        return scaled_img.clip(min=0)  # Clip values to be non-negative

    def resize_img_4d_pad(self, input_img):
        # Pad a 4D image to match the specified dimensions
        c, h, w, d = input_img.shape
        pad_one_side = (self.input_size - h) // 2
        padding = [(0, 0), (pad_one_side, pad_one_side), (pad_one_side, pad_one_side), (pad_one_side, pad_one_side)]
        scaled_img = np.pad(input_img, padding, mode='constant', constant_values=0)  # Pad with zeros
        return scaled_img.clip(min=0)

    def resize_img_4d_01(self, input_img):
        # Resize a 4D image using nearest-neighbor interpolation and binarize the result
        c, h, w, d = input_img.shape
        scaled_img = snd.zoom(input_img, [c, self.input_size / h, self.input_size / w, self.depth_size / d], order=0)
        scaled_img = np.where(scaled_img > 0.5, 1, 0)  # Binarize the image based on a threshold of 0.5
        return scaled_img

    def sample_conditions(self, batch_size: int):
        # Sample a batch of conditions from the dataset for conditional training
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = np.expand_dims(input_img, 0)  # Add batch dimension
            # input_img = self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)  # Apply transformation and add batch dimension
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()  # Concatenate tensors and move to GPU
    def resize_img2(self, img):
        # 获取图像尺寸
        c, w, h, d= img.shape
        # print(img.shape)

        # 目标尺寸
        target_h, target_w, target_d = self.input_size, self.input_size, self.depth_size

        # 如果图像在某个维度上大于目标尺寸，进行裁剪
        if h > target_h or w > target_w or d > target_d:
            # 裁剪多余的通道，只保留前 target_d 层
            img = img[:, :target_h, :target_w, :target_d]

        return img

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.pair_files)

    def __getitem__(self, index):
        # Get a specific sample from the dataset
        input_file, target_file = self.pair_files[index]
        # input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
        input_img = self.read_image(input_file)
        # input_img = self.resize_img_4d(input_img) # if not self.full_channel_mask else self.resize_img_4d(input_img)
        # input_img = self.resize_img_4d_01(input_img)  # Resize and binarize the input image
        input_img = self.resize_img2(input_img)
        target_img = self.read_image(target_file)
        target_img = self.resize_img2(target_img)
        # target_img = self.resize_img_4d(target_img)  # Resize the target image
        # target_img = target_img / target_img.max()  # Normalize target image to range [0, 1]
        # print(target_img.min())
        # Apply transformations if defined
        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        # Combine input and target if specified
        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input': input_img, 'target': target_img}
        # return {'input': torch.tensor(input_img, dtype=torch.float32), 'target': torch.tensor(target_img, dtype=torch.float32)}




class NiftiPairImageGenerator2D(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            slice_axis: int = 2,  # 切片方向，0、1、2 分别表示 x、y、z 轴
            input_channel: int = 2,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.slice_axis = slice_axis
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        # 查找输入和目标文件并进行配对
        input_files = sorted(glob(os.path.join(self.input_folder, '*.nii')) + 
               glob(os.path.join(self.input_folder, '*.nii.gz')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*.nii')) + 
               glob(os.path.join(self.target_folder, '*.nii.gz')))
        pairs = []

        for input_file, target_file in zip(input_files, target_files):
            input_name = os.path.basename(input_file).replace(".nii.gz", "")
            target_name = os.path.basename(target_file).replace(".nii.gz", "")
            
            if input_name == target_name:
                pairs.append((input_file, target_file))

        return pairs

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        # img = img.clip(min=0)  # 将负值裁剪为 0
        return img

    def plot(self, index, slice_idx=30):
        # 用于可视化 2D 切片
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(target_img, cmap='gray')
        plt.show()

    def resize_img(self, img):
        # 这里可以根据需要对 2D 图像进行缩放，比如使用 cv2 或 scipy
        # 假设 img 为 (H, W) 形状
        h, w = img.shape
        if h != self.input_size or w != self.input_size:
            img = snd.zoom(img, [self.input_size / h, self.input_size / w])
        # return img.clip(min=0)
        return img
    def __len__(self):
        # 返回数据集中样本对的数量
        return len(self.pair_files) * self.get_num_slices()

    def get_num_slices(self):
        # 获取每个 3D 图像中切片的数量
        # print(len(self.pair_files))
        # sample_img = nib.load(self.pair_files[0][0]).get_fdata()
        # return sample_img.shape[self.slice_axis]
        return 1
    def __getitem__(self, index):
        # 计算文件索引和切片索引
        file_idx = index // self.get_num_slices()
        slice_idx = index % self.get_num_slices()

        input_file, target_file = self.pair_files[file_idx]
        
        # 读取图像并提取切片
        input_img = self.read_image(input_file)
        input_img[input_img>3]=3
        
        input_img = self.resize_img(input_img)
        
        target_img = self.read_image(target_file)
        target_img[target_img>3]=3

        target_img = self.resize_img(target_img)
        
        # print(input_img.shape)

        # if self.slice_axis == 0:
        #     input_img = input_img_3d[slice_idx, :, :]
        #     target_img = target_img_3d[slice_idx, :, :]
        # elif self.slice_axis == 1:
        #     input_img = input_img_3d[:, slice_idx, :]
        #     target_img = target_img_3d[:, slice_idx, :]
        # elif self.slice_axis == 2:
        #     input_img = input_img_3d[:, :, slice_idx]
        #     target_img = target_img_3d[:, :, slice_idx]
        # else:
        #     raise ValueError("Invalid slice axis. Must be 0, 1, or 2.")
        
        # # 归一化和裁剪
        # input_img = self.resize_img(input_img)
        # target_img = self.resize_img(target_img)
        # target_img = target_img / (target_img.max() if target_img.max() != 0 else 1)

        # 进行 transform 操作
        if self.transform is not None:
        # 确保以字典形式传递输入数据
            augmented = self.transform(image=input_img)
            input_img = augmented["image"]  # 提取增强后的图像

        if self.target_transform is not None:
            # 确保以字典形式传递目标数据
            augmented = self.target_transform(image=target_img)
            target_img = augmented["image"]  # 提取增强后的目标图像


        # 添加通道维度 (C, H, W)
        input_img = np.expand_dims(input_img, axis=0)
        target_img = np.expand_dims(target_img, axis=0)

        if self.combine_output:
            return torch.cat([torch.tensor(target_img, dtype=torch.float32), torch.tensor(input_img, dtype=torch.float32)], dim=0)

        return {'input': torch.tensor(input_img, dtype=torch.float32), 'target': torch.tensor(target_img, dtype=torch.float32), 'input_file': input_file}



class NiftiPairImageGenerator2D2(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            slice_axis: int = 2,  # 切片方向，0、1、2 分别表示 x、y、z 轴
            input_channel: int = 2,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.slice_axis = slice_axis
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        # 获取两个目录下的所有 nii / nii.gz 文件
        input_files = sorted(glob(os.path.join(self.input_folder, '*.nii')) + 
                             glob(os.path.join(self.input_folder, '*.nii.gz')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*.nii')) + 
                              glob(os.path.join(self.target_folder, '*.nii.gz')))

        # 建立 target 文件名到路径的映射
        target_dict = {os.path.basename(f): f for f in target_files}

        pairs = []
        for input_file in input_files:
            basename = os.path.basename(input_file)
            if basename in target_dict:
                pairs.append((input_file, target_dict[basename]))

        return pairs

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        # img = img.clip(min=0)  # 将负值裁剪为 0
        return img

    def plot(self, index, slice_idx=30):
        # 用于可视化 2D 切片
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(target_img, cmap='gray')
        plt.show()

    def resize_img(self, img):
        # 这里可以根据需要对 2D 图像进行缩放，比如使用 cv2 或 scipy
        # 假设 img 为 (H, W) 形状
        h, w = img.shape
        if h != self.input_size or w != self.input_size:
            img = snd.zoom(img, [self.input_size / h, self.input_size / w])
        # return img.clip(min=0)
        return img
    def __len__(self):
        # 返回数据集中样本对的数量
        return len(self.pair_files) * self.get_num_slices()

    def get_num_slices(self):
        # 获取每个 3D 图像中切片的数量
        # print(len(self.pair_files))
        # sample_img = nib.load(self.pair_files[0][0]).get_fdata()
        # return sample_img.shape[self.slice_axis]
        return 1
    def __getitem__(self, index):
        # 计算文件索引和切片索引
        file_idx = index // self.get_num_slices()
        slice_idx = index % self.get_num_slices()

        input_file, target_file = self.pair_files[file_idx]
        
        # 读取图像并提取切片
        input_img = self.read_image(input_file)
        target_img = self.read_image(target_file)
        # print(input_img.shape)

        # if self.slice_axis == 0:
        #     input_img = input_img_3d[slice_idx, :, :]
        #     target_img = target_img_3d[slice_idx, :, :]
        # elif self.slice_axis == 1:
        #     input_img = input_img_3d[:, slice_idx, :]
        #     target_img = target_img_3d[:, slice_idx, :]
        # elif self.slice_axis == 2:
        #     input_img = input_img_3d[:, :, slice_idx]
        #     target_img = target_img_3d[:, :, slice_idx]
        # else:
        #     raise ValueError("Invalid slice axis. Must be 0, 1, or 2.")
        
        # # 归一化和裁剪
        # input_img = self.resize_img(input_img)
        # target_img = self.resize_img(target_img)
        # target_img = target_img / (target_img.max() if target_img.max() != 0 else 1)

        # 进行 transform 操作
        if self.transform is not None:
        # 确保以字典形式传递输入数据
            augmented = self.transform(image=input_img)
            input_img = augmented["image"]  # 提取增强后的图像

        if self.target_transform is not None:
            # 确保以字典形式传递目标数据
            augmented = self.target_transform(image=target_img)
            target_img = augmented["image"]  # 提取增强后的目标图像


        # 添加通道维度 (C, H, W)
        input_img = np.expand_dims(input_img, axis=0)
        target_img = np.expand_dims(target_img, axis=0)

        if self.combine_output:
            return torch.cat([torch.tensor(target_img, dtype=torch.float32), torch.tensor(input_img, dtype=torch.float32)], dim=0)

        return {'input': torch.tensor(input_img, dtype=torch.float32), 'target': torch.tensor(target_img, dtype=torch.float32), 'input_file': input_file}

