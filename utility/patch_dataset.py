import os
from os import listdir
from os.path import isfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import random


def get_loader(config, parse_patches=True, val=False):
    if val:
        data_path = os.path.join(config.data.data_dir, "test_256")
        dset = PatchDataset(data_path, n = config.training.patch_n, 
                            patch_size = config.data.image_size, 
                            transforms = T.Compose([T.ToTensor()]), 
                            parse_patches = parse_patches, 
                            mode = 'val')
        if not parse_patches:
            config.sampling.batch_size = 1
        return DataLoader(dset, batch_size=config.sampling.batch_size, shuffle=False, num_workers=config.data.num_workers, pin_memory=True)
    else:
        data_path = os.path.join(config.data.data_dir, 'train_256')
        dset = PatchDataset(data_path, n = config.training.patch_n, 
                            patch_size = config.data.image_size, 
                            transforms = T.Compose([T.ToTensor()]), 
                            parse_patches = parse_patches, 
                            mode = 'train')
        if not parse_patches:
            config.training.batch_size = 1        
        return DataLoader(dset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.data.num_workers, pin_memory=True)


class PatchDataset(Dataset):
    def __init__(self, data_path: str, patch_size: int, n, transforms: T, parse_patches=True, mode='train' or 'val'):
        super().__init__()

        self.path = data_path
        self.mode = mode
        if self.mode == 'train':
            self.input_dir = os.path.join(data_path, 'gt')
        elif self.mode == 'val':
            self.input_dir = os.path.join(data_path, 'input')
            self.gt_dir = os.path.join(data_path, 'gt')
        else:
            raise ValueError("mode must be either 'train' or 'val'")
        
        self.input_names = [f for f in listdir(self.input_dir) if isfile(os.path.join(self.input_dir, f))]
        self.gt_names = [f for f in listdir(self.gt_dir) if isfile(os.path.join(self.gt_dir, f))] if mode == 'val' else None
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        if img is None:
            return None
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        img_id = int(input_name[:-4])
        input_img = Image.open(os.path.join(self.input_dir, input_name)).convert('L')
        gt_img = None
        if self.mode == "val":
            gt_name = self.gt_names[index]
            gt_img = Image.open(os.path.join(self.gt_dir, gt_name)).convert('L')
        
        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            
            input_tensor = torch.stack([self.transforms(i) for i in input_img], dim=0)
            gt_tensor = torch.stack([self.transforms(i) for i in gt_img], dim=0) if self.mode == "val" else None
            
            if self.mode == "train":
                return input_tensor, img_id  
            else:
                return input_tensor, gt_tensor, img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            input_img = input_img.resize((wd_new, ht_new), Image.LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), Image.LANCZOS) if self.mode == "val" else None

            if self.mode == "val" :
                return self.transforms(input_img), self.transforms(gt_img), img_id 
            return self.transforms(input_img), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)