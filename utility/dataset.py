import os
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from src import HFRM, fwt2

class Grayscale_Dataset(Dataset):
    def __init__(self, data_path: str, img_size: int = 128, if_crop: bool = False):
        super(Grayscale_Dataset, self).__init__()
        self.input_dir = os.path.join(data_path, 'input')
        self.gt_dir = os.path.join(data_path, 'gt')
        self.img_names = [f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))]
        self.img_size = img_size
        self.if_crop = if_crop

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_dir, img_name)).convert('L')
        gt_img = Image.open(os.path.join(self.gt_dir, img_name)).convert('L')

        # input_img = refine(T.ToTensor()(input_img).unsqueeze(0)).squeeze(0)
        gt_img = T.ToTensor()(gt_img).to(torch.float32)
        gt_img = gt_img / 255.0 if gt_img.max() > 1.0 else gt_img
        input_img = T.ToTensor()(input_img).to(torch.float32)
        input_img = input_img / 255.0 if input_img.max() > 1.0 else input_img
        if self.if_crop:
            i, j = self.generate_coord()
            input_img = self.crop(input_img, (i, j))
            gt_img = self.crop(gt_img, (i, j))

        return T.Resize((self.img_size, self.img_size))(input_img), T.Resize((self.img_size, self.img_size))(gt_img)

    def crop(self, img, coord: Tuple[int]):
        i, j = coord
        crop_size = self.img_size // 4
        patch = img[:, i:i+crop_size, j:j+crop_size]
        return patch
    
    def generate_coord(self):
        crop_size = self.img_size // 4
        i = torch.randint(0, self.img_size - crop_size, (1,)).item()
        j = torch.randint(0, self.img_size - crop_size, (1,)).item()
        
        return i, j
        

def get_dataloaders(args):
    train_dataset = Grayscale_Dataset(os.path.join(args.data_path, 'train_256'), 256, if_crop=True)
    val_dataset = Grayscale_Dataset(os.path.join(args.data_path, 'test_256'), 256)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=3)
    return train_dataloader, val_dataloader

def refine(img: torch.Tensor):
    img_channel = 3
    dim = 32
    enc_blks = [2, 2, 2, 4]
    middle_blk_num = 6
    dec_blks = [2, 2, 2, 2]
    generator = HFRM(in_channel=img_channel, dim=dim, mid_blk_num=middle_blk_num,enc_blk_nums=enc_blks,dec_blk_nums=dec_blks).to(img.device).eval()
    generator.load_state_dict(torch.load(f"ckpts/hfrm_best.pth", map_location=img.device),strict=True)
    generator.requires_grad_(False)
    return generator(img)[:,:1,:, :]

class Grayscale_Dataset_Solo(Dataset):
    def __init__(self, data_path: str, img_size: int = 128, if_crop: bool = False, level: int = 2):
        super(Grayscale_Dataset_Solo, self).__init__()
        self.img_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        self.root = data_path
        self.img_size = img_size
        self.if_crop = if_crop
        self.level = level

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = Image.open(os.path.join(self.root, img_name)).convert('L')
        img = T.ToTensor()(img)
        
        if self.if_crop:
            i, j = self.generate_coord()
            img = self.crop(img, (i, j))
        return T.Resize((self.img_size, self.img_size))(img)

    def crop(self, img, coord: Tuple[int]):
        i, j = coord
        crop_size = self.img_size // 4
        patch = img[:, i:i+crop_size, j:j+crop_size]
        return patch
    
    def generate_coord(self):
        crop_size = self.img_size // 4
        i = torch.randint(0, self.img_size - crop_size, (1,)).item()
        j = torch.randint(0, self.img_size - crop_size, (1,)).item()
        
        return i, j

def load_data(args, val: bool = False):
    if not val:
        dset = Grayscale_Dataset_Solo(os.path.join(args.data_path, 'train_256', 'gt'), 256)
        return DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        dset = Grayscale_Dataset(os.path.join(args.data_path, "test_256"), 256)
        return DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)