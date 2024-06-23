import os
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid

from src import NaiveUnet, UNet, UNetModel, MinDDPM, DDPM, fwt2, iwt2
from utility import Config, Grayscale_Dataset_Solo, Grayscale_Dataset
'''
from src.base_ddpm import MinDDPM
from src.base_unet import NaiveUnet
from src.ddpm import DDPM
from src.unet import UNet
from src.wavelet import fwt2, iwt2
from utility.config import Config
from utility.dataset import Grayscale_Dataset_Solo, Grayscale_Dataset
'''

def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')

def create_model(naive: bool) -> nn.Module:
    '''
    Wrapper function to create the model based on the configuration.
    :param naive: Whether to create the naive model or not.    
    :return model: The model to be trained.
    '''
    if naive:
        return NaiveUnet(**Config.naive_model())
    return UNet(**Config.model())

def create_model_copied(
    image_size=64,
    num_channels = 128,
    num_res_blocks=2,
    learn_sigma=False,
    class_cond=None,
    use_checkpoint=False,
    attention_resolutions="16,8",
    num_heads=4,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0.0,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

def create_ddpm(eps_model: nn.Module, min: bool) -> nn.Module:
    '''
    Wrapper function to create the DDPM model based on the configuration.
    :param eps_model: UNet to be used.
    :param min: Whether to create the minimal DDPM model or not.
    :return ddpm: The DDPM model to be trained.
    '''
    if min:
        return MinDDPM(eps_model=eps_model, **Config.min_ddpm())
    return DDPM(eps_model=eps_model, **Config.ddpm())

def train(
            diffusion: nn.Module,
            data: DataLoader,
            optimizer: torch.optim.Adam,
            device: torch.device,
        ) -> float:
    '''
    The training loop for the DDPM model.
    :param data: The dataset to be used for training.
    :param current_epoch: The current epoch number.
    :param device: The device to be used for the model.
    :return avg_loss: The loss value for the current epoch.
    '''
    loss_ema = None
    for _, x in enumerate(data):
        ll, (lh, hl, hh) = fwt2(x, wave='haar', level=1)
        model_in = torch.cat((lh, hl, hh), dim=1).to(device, dtype=torch.float32)
        optimizer.zero_grad()
        eps, eps_theta = diffusion.loss(model_in)
        diff_loss = F.mse_loss(eps, eps_theta)
        #model_out = torch.tensor_split(eps_theta, 3, dim = 1)
        #model_out = iwt2([ll, model_out], wave='haar')
        #spatial_loss = F.mse_loss(model_out, x)
        loss = diff_loss #+ spatial_loss * 0.01
        loss.backward()
        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
        optimizer.step()
    
    return loss_ema

def sample(
            diffusion: nn.Module,
            device: torch.device,
            n_samples: int,
            current_epoch: int,
            save_dir: str,
            val_data: Optional[DataLoader] = None,
            ) -> List[torch.Tensor]:
        """
        ### Sample images
        """
        with torch.no_grad():
            assert val_data is not None, 'Validation data is required for PSNR calculation.'
            wavelet_psnr_vals = []
            for i, (x,y) in enumerate(val_data):
                if i == n_samples:
                    break    
                ll_x, (lh, hl, hh), (h1,h2,h3) = fwt2(x, wave='haar', level=2)
                x_cond = torch.cat((lh, hl, hh), dim=1).to(device, dtype=torch.float32)
                x_cond_2 = torch.cat((h1, h2, h3), dim=1).to(device, dtype=torch.float32)
                _, (lh, hl, hh), (h1,h2,h3) = fwt2(y, wave='haar', level=2)
                gt = torch.cat((lh, hl, hh), dim=1).to(device, dtype=torch.float32)
                gt_2 = torch.cat((h1, h2, h3), dim=1).to(device, dtype=torch.float32)
                peak_wavelet_psnr = [0.0, 0, torch.zeros_like(x_cond)]
                for t in range(diffusion.n_steps):
                    t = diffusion.n_steps - t - 1
                    x_cond = diffusion.p_sample(x_cond, x_cond.new_full((n_samples,), t, dtype=torch.long))
                    x_cond_2 = F.interpolate(diffusion.p_sample(x_cond, x_cond_2.new_full((n_samples,), t, dtype=torch.long)), scale_factor=2)
                    timestep_psnr = 10 * torch.log10(1 / (F.mse_loss(gt, x_cond) + F.mse_loss(gt_2, x_cond_2)))
                    if peak_wavelet_psnr[0] < timestep_psnr:
                        peak_wavelet_psnr[0] = timestep_psnr
                        peak_wavelet_psnr[1] = t
                        peak_wavelet_psnr[2] = x_cond
                wavelet_psnr_vals.append(peak_wavelet_psnr)
                coeff_1 = torch.tensor_split(x_cond.to(ll_x.device, dtype=ll_x.dtype), 3, dim = 1)
                coeff_2 = torch.tensor_split(x_cond_2.to(ll_x.device, dtype=ll_x.dtype), 3, dim = 1)
                x_rec = iwt2([ll_x, coeff_1, coeff_2], wave='haar')
                grid = make_grid(torch.cat([x, x_rec, y], dim=0), nrow=3)
                save_image(grid, os.path.join(save_dir, f"epoch_{current_epoch}.png"))
        return wavelet_psnr_vals

def train_loop(
                diffusion: nn.Module,
                train_data: DataLoader,
                val_data: DataLoader,
                device: torch.device,
                num_epochs: int,
                optimizer: torch.optim.Adam,
                n_samples: int,
                writer: SummaryWriter,
                save_dir: str,
                ):
    pbar = tqdm(range(num_epochs))
    for _,epoch in enumerate(pbar):
        loss = train(diffusion, train_data, optimizer, device)
        pbar.set_description(f"Epoch {epoch} : loss {loss:.4f}")
        writer.add_scalar("Loss / epoch", loss, epoch)
        [best_psnr, best_timestep, _] = max(sample(diffusion, device, n_samples, epoch, val_data, save_dir), key=lambda x: x[0])
        writer.add_scalar("PSNR (dB) / epoch", best_psnr, epoch)
        writer.add_histogram("Best timestep", best_timestep)
        
        if epoch % 5 == 0:
            torch.save(diffusion.state_dict(), f"ckpts/ddpm_deep_epoch_{epoch}.pth")

def sample_loop(
                diffusion: nn.Module,
                val_data: DataLoader,
                device: torch.device,
                n_samples: int,
                num_iters: int,
                save_dir: str,
                ):
    pbar = tqdm(range(num_iters))
    for _,iter in enumerate(pbar):
        [best_psnr, _, _] = max(sample(diffusion, device, n_samples, iter, val_data = val_data, save_dir = save_dir), key=lambda x: x[0])
        pbar.set_description(f"Iteration {iter} : PSNR {best_psnr:.4f}")