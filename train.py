import argparse
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'
import shutil
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from utility.dataset import load_data
from src import *

torch.backends.cudnn.benchmark = True

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Minimal DDPM')
    parser.add_argument('--resume', default="", type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--epoch", type=int, default=50,
                        help="Number of epochs to train the model")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size of the input image")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--data_path", default='F:\\data\\Pristine', type=str,
                        help="Location to training / validation data folder")
    parser.add_argument("--exp_name", default='pristine_3_deep', type=str,
                        help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for dataloader")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')

    args = parser.parse_args()
    
    return args

def check_duplicate_dir(dir_name: str) -> None:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

def train(args, small: bool = False) -> None:
    log_dir = os.path.join("logs", args.exp_name)
    model_name = "ddpm_small" if small else "ddpm_large"
    ckpts_save_dir = os.path.join("ckpts", args.exp_name)
    # check_duplicate_dir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
    if small:
        ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=args.image_size // 4), betas=(1e-4, 0.02), n_T=1000)
    else:
        ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=args.image_size // 2), betas=(1e-4, 0.02), n_T=1000)
    '''
    eps_model = create_model(args.image_size // 4)
    ddpm = DDPM(eps_model=eps_model, betas=(1e-4, 0.02), n_T=1000)
    if os.path.exists(args.resume):
        ddpm.load_state_dict(torch.load(args.resume))
    ddpm.to(device)

    data = load_data(args, val=False)
    val_data = load_data(args, val=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-5)

    for i in range(args.epoch):
        print(f"Epoch {i} : ")
        ddpm.train()                

        pbar = tqdm(data)
        loss_ema = None
        for k, y in enumerate(pbar):
            optim.zero_grad()
            y = y.to(device)
            y_wave = fwt2(y, wave='haar', level=2 if small else 1)
            y_wave = torch.cat([j for j in y_wave[1]], dim=1)
            loss = ddpm(y_wave)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            writer.add_scalar("Loss / step", loss.item(), k+ i * len(data))
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            pbar = tqdm(val_data)
            for k, (x_i, y_i) in enumerate(pbar):
                y_i = y_i.to(device)
                x_i = x_i.to(device)
                x_wave = fwt2(x_i, wave='haar', level=2 if small else 1)
                x_wave_hh = torch.cat([j for j in x_wave[1]], dim=1)
                xh = ddpm.sample(x_wave_hh, device)
                if small:
                    x_out = iwt2([x_wave[0], torch.tensor_split(xh, 3, dim = 1), x_wave[2]], wave='haar')
                    x_out_2 = iwt2([x_wave[0], torch.tensor_split(x_wave_hh + xh, 3, dim = 1), x_wave[2]], wave='haar')
                else:
                    x_out = iwt2([x_wave[0], torch.tensor_split(xh, 3, dim = 1)], wave='haar')
                    x_out_2 = iwt2([x_wave[0], torch.tensor_split(x_wave_hh + xh, 3, dim = 1)], wave='haar')
                xset = torch.cat([x_i, x_out, x_out_2, y_i], dim=0)
                diff_set = torch.abs(torch.cat([y_i - x_i, y_i - x_out, y_i - x_out_2, x_out_2 - x_out], dim=0))
                grid = make_grid(torch.cat([xset, diff_set], dim = 0), nrow=4)
                psnr = 10 * torch.log10(1 / torch.mean((y_i - x_out) ** 2))
                writer.add_scalar("PSNR (dB) / epoch", psnr, i)
                pbar.set_description("PSNR: {:.4f}".format(psnr.item()))
                save_dir = os.path.join("results")
                # check_duplicate_dir(save_dir) if i == 0 else None
                save_image(grid, os.path.join(save_dir, f"sample_pristine_epoch_{i:03d}.png"))
                break

            # save model
            if i%5 == 0:
                torch.save(ddpm.state_dict(), os.path.join(ckpts_save_dir, f"{model_name}_epoch_{i}.pth"))


if __name__ == "__main__":
    train(args=parse_args_and_config(), small=True)