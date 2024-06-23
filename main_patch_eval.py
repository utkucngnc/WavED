import argparse
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from src import DenoisingDiffusion, DiffusiveRestoration
from utility import Grayscale_Dataset


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config_small", type=str, default="patch.yml",
                        help="Path to the config file")
    parser.add_argument("--config_large", type=str, default="patch_large.yml",
                        help="Path to the config file")
    parser.add_argument('--resume_small', default='ckpts/small_ddpm_epoch_119.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument('--resume_large', default='ckpts/large_ddpm_epoch_204.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--test_set", type=str, default='raindrop',
                        help="restoration test set options: ['raindrop', 'snow', 'rainfog']")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(args.config_small, "r") as f:
        config = yaml.safe_load(f)
    new_config_small = dict2namespace(config)
    
    with open(args.config_large, "r") as f:
        config = yaml.safe_load(f)
    new_config_large = dict2namespace(config)

    return args, new_config_small, new_config_large


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config_small, config_large = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config_small.device = device
    config_large.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    
    val_loader = DataLoader(Grayscale_Dataset(os.path.join(config_small.data.data_dir, "test_256"), 256), batch_size=1, shuffle=True, num_workers=config_small.data.num_workers)
    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion_small = DenoisingDiffusion(args, config_small)
    diffusion_large = DenoisingDiffusion(args, config_large)
    model = DiffusiveRestoration(diffusion_small, diffusion_large, args, config_small, config_large)
    model.restore(val_loader, r=args.grid_r)


if __name__ == '__main__':
    main()