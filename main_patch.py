import argparse
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src import DenoisingDiffusion
from utility import Grayscale_Dataset_Solo, Grayscale_Dataset, get_loader


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, default="patch_large.yml",
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument('--log_dir', default='logs', type=str,
                        help='Directory for logging')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


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
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, config.data.exp_name))
    os.makedirs(config.data.save_dir, exist_ok=False) if not os.path.exists(config.data.save_dir) else None
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    # train_data = DataLoader(Grayscale_Dataset_Solo(os.path.join(config.data.data_dir, 'train_256', 'gt'), 256), batch_size=config.training.batch_size, shuffle=True, num_workers=config.data.num_workers)
    # val_data = DataLoader(Grayscale_Dataset(os.path.join(config.data.data_dir, "test_256"), 256), batch_size=1, shuffle=True, num_workers=config.data.num_workers)
    
    train_data_2 = get_loader(config, val=False)
    val_data_2 = get_loader(config, val=True)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    diffusion.train(train_data_2, val_data_2, writer)


if __name__ == "__main__":
    main()