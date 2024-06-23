import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utility import Config, Grayscale_Dataset_Solo, Grayscale_Dataset, create_model, create_ddpm, train_loop, create_model_copied, sample_loop

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'results/pristine_deep_sample_only_2'
    min = False
    state_dict = torch.load('ckpts/ddpm_deep_epoch_140.pth')
    eps_model = create_model_copied().to(device)
    diffusion = create_ddpm(eps_model, min).to(device)
    diffusion.load_state_dict(state_dict)
    # optimizer = torch.optim.Adam(eps_model.parameters(), lr=2e-5)
    # train_data = DataLoader(Grayscale_Dataset_Solo(os.path.join(Config.data_path, 'train_256', 'gt'), 256), batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    val_data = DataLoader(Grayscale_Dataset(os.path.join(Config.data_path, "test_256"), 256), batch_size=1, shuffle=True, num_workers=Config.num_workers)
    # writer = SummaryWriter(log_dir='logs/pristine_deep')
    os.makedirs(save_dir, exist_ok=True) if not os.path.exists(save_dir) else None
    # train_loop(diffusion, train_data, val_data, device, 600, optimizer, 1, writer, save_dir)
    sample_loop(diffusion, val_data, device, 1, 2, save_dir)


if __name__ == '__main__':
    main()