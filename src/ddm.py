import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from src import DiffusionUNet, fwt2, iwt2
from utility import save_checkpoint, sampling
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import torch.optim as optim


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0, x], dim=1), t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                          betas=(0.9, 0.999), amsgrad=self.config.optim.amsgrad, eps=self.config.optim.eps)
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, train_loader, val_loader, writer):
        cudnn.benchmark = True

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (img, _) in enumerate(train_loader):
                img = img.flatten(start_dim=0, end_dim=1) if img.ndim == 5 else img
                if self.config.data.small:
                    _, (lh, hl, hh), _ = fwt2(img, wave='haar', level=2)
                else:
                    _, (lh, hl, hh) = fwt2(img, wave='haar', level=1)
                x = torch.cat((lh, hl, hh), dim=1).to(self.device, dtype=torch.float32)
                img = img.to(self.device, dtype=torch.float32)
                n = img.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                e = torch.randn_like(img)
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                # loss = noise_estimation_loss(self.model, x, t, e, b)
                loss = noise_estimation_loss(self.model, img, t, e, b)

                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    psnr = self.sample_validation_patches(val_loader, self.step)
                    writer.add_scalar('PSNR(dB)', psnr, self.step / self.config.training.validation_freq)
                    writer.add_scalar('Loss', loss.item(), self.step / self.config.training.validation_freq)

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join('ckpts', 'patch_2', self.config.data.exp_name + f'_ddpm_epoch_{str(epoch)}'))

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = self.config.data.save_dir
        psnr = []
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y, _) in enumerate(val_loader):
                if i == 2:
                    break
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                y = y.flatten(start_dim=0, end_dim=1) if y.ndim == 5 else y
                if self.config.data.small:
                    ll, (lh, hl, hh), hf_coeffs = fwt2(x, wave='haar', level=2)
                else:
                    ll, (lh, hl, hh) = fwt2(x, wave='haar', level=1)
                # x_cond = torch.cat((lh, hl, hh), dim=1).to(self.device, dtype=torch.float32)
                x_cond = x.to(self.device, dtype=torch.float32)    
                n = x.size(0)
                x_rec = torch.randn(n, 1, self.config.data.image_size, self.config.data.image_size, device=self.device)
                x_rec = self.sample_image(x_cond, x_rec).to(x.device, dtype=x.dtype)
                # l_coeffs = torch.tensor_split(x_rec.to(ll.device, dtype=ll.dtype), 3, dim = 1)
                # x_rec = iwt2([ll, l_coeffs, hf_coeffs], wave='haar') if self.config.data.small else iwt2([ll, l_coeffs], wave='haar')
                psnr.append(10 * torch.log10(1 / (F.mse_loss(y, x_rec))))
                grid = make_grid(torch.cat([x, x_rec, y], dim=0), nrow=3)
                save_image(grid, os.path.join(image_folder, f"step{str(step)}_{i}.png"))
            return sum(psnr) / len(psnr)