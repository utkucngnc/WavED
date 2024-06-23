import os
from tqdm import tqdm
import torch
from torchvision.utils import save_image, make_grid

from src import fwt2, iwt2, HFRM

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

class DiffusiveRestoration:
    def __init__(self, diffusion_small, diffusion_large, args, config_small, config_large):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config_small = config_small
        self.config_large = config_large
        self.diffusion_small = diffusion_small
        self.diffusion_large = diffusion_large

        if os.path.isfile(args.resume_small):
            self.diffusion_small.load_ddm_ckpt(args.resume_small, ema=True)
            self.diffusion_small.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')
        
        if os.path.isfile(args.resume_large):
            self.diffusion_large.load_ddm_ckpt(args.resume_large, ema=True)
            self.diffusion_large.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, r=None):
        image_folder = self.config_small.data.save_dir
        pbar = tqdm(val_loader)
        with torch.no_grad():
            for i, (x, y) in enumerate(pbar):
                x_refined = refine(torch.cat((x, x, x), dim=1)).to(x.device)
                lll, (llh, lhl, lhh), (lh, hl, hh) = fwt2(x_refined, level=2)
                x_cond_small = torch.cat([llh, lhl, lhh], dim=1).to(self.diffusion_small.device)
                x_cond_large = torch.cat([lh, hl, hh], dim=1).to(self.diffusion_large.device)
                x_out_small = self.diffusive_restoration(x_cond_small, r=r, use_small=True)
                x_out_large = self.diffusive_restoration(x_cond_large, r=r)
                low_coeffs = torch.tensor_split(x_out_small.to(lll.device, dtype = lll.dtype), 3, dim=1)
                high_coeffs = torch.tensor_split(x_out_large.to(lll.device, dtype = lll.dtype), 3, dim=1)
                x_rec = iwt2([lll, low_coeffs, high_coeffs]).to(x.device)
                grid = make_grid(torch.cat((x, x_rec, y),dim=0), nrow=3)
                psnr = -10 * torch.log10(torch.mean((x_rec - y) ** 2))
                pbar.set_description(f'PSNR: {psnr:.4f} dB')
                save_image(grid, os.path.join(image_folder, f"sample_pre_refined_{i}.png"))
                if i ==5:
                    break

    def diffusive_restoration(self, x_cond, r=None, use_small=False):
        if use_small:
            p_size = self.config_small.data.image_size
            h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
            corners = [(i, j) for i in h_list for j in w_list]
            x = torch.randn(x_cond.size(), device=self.diffusion_small.device)
            x_output = self.diffusion_small.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
            return x_output
        p_size = self.config_large.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion_large.device)
        x_output = self.diffusion_large.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list