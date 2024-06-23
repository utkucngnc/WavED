from typing import Union, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM(nn.Module):

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, schedule: str = "linear") -> None:
        '''
        PyTorch Module for Denoising Diffusion Probabilistic Models.

        :param eps_model: The model to be used for the diffusion process.
        :param n_steps: The number of steps for the diffusion process.
        :param device: The device to be used for the model.
        :param schedule: The schedule to be used for the beta array.
        '''
        super().__init__()
        self.model = eps_model
        self.n_steps = n_steps
        
        # Create the beta schedule
        self.beta = torch.from_numpy(get_named_beta_schedule(schedule_name = schedule, num_diffusion_timesteps = n_steps)).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)
        
        self.sigma_square = self.beta
    
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Compute the distribution x_t given x_0.

        :param x0: The initial image.
        :param t: The time step.
        :return: Tuple[torch.Tensor, torch.Tensor]: The mean and variance of the distribution.
        '''
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Sample from the distribution of x_t given x_0.

        :param x0: The initial image.
        :param t: The time step.
        :param eps: The noise to be added.
        :return: torch.Tensor: The sampled image.
        '''
        mean, var = self.q_xt_x0(x0, t)
        if eps is None:
            eps = torch.randn_like(x0)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        '''
        Sample from the distribution of x_{t-1} given x_t.
        :param xt: The image at time t.
        :param t: The time step.
        :return: torch.Tensor: The sampled image.
        '''
        eps_theta = self.model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coeff = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_theta * eps_coeff)
        var = gather(self.sigma_square, t)
        
        eps = torch.randn_like(xt).to(xt.device)
        return mean + (var ** 0.5) * eps
    
    def loss(self, x0: torch.Tensor, eps: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Compute the loss for the model.

        :param x0: The initial image.
        :param t: The time step.
        :return: torch.Tensor: The loss.
        '''
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        if eps is None:
            eps = torch.randn_like(x0)
        
        xt = self.q_sample(x0, t, eps = eps)
        eps_theta = self.model(xt.to(torch.float32), t)
        return eps, eps_theta

####################################################################################################
# The following code is copied from 
# Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672)
####################################################################################################
    
def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    

def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta = 0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)