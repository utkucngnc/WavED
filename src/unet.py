import math
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn

'''
Modules for the U-Net model.
Ported from labML's implementation:
nn.labml.ai/diffusion/ddpm/unet.html
'''

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)
    
    def forward(self, t: torch.Tensor):
        '''
        Create sinusoidal position embeddings for the input tensor.
        Inputs:
            t: torch.Tensor, shape (n_channels,)
        Returns:
            emb: torch.Tensor, shape (n_channels,)
        '''
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        
        return emb

class ResidualBlock(nn.Module):
    '''
    Initialize the residual block.
    Inputs:
        in_channels: int, the number of input channels
        out_channels: int, the number of output channels
        time_channels: int, the number of time channels
        n_groups: int, the number of groups for group normalization
        dropout: float, the dropout rate
    '''
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        
        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        else:
            self.shortcut = nn.Identity()
        
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        Run a forward pass through the residual block.
        Inputs:
            x: torch.Tensor, shape (batch_size, in_channels, height, width)
            t: torch.Tensor, shape (batch_size, time_channels)
        Returns:
            x: torch.Tensor, shape (batch_size, out_channels, height, width)
        '''
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        
        # Add the time embedding
        h = h + self.time_emb(self.time_act(t))[:, None, None, :]
        
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    '''
    Initialize the attention block.
    Inputs:
        n_channels: int, the number of input channels
        n_heads: int, the number of attention heads
        d_k: int, the number of dimensions for the queries, keys, and values
        n_groups: int, the number of groups for group normalization
    '''
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()
        # Initialize d_k if it is not provided
        if d_k is None:
            d_k = n_channels
        
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Linear layer for the queries, keys, and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for the output
        self.out = nn.Linear(n_heads * d_k, n_channels)
        
        # Scale for the dot product
        self.scale = d_k ** -0.5
        
        self.n_heads = n_heads
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        '''
        Run a forward pass through the attention block.
        Inputs:
            x: torch.Tensor, shape (batch_size, in_channels, height, width)
            t: torch.Tensor, shape (batch_size, time_channels)
        Returns:
            x: torch.Tensor, shape (batch_size, in_channels, height, width)
        '''
        _ = t
        batch_size, n_channels, height, width = x.shape
        # Change x to the shape (batch_size, seq, n_channels)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        
        # Compute queries, keys, and values       
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Compute the attention scores
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = torch.softmax(attn, dim=2)
                             
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.out(res)
        # Add skip connection
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        
        return res

class DownBlock(nn.Module):
    '''
    This combines ResidualBlock and AttentionBlock . These are used in the first half of U-Net at each resolution.
    '''
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        This function runs a forward pass through the down block.
        Inputs:
            x: torch.Tensor, shape (batch_size, in_channels, height, width)
            t: torch.Tensor, shape (batch_size, time_channels)
        Returns:
            x: torch.Tensor, shape (batch_size, out_channels, height // 2, width // 2)
        '''
        x = self.res(x, t)
        x = self.attn(x, t) if isinstance(self.attn, AttentionBlock) else x
        
        return x

class UpBlock(nn.Module):
    '''
    This combines ResidualBlock and AttentionBlock . These are used in the second half of U-Net at each resolution.
    '''
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool) -> None:
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        This function runs a forward pass through the up block.
        Inputs:
            x: torch.Tensor, shape (batch_size, in_channels, height, width)
            t: torch.Tensor, shape (batch_size, time_channels)
        Returns:
            x: torch.Tensor, shape (batch_size, out_channels, height * 2, width * 2)
        '''
        x = self.res(x, t)
        x = self.attn(x)if isinstance(self.attn, AttentionBlock) else x
        
        return x

class MiddleBlock(nn.Module):
    '''
    It combines a ResidualBlock , AttentionBlock , followed by another ResidualBlock .
    This block is applied at the lowest resolution of the U-Net.
    '''
    def __init__(self, n_channels: int, time_channels: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        This function runs a forward pass through the middle block.
        Inputs:
            x: torch.Tensor, shape (batch_size, n_channels, height, width)
            t: torch.Tensor, shape (batch_size, time_channels)
        Returns:
            x: torch.Tensor, shape (batch_size, n_channels, height, width)
        '''
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        
        return x

class Upsample(nn.Module):
    '''
    Scale up the feature map by a factor of 2.
    '''
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1,1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return self.conv(x)

class Downsample(nn.Module):
    '''
    Scale down the feature map by a factor of 2.
    '''
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return self.conv(x)

class UNet(nn.Module):
    '''
    Initialize the U-Net model.
    Inputs:
        image_channels: int, the number of channels in the input image
        n_channels: int, the number of channels in the first layer
        ch_mults: Tuple[int], the channel multiplier for each resolution
        is_attn: Tuple[bool], whether to use attention at each resolution
        n_blocks: int, the number of residual blocks at each resolution
    '''
    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks: int = 2):
        super().__init__()
        # Number of resolutions
        n_resolutions = len(ch_mults)
        # Project the input image to the first layer
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        # Time embedding
        self.time_emb = TimeEmbedding(n_channels * 4)
        # Down & up blocks
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        # Output channels
        out_channels = in_channels = n_channels
        
        # Create the down blocks
        for i in range(n_resolutions):
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                self.down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Downsample all except the last resolution
            if i < n_resolutions - 1:
                self.down.append(Downsample(in_channels))
        
        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4)
        in_channels = out_channels
        # Up blocks
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                self.up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            self.up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            
            # Upsample all except the last resolution
            if i > 0:
                self.up.append(Upsample(in_channels))
        
        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.conv = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        This function runs a forward pass through the U-Net model.
        Inputs:
            x: torch.Tensor, shape (batch_size, image_channels, height, width)
            t: torch.Tensor, shape (batch_size, time_channels)
        Returns:
            x: torch.Tensor, shape (batch_size, image_channels, height, width)
        '''
        # Time embedding
        t = self.time_emb(t)
        # Get image projection
        x = self.image_proj(x)
        # This will store the skip connections and outputs at each resolution
        h = [x]
        
        # First half of the U-Net
        for block in self.down:
            x = block(x, t)
            h.append(x)
        
        # Middle block
        x = self.middle(x, t)
        
        # Second half of the U-Net
        for block in self.up:
            if isinstance(block, Upsample):
                x = block(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = block(x, t)
        
        return self.conv(self.act(self.norm(x)))