import math
from ptwt import wavedec2, waverec2
from typing import List
import torch
from torchvision import transforms

def fwt2(x, wave='haar', level=1):
    if not isinstance(x, torch.Tensor):
        x = transforms.ToTensor()(x)
    coeffs = wavedec2(x, wavelet=wave, level=level)
    return coeffs # return LL, (LH, HL, HH)

def iwt2(coeffs, wave='haar'):
    return waverec2(coeffs, wavelet=wave)

def zip_coeffs(coeffs_list: List[torch.Tensor | tuple]):
    coeffs_tensor = coeffs_list[0]
    b,_,h,w = coeffs_list[0].shape
    
    for i in range(1, len(coeffs_list)):
        for _,j in enumerate(coeffs_list[i]):
            temp = j.view(b, -1, h, w) if i != 1 else j
            coeffs_tensor = torch.cat((coeffs_tensor, temp), dim=1)
    
    return coeffs_tensor # return b, c, h, w

def extract_coeffs(coeffs_tensor: torch.Tensor):
    coeffs_list = [coeffs_tensor[:,:1]]
    b,c,h,w = coeffs_tensor.shape
    level = math.log2(c) // 2
    current_idx = 1
    
    for i in range(int(level)):
        window = 4 ** i
        hf_comp = ()
        if window != 1:
            h,w = h*2, w*2
        for _ in range(3):
            hf_comp += (coeffs_tensor[:,current_idx:current_idx+window, :, :].view(b, -1, h, w),)
            current_idx += window
        coeffs_list.append(hf_comp)
    
    return coeffs_list # return LL, (LH, HL, HH)