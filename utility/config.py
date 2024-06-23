import torch

class Config:
    
    data_path: str = "F:\\data\\Pristine"
    batch_size: int = 4
    num_workers: int = 4
    
    def naive_model():
        return dict(
            in_channels = 3,
            out_channels = 3,
            num_features = 64,
        )
    
    def model():
        return dict(
            image_channels = 3,
            n_channels = 128,
            ch_mults = [1, 2, 2, 4],
            is_attn = [False, False, False, True],
        )
    
    def model_v2():
        return dict(
            
        )
    
    def min_ddpm():
        return dict(
            betas = (1e-4, 0.02),
            n_T = 1000,
        )
    
    def ddpm():
        return dict(
            n_steps = 1000,
            device = "cuda" if torch.cuda.is_available() else "cpu",
            schedule = "linear",
        )
    
    def train_defaults():
        return dict(
            min = False,
            naive = False,
            epochs = 100,
            learning_rate = 2e-5,
            n_samples = 100,
            batch_size = 32,
            num_workers = 4,
            optimizer = torch.optim.Adam,
        )