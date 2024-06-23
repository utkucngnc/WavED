import torch as th
from matplotlib import pyplot as plt
from logging import getLogger

def show_grid(grid: th.Tensor, title: str = '') -> None:
    '''
    Show the grid of images.
    :param grid: The grid of image /-s.
    :param title: The title of the plot.
    '''
    if grid.ndim == 4:
        temp = grid.squeeze(0).permute(1,2,0).detach().cpu()
    elif grid.ndim == 3:
        temp = grid.permute(1,2,0).detach().cpu()
    plt.imshow(temp)
    plt.title(title)
    plt.show()