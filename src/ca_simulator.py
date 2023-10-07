import torch

from src.utils import insert_cellular_padding


class Simulator:
    """
    Inputs are of shape (batch, 1, W, H)
    """
    def __init__(self, born_interval: (int, int), live_interval: (int, int), neighbourhood_size: int):
        self.born_interval = born_interval
        self.live_interval = live_interval

        self.n_size = neighbourhood_size
        self.conv_weight = torch.ones(1, 1, self.n_size*2+1, self.n_size*2+1, requires_grad=False)
        self.conv_weight[0, 0, self.n_size, self.n_size] = 0

    def get_next_state(self, prev_state: torch.Tensor):
        prev_state_padded = insert_cellular_padding(prev_state, self.n_size)
        n_values = torch.conv2d(prev_state_padded, self.conv_weight)

        new_state = torch.zeros_like(prev_state)

        born_mask = (self.born_interval[0] < n_values) & (
                n_values < self.born_interval[1]) & (prev_state == 0.)
        live_mask = (self.live_interval[0] < n_values) & (
                    n_values < self.live_interval[1]) & (prev_state == 1.)
        new_state[born_mask.view(new_state.shape)] = 1.
        new_state[live_mask.view(new_state.shape)] = 1.
        return new_state
