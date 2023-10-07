import torch
from torch.nn import Module

from src.utils import insert_cellular_padding


class CaModule(Module):
    """
    Inputs are of shape (batch, 1, W, H)
    """
    def __init__(self, n_size: int = 1, filters: int = 100, conv1x1_number: int = 11):
        super().__init__()
        self.n_size = n_size
        self.conv = torch.nn.Conv2d(1, filters, kernel_size=(2*n_size+1, 2*n_size+1))
        self.layers = [torch.nn.Conv2d(filters, filters, kernel_size=(1, 1)) for _ in range(conv1x1_number)]
        self.last = torch.nn.Conv2d(filters, 1, kernel_size=(1, 1))

    def forward(self, x):
        x = insert_cellular_padding(x)
        x = self.conv(x)
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.last(x)
        x = torch.sigmoid(x)
        return x
