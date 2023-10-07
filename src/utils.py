import torch


def insert_cellular_padding(matrix: torch.Tensor, size: int = 1):
    """
    Inputs are of shape (batch, 1, W, H)
    """
    pad_left = matrix[:, :, :, -size:]
    pad_right = matrix[:, :, :, :size]

    matrix_padded_middle = torch.concat((pad_left, matrix, pad_right), dim=3)

    pad_up = matrix_padded_middle[:, :, -size:]
    pad_down = matrix_padded_middle[:, :, :size]

    result = torch.concat((pad_up, matrix_padded_middle, pad_down), dim=2)
    return result
