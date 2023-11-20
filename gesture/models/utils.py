import torch
from torch.utils.data import Dataset, DataLoader

def squeeze_all(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    return torch.squeeze(x)
