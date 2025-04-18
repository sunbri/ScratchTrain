import numpy as np
import numpy.typing as npt
import torch

def get_batch(x: npt.NDArray, batch_size: int, context_length: int, device: str = None):
    effective_size = len(x) - context_length
    indices = torch.randint(0, effective_size, (batch_size,))

    a = torch.stack([torch.from_numpy(x[i:i+context_length]).long() for i in indices])
    b = torch.stack([torch.from_numpy(x[i+1:i+1+context_length]).long() for i in indices])

    # pin memory if CUDA
    if device == 'cuda':
        a = a.pin_memory().to(device, non_blocking=True)
        b = b.pin_memory().to(device, non_blocking=True)
    else:
        a, b = a.to(device), b.to(device)

    return (a, b)