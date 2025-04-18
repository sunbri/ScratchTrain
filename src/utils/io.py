import os
import pathlib
import torch

CHECKPOINTS_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "checkpoints"

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    iteration: int, out: str | os.PathLike):
    obj = {}
    obj['model'] = model.state_dict()
    obj['optimizer'] = optimizer.state_dict()
    obj['iteration'] = iteration

    torch.save(obj, CHECKPOINTS_PATH / out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    obj = torch.load(CHECKPOINTS_PATH / src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']