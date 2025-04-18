from collections.abc import Callable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        # we are only doing one parameter group here
        # where a group is a dict with parameters to optimize and hyperparameters
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        # the user might pass in a callable closure to recompute the 
        # loss before the optimizer step
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0) # get iteration number or initialize to zero
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss

# here, based on the magnitude of things, lr=1e2 works best
# lower order of magnitude has slower convergence and higher
# order of magnitude overshoots and sends to infinity

# weights = torch.nn.Parameter(5 * torch.randn(10, 10))
# opt = SGD([weights], lr=1e2)

# for t in range(100):
#     opt.zero_grad()
#     loss = (weights**2).mean()
#     print(loss.cpu().item())
#     loss.backward()
#     opt.step()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "beta_1": betas[0],
                    "beta_2": betas[1], "epsilon": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = beta_1 * state.get("m", 0) + (1 - beta_1) * grad
                v = beta_2 * state.get("v", 0) + (1 - beta_2) * grad * grad
                lr_t = lr * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + epsilon)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

def learning_rate_schedule(t, max_learning_rate, min_learning_rate, warmup_iters, 
                           cosine_cycle_iters):
    if t < warmup_iters:
        return t / warmup_iters * max_learning_rate
    elif warmup_iters <= t and t <= cosine_cycle_iters:
        return min_learning_rate + 1/2 * \
            (1 + math.cos((t-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi)) * \
            (max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters, max_norm):
    epsilon = 10e-6
    norm_s = 0
    for param in parameters:
        if param.grad is not None:
            norm_s += (param.grad.data ** 2).sum()
    norm = math.sqrt(norm_s)
    if norm >= max_norm:
        scale = max_norm / (norm + epsilon)
        for param in parameters:
            if param.grad is not None:
                param.grad.data *= scale