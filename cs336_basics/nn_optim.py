from __future__ import annotations

import math
from typing import Optional, Callable
from collections.abc import Iterable

from torch.optim import Optimizer
from typing import Optional
import torch
import math

class SGD(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter] | Iterable[dict],
                lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data = p.data - lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter] | Iterable[dict],
                lr: float,
                betas: tuple[float, float] = (0.9, 0.999),
                eps: float = 1e-8,
                weight_decay: float = 0.01) -> None:
        defaults = {"lr": lr, "beta1": betas[0],
                    "beta2": betas[1], "eps": eps,
                    "weight_decay": weight_decay}
        self.eps = eps
        self.weight_decay = weight_decay
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                t = state.get("t", 1)
                lr_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data = p.data - lr_t * m / (torch.sqrt(v) + self.eps)
                p.data = p.data - lr * self.weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

def get_lr_schedule(it: int,
                    max_learning_rate: float,
                    min_learning_rate: float,
                    warmup_iters: int,
                    cosine_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    if it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi*(it - warmup_iters)/ (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    return min_learning_rate

def l2_norm(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2)))

def clip_gradients(param_list, max_l2_norm):
    grads = [p.grad.flatten() for p in param_list if p.grad is not None]
    total_norm = l2_norm(torch.cat(grads))

    with torch.no_grad():
        if total_norm > max_l2_norm:
            for param in param_list:
                if param.grad is None:
                    continue
                param.grad.mul_(max_l2_norm/ (total_norm  + 1e-6))