import torch
import numpy as np
from torch import nn
from einops import einsum
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor


def softmax(
    x: Float[Tensor, "..."],
    dim: int,
) -> Float[Tensor, "..."]:
    values, _ = torch.max(x, dim=dim, keepdims=True)
    x = x - values
    num = torch.exp(x)
    den = torch.sum(num, dim=dim, keepdims=True)
    return num/den

def silu(
    x: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    return x / (1 + torch.exp(-x))
