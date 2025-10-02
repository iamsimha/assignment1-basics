import torch
import numpy as np
from torch import nn
from einops import einsum, reduce
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.in_dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        x = x.to(torch.float32)
        norm = torch.sqrt((1 / (self.d_model) * reduce(x**2, "... d_model -> ...", "sum")) + self.eps)
        return (x / norm.unsqueeze(-1) * self.weight).to(self.in_dtype)