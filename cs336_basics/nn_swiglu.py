import torch
import numpy as np
from torch import nn
from einops import einsum
from cs336_basics.nn_ops import silu
from cs336_basics.nn_linear import Linear
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor

class SwiGlu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(SwiGlu, self).__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear( d_ff, d_model, device=device, dtype=dtype)
        
        std = 2.0 / (d_ff + d_model)

        for w in [self.w1.weight, self.w2.weight, self.w3.weight]:
            nn.init.trunc_normal_(w, mean=0, std=std, a=-3*std, b=3*std)

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:

        lft = silu(einsum(self.w1.weight, x, "d_ff d_model, ... d_model -> ... d_ff"))
        
        rht = einsum(self.w3.weight, x, "d_ff d_model, ... d_model -> ... d_ff")

        return einsum(self.w2.weight, (lft * rht), "d_model d_ff, ... d_ff -> ... d_model")