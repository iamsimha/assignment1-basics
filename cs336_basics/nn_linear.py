import torch
import numpy as np
from torch import nn
from einops import einsum
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(Linear, self).__init__()
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(
        self,
        x: Float[Tensor, "... d_in"],
    ) -> Float[Tensor, "... d_out"]:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
