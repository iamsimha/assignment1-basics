import torch
import numpy as np
from torch import nn
from einops import einsum
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor

class Rope(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(Rope, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.initialize_sin_cos()

        
    def initialize_sin_cos(self) -> None:
        theta = self.theta
        max_seq_len = self.max_seq_len
        d_k = self.d_k
        assert d_k % 2 == 0
        den = theta ** (torch.arange(0, d_k, 2) / d_k)
        num = torch.arange(0, max_seq_len)
        freqs = num.unsqueeze(1) / den
        self.register_buffer("cos_matrix", torch.cos(freqs))
        self.register_buffer("sin_matrix", torch.sin(freqs))

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:

        d_k = self.d_k
        result = torch.zeros_like(x)
        c, s = self.cos_matrix, self.sin_matrix
        result[..., 0:d_k:2] = x[..., token_positions, 0:d_k:2] * c[token_positions, :] + x[..., token_positions, 1:d_k:2] * -s[token_positions, :]
        result[..., 1:d_k:2] = x[..., token_positions, 0:d_k:2] * s[token_positions, :] + x[..., token_positions, 1:d_k:2] * c[token_positions, :]
        return result
        