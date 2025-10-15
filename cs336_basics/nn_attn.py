import torch
import numpy as np
from torch import nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
from cs336_basics.nn_ops import softmax
from cs336_basics.nn_rope import Rope
from cs336_basics.nn_linear import Linear
from torch import Tensor
from typing import Optional


def scaled_dot_product_attn(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None):

    dot_prod = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / np.sqrt(Q.shape[-1])
    dot_prod = dot_prod.masked_fill(~mask, float("-inf"))
    dot_prod = softmax(dot_prod, dim=-1)
    return einsum(dot_prod, V, "... queries keys, ... keys d_v -> ... queries d_v")

class MultiHeadSelfAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(MultiHeadSelfAttn, self).__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        assert d_model % num_heads == 0
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if use_rope:
            self.theta = theta
            self.rope = Rope(d_model // num_heads, theta, max_seq_len, dtype=dtype, device=device)


    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_in"]
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_length = x.shape[-2]
        q = einsum(x, self.q_proj.weight, "... sequence_length d_in, d_model d_in -> ... sequence_length d_model")
        k = einsum(x, self.k_proj.weight, "... sequence_length d_in, d_model d_in -> ... sequence_length d_model")
        v = einsum(x, self.v_proj.weight, "... sequence_length d_in, d_model d_in -> ... sequence_length d_model")

        q = rearrange(q, "... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.num_heads)
        k = rearrange(k, "... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.num_heads)
        v = rearrange(v, "... sequence_length (h d_v) -> ... h sequence_length d_v", h=self.num_heads)
        mask = (1 - torch.triu(torch.ones((seq_length, seq_length), device=self.device), diagonal=1)).to(torch.bool)
        if self.use_rope:
            q = self.rope(q, torch.arange(seq_length))
            k = self.rope(k, torch.arange(seq_length))        
        result = scaled_dot_product_attn(q, k, v, mask)
        result = rearrange(result, "... h sequence_length d_v -> ... sequence_length (h d_v)")

        return einsum(self.output_proj.weight, result, "d_model d_out, ... sequence_length d_out -> ... sequence_length d_model")