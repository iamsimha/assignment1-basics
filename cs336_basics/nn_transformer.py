import torch
import numpy as np
from torch import nn
from einops import einsum
from cs336_basics.nn_attn import MultiHeadSelfAttn
from cs336_basics.nn_swiglu import SwiGlu
from cs336_basics.nn_rms_norm import RMSNorm
from cs336_basics.nn_embedding import Embedding
from cs336_basics.nn_linear import Linear
from cs336_basics.nn_ops import softmax
from typing import Optional
from torch import Tensor
from jaxtyping import Float, Int

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttn(d_model, num_heads,
                                use_rope=True, theta=theta,
                                max_seq_len=max_seq_len,
                                device=device, dtype=dtype)
        self.ffn = SwiGlu(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
    
    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
    ) -> Float[Tensor, "... seq_len d_model"]:
        first_sub_layer = self.attn(self.ln1(x))
        x = x + first_sub_layer
        second_sub_layer = self.ffn(self.ln2(x))
        x = x + second_sub_layer
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        vocab_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(Transformer, self).__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff,
                        max_seq_len, theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        input_indices: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        out = self.token_embeddings(input_indices)
        for layer in self.layers:
            out = layer(out)
        out = self.lm_head(self.ln_final(out))
        return out
