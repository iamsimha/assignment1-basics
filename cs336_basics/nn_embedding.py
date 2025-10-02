import torch
import numpy as np
from torch import nn
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(Embedding, self).__init__()
        std = np.sqrt(2.0 / (num_embeddings + embedding_dim))
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim),
                                        device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(
        self,
        token_ids: Int[Tensor, "..."],   # token indices of arbitrary shape
    ) -> Float[Tensor, "... embedding_dim"]:

        return self.weight[token_ids, :]
