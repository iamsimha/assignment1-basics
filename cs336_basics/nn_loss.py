import torch
import numpy as np
from torch import nn
from einops import rearrange, reduce
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor

def cross_entropy(logits: Float[Tensor, "batch_size num_classes"],
                    targets: Int[Tensor, "batch_size"]):
    num_classes = logits.shape[-1]
    logits = rearrange(logits, "... num_classes -> (...) num_classes")

    _max, _ = torch.max(logits, dim=-1, keepdims=True)
    logits = logits - _max
    o_i = torch.gather(logits, 1, targets.unsqueeze(1))
    exp_logits = logits.exp()
    log_sum_exp = reduce(exp_logits, "... num_classes -> (...)", "sum").log()
    return (log_sum_exp - o_i).mean()