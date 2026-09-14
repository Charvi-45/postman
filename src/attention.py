import math

import torch
device="cuda"
def dense(
    
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor| None = None,
) -> torch.Tensor:


   d = q.shape[-1]
   scale = 1.0 / math.sqrt(d)
   scores = torch.matmul(q, k.transpose(-2, -1)) * scale

   scores = scores.masked_fill(~mask, float("-inf"))
   weights = torch.softmax(scores, dim=-1)

   return torch.matmul(weights, v)

