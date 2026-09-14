
import torch
def mask(q):
  seq_len = q.shape[-2]
  #causal mask

  mask=torch.tril(torch.ones(seq_len, seq_len))
  return mask
