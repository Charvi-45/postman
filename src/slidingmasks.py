
import torch
device = torch.device("cpu")
def sliding_win(seq_len:int,w:int,device=device):

  mask = torch.zeros(seq_len,seq_len)
  for i in range(seq_len):
    for j in range(seq_len):
        if max(0, i-w+1) <= j <= i:
            mask[i, j] = 1
  %%writefile src/masks.py
  return mask
