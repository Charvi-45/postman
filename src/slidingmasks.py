import torch

def sliding_win(seq_len: int, w: int, device):
    mask = torch.zeros(
        seq_len,
        seq_len,
        dtype=torch.bool,
        device=device
    )

    for i in range(seq_len):
        start = max(0, i - w + 1)
        mask[i, start:i + 1] = True

    return mask
