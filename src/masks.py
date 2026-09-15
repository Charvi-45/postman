import torch
def mask(q):

    seq_len = q.shape[-2]

    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=q.device
        )
    )

    return mask
