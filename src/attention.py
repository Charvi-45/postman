
import math
import torch

def dense(q, k, v, mask=None):
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if mask is not None:
        mask = mask.bool()
        valid = mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
    else:
        weights = torch.softmax(scores, dim=-1)

    return torch.matmul(weights, v)
def sparse(q, k, v, mask):

    mask = mask.bool()

    seq_len = q.shape[-2]
    d = q.shape[-1]

    scale = 1.0 / math.sqrt(d)

    outputs = []

    for i in range(seq_len):
        allowed = mask[i]

        q_i = q[..., i, :]

        k_i = k[..., allowed, :]
        v_i = v[..., allowed, :]

        scores_i = torch.matmul(
            q_i.unsqueeze(-2),
            k_i.transpose(-2, -1)
        ) * scale

        weights_i = torch.softmax(scores_i, dim=-1)

        output_i = torch.matmul(weights_i, v_i)

        outputs.append(output_i)

    return torch.cat(outputs, dim=-2)
