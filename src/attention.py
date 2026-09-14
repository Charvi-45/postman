
import math
import torch

def dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:

    # 1. Dimension of each key/query vector
    d = q.shape[-1]

    # 2. Scaling factor
    scale = 1.0 / math.sqrt(d)

    # 3. Calculate attention scores
    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    ) * scale

    # 4. Apply mask
    if mask is not None:

        mask = mask.bool()

        # Check which rows have at least one allowed position
        valid = mask.any(dim=-1, keepdim=True)

        # Disallowed positions become -infinity
        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # 5. Softmax
        weights = torch.softmax(
            scores,
            dim=-1
        )

        # 6. Fix fully-masked rows
        weights = torch.where(
            valid,
            weights,
            torch.zeros_like(weights)
        )

    else:

        # No mask → normal softmax
        weights = torch.softmax(
            scores,
            dim=-1
        )

    # 7. Weighted sum of values
    return torch.matmul(weights, v)
