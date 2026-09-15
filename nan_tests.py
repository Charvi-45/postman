
import torch

from src.attention import dense
from src.masks import mask
from src.slidingmasks import sliding_win
from src.bigbirdmasks import bigbird
def check_empty_attention():
    torch.manual_seed(0)

    n = 8
    q = torch.randn(1, 1, n, 8)
    k = torch.randn(1, 1, n, 8)
    v = torch.randn(1, 1, n, 8)

    # No token allowed for first query
    attn_mask = sliding_win(n, 0, "cuda")

    result = dense(q, k, v, attn_mask)

    # completely masked query should not produce NaN
    assert torch.isnan(result).any().item() == False

    first = result[..., 0, :]
    assert torch.all(first == 0)

    print("Fully masked row test passed")


def check_regular_attention():
    torch.manual_seed(0)

    n = 8
    q = torch.randn(1, 1, n, 8)
    k = torch.randn(1, 1, n, 8)
    v = torch.randn(1, 1, n, 8)

    normal_mask = mask(q)
    normal_result = dense(q,k,v,normal_mask)
    modified_mask = normal_mask.clone()
    modified_mask[0] = False

    modified_result = dense(q, k, v, modified_mask)

    assert torch.allclose(
        modified_result[..., 1:, :],
        normal_result[..., 1:, :],
        atol=1e-6
    )

    assert torch.all(
        modified_result[..., 0, :] == 0
    )

    print("Normal row test passed")


check_empty_attention()
check_regular_attention()
