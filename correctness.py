from src.masks import causal_mask
from src.slidingmasks import  sliding_win
from src.bigbirdmasks import  bigbird
from src.attention  import dense

def assert_matches_dense_where_masks_agree(dense_mask, sparse_mask, q, k, v):
    dense_out = dense_attention(q, k, v, dense_mask)
    sparse_out = dense_attention(q, k, v, sparse_mask)

    checked = 0

    for i in range(dense_mask.shape[0]):
        if torch.equal(dense_mask[i], sparse_mask[i]):
            assert torch.allclose(
                dense_out[..., i, :],  #output for token i, keeping all the earlier dimensions.
                sparse_out[..., i, :],
                atol=1e-6
            )
            checked += 1

    assert checked > 0
q = torch.randn(1, 1, 8, 4)
k = torch.randn(1, 1, 8, 4)
v = torch.randn(1, 1, 8, 4)

mask = mask(8)
sparse_mask=sliding_win(8,4,"cuda")

print(out.shape)

dense_out = dense_attention(
        q, k, v, mask
    )

sparse_out = dense_attention(
        q, k, v, sparse_mask
    )
