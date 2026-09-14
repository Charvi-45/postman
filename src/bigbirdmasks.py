

import torch
import random
device = torch.device("cuda")
def bigbird(q, k, v, w, r, g, device=None):

    if device is None:
        device = q.device

    seq_len = q.shape[-2]
    nb = seq_len // w

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    step = max(1, nb // g)
    g_blocks = [i * step for i in range(g)]

    for i in range(nb):

        poss = list(range(nb))

        # local blocks
        local_blocks = list(range(max(0, i-1), min(nb, i+2)))

        if i in poss:
            poss.remove(i)

        random.shuffle(poss)
        rand_blocks = poss[:r]

        allowed_blocks = set(local_blocks + g_blocks + rand_blocks)

        q_start = i * w
        q_end = min((i + 1) * w, seq_len)

        for j in allowed_blocks:

            k_start = j * w
            k_end = min((j + 1) * w, seq_len)

            mask[q_start:q_end, k_start:k_end] = True

    # causal mask
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )

    mask = mask & causal

    return mask

import shutil
shutil.move("/content/src/bigbirdmasks.py", "/content/postman/")

