import torch
import gc
import json
import math
import statistics
import matplotlib.pyplot as plt
from src.attention import dense
from src.masks import mask
from src.slidingmasks import sliding_win
from src.bigbirdmasks import bigbird
from torch.utils.benchmark import Timer
torch.manual_seed(42)
sl=[512, 1024, 2048, 4096, 8192]
shape = (1, 1, , HEAD_DIM)
timer = Timer(
    stmt="(q, k, v)",
    globals={"q": q, "k": k, "v": v}
)

result = timer.blocked_autorange(min_run_time=1)#performs repeated measurements and reports statistics rather than relying on one execution
print(result)
