import torch
import gc
import json
import math
import statistics
import matplotlib.pyplot as plt
from attention import dense
from masks import mask
from slidingmasks import sliding_win
from bigbirdmasks import bigbird
from torch.utils.benchmark import Timer
CONFIGS = ["dense", "sliding", "bigbird"]
device = "cuda"
sl=[512, 1024, 2048, 4096, 8192]
def make_qkv(seq_len):
    shape = (1, 1, seq_len, 64)  
    q = torch.randn(shape, device=device)
    k = torch.randn(shape, device=device)
    v = torch.randn(shape, device=device)
    return q, k, v

def build_config(name, seq_len):
    torch.manual_seed(42+seq_len)
    q, k, v = make_qkv(seq_len)
    
    if name == "dense":
        m = mask(q).to(device)
        run_once = lambda: dense(q, k, v, mask=m)

    elif name == "sliding":
        m = sliding_win(seq_len, 4, device).to(device)
        run_once = lambda: sparse(q, k, v, mask=m)

    elif name == "bigbird":
        m = bigbird(q, k, v, 4, 2, 2, device).to(device)
        run_once = lambda: sparse(q, k, v, mask=m)

    else:
        raise ValueError(name)

    return run_once, [q, k, v, m]
@torch.no_grad()
def benchmark_fwd(name, sl, n_warmup=3, n_iters=10):
    # Running forward pass only (no gradients)
    # sparse() is slower here because it loops per token in Python,
    # so keeping iterations low. Very large seq_len might still be slow.

    run_once, tensors = build_config(name, sl)

    # Warmup runs (GPU needs a few runs before timing becomes stable)
    for _ in range(n_warmup):
        run_once()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)#reset mem

    fwd_ms = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        run_once()
        end.record()

        torch.cuda.synchronize()  # wait for GPU to finish
        fwd_ms.append(start.elapsed_time(end))  # time in ms

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del run_once, tensors
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "seq_len": sl,
        "fwd_ms": statistics.median(fwd_ms),
        "peak_mem_mb": peak_mem_mb
    }


print(f"{'config':<10} {'seq_len':>8} {'median fwd ms':>15} {'peak mem MB':>13}")

results = []

for seq_len in sl:
    for cfg in CONFIGS:
        r = benchmark_fwd(cfg, seq_len)
        results.append(r)

        print(f"{r['name']:<10} {r['seq_len']:>8} {r['fwd_ms']:>15.3f} {r['peak_mem_mb']:>13.1f}")
 
with open("benchmark_results.json", "w") as f:# Save results 
    json.dump(results, f, indent=2)


# Plotting time and memory vs sequence length
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for cfg in CONFIGS:
    xs = [r["seq_len"] for r in results if r["name"] == cfg]

    axes[0].plot(xs,
                 [r["fwd_ms"] for r in results if r["name"] == cfg],
                 marker="o", label=cfg)

    axes[1].plot(xs,
                 [r["peak_mem_mb"] for r in results if r["name"] == cfg],
                 marker="o", label=cfg)

axes[0].set_xlabel("Sequence length")
axes[0].set_ylabel("Median forward time (ms)")
axes[0].set_title("Forward pass time")
axes[0].set_xscale("log", base=2)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Sequence length")
axes[1].set_ylabel("Peak memory (MB)")
axes[1].set_title("Forward pass memory")
axes[1].set_xscale("log", base=2)
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("benchmark_dense_vs_sparse.png", dpi=150)

print("Saved plot to benchmark_dense_vs_sparse.png")
