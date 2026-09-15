# Sparse Attention from Scratch

This project is about implementing and experimenting with sparse attention from scratch using PyTorch.

The main aim is to understand how attention works, how different sparse patterns work, and how they compare with normal dense attention.

## What I implemented

- Manual dense causal attention
- Sliding-window attention
- BigBird-style block sparse attention
- Correctness tests
- Handling of fully masked rows to avoid NaNs
- Runtime and memory benchmarking
- Small character-level GPT experiment on TinyShakespeare

## Files

src/
├── attention.py
├── masks.py
├── slidingmasks.py
├── bigbirdmasks.py
└── gpt.py

correctness.py
benchmark.py
README.md

## Dense Attention

The dense attention implementation does the basic attention steps manually:

QKᵀ → scaling → masking → softmax → multiply by V

I used this as the reference implementation for the sparse versions.

## Sparse Attention

### Sliding Window

In sliding-window attention, each token only attends to a fixed number of nearby tokens.

This reduces the number of attention connections, but it also means that a token cannot directly see information that is far away.

### BigBird

The BigBird-style pattern uses a combination of local, global, and random blocks.

The idea is to keep the number of connections smaller than dense attention while still allowing information to move across different parts of the sequence.

## Correctness

I compare the sparse attention outputs with the dense reference to make sure the implementations give the same results where the attention pattern allows them.

Run:

python correctness.py

## NaN Handling

A problem occurs when an entire row is masked.

Softmax on:

-inf, -inf, -inf, ...

results in NaN.

I handle this case by setting the attention weights for completely masked rows to zero.

## Benchmark

I benchmark dense, sliding-window, and BigBird attention for different sequence lengths:

512
1024
2048
4096
8192

The benchmark records forward-pass time and peak GPU memory.

Run:

python benchmark.py

## GPT Experiment

I also train a small 2-layer character-level GPT on TinyShakespeare using the three attention patterns.

The model uses:

- 2 transformer layers
- 4 attention heads
- embedding size 128
- context length 128
- batch size 32

Run:

python src/gpt.py

The training and validation losses are plotted to compare the three attention methods.

## Main Idea

Dense attention can look at every previous token, but its computation and memory grow quickly as the sequence gets longer.

Sparse attention reduces the number of connections, which can make long sequences cheaper to process. The trade-off is that restricting attention can cause the model to lose some long-range information.

The experiments in this project are meant to see this trade-off in practice.
