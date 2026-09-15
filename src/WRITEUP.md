# Sparse Attention from Scratch

## 1. Introduction

In this project, I implemented attention from scratch using PyTorch and experimented with two sparse attention patterns.

The main goal was to understand how dense attention works and then see what happens when the number of attention connections is reduced.

I implemented:

- Manual dense causal attention
- Sliding-window attention
- BigBird-style block sparse attention
- Correctness tests
- NaN handling for fully masked rows
- Runtime and memory benchmarking
- A small character-level GPT experiment on TinyShakespeare

## 2. Dense Attention

I first implemented dense attention manually instead of using a built-in scaled dot product attention function.

The main steps are:

QK transpose -> scaling -> masking -> softmax -> multiply by V

I used the dense implementation as the reference for checking the sparse versions.

## 3. Sliding-Window Attention

In sliding-window attention, each token only attends to a fixed number of nearby tokens.

This reduces the number of attention connections compared with dense attention.

The disadvantage is that the model cannot directly access information that is far away in the sequence.

## 4. BigBird-Style Attention

The second sparse pattern I implemented was a simplified BigBird-style attention pattern.

It combines:

- Local blocks
- Global blocks
- Random blocks

The local blocks capture nearby information, while the global and random blocks allow some information to move between distant parts of the sequence.

## 5. Correctness Testing

I created a correctness test to compare the sparse attention results with the dense reference.

The same queries, keys and values are used for the comparison.

The test checks that the outputs agree on the positions allowed by the sparse attention pattern.

The test can be run using:

python correctness.py

## 6. NaN Handling

An issue occurs when an entire row of the attention matrix is masked.

For example:

-inf, -inf, -inf, -inf

Applying softmax to this produces NaN values.

To handle this, I check whether a query has at least one valid attention position. If there are no valid positions, its attention weights are set to zero.

This prevents NaNs from propagating through the model.

## 7. Benchmarking

I ran the benchmark on a Tesla T4 GPU using CUDA 12.8.

The sequence lengths tested were:

512
1024
2048
4096
8192

For each sequence length, I measured median forward-pass time and peak GPU memory.

The results were:

| Sequence Length | Dense Time (ms) | Sliding Time (ms) | BigBird Time (ms) | Dense Memory (MB) | Sparse Memory (MB) |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.278 | 117.684 | 156.425 | 13.8 | 9.1 |
| 1024 | 0.540 | 237.160 | 240.604 | 29.9 | 10.6 |
| 2048 | 1.730 | 498.455 | 481.878 | 93.6 | 15.1 |
| 4096 | 6.777 | 952.839 | 951.851 | 347.1 | 30.1 |
| 8192 | 24.095 | 2211.526 | 2066.314 | 1358.1 | 84.1 |

The memory results show a large difference between dense and sparse attention.

At sequence length 8192, dense attention used about 1358 MB of peak GPU memory, while the sparse methods used about 84 MB.

However, dense attention was much faster in this experiment.

At 8192 tokens, dense attention took about 24 ms, while sliding-window attention took about 2212 ms and BigBird took about 2066 ms.

## 8. Why Was Sparse Attention Slower?

At first, this result was surprising because sparse attention uses fewer attention connections.

The main reason is how I implemented sparse attention.

The sparse implementation uses Python loops and processes the allowed positions individually. This creates a lot of overhead.

Dense attention uses large matrix multiplication operations, which are highly optimized on the GPU.

So even though sparse attention does less attention computation, the implementation itself can still be slower.

A more optimized sparse attention implementation could give different runtime results.

## 9. Theoretical Scaling

Dense attention creates an attention matrix of size S x S, where S is the sequence length.

Therefore, the number of attention connections grows approximately as:

O(S^2)

For a fixed local window, sliding-window attention has approximately:

O(S)

connections.

BigBird-style attention also keeps the number of connections much smaller than dense attention by using local, global, and random blocks.

This explains the large memory difference at longer sequence lengths.

## 10. TinyShakespeare Experiment

I also used the three attention patterns in a small character-level GPT trained on TinyShakespeare.

The model has:

- 2 transformer layers
- 4 attention heads
- Embedding dimension of 128
- Context length of 128
- Batch size of 32

I trained the model separately using dense, sliding-window, and BigBird-style attention.

Training and validation losses were recorded and plotted in:

quality_loss_curves.png

The purpose of this experiment was to see how restricting the attention pattern affects the model's ability to learn.

## 11. Information Trade-Off

Dense attention gives every token access to all previous tokens.

Sliding-window attention focuses mainly on local information, but this can make long-range relationships harder to capture.

BigBird-style attention tries to balance local and long-range information by adding global and random connections.

The global connections are useful because they provide a way for information from distant parts of the sequence to interact.

## 12. Limitations

The main limitation of this project is that the sparse attention implementation uses Python loops.

Because of this, the benchmark does not represent the performance of an optimized sparse attention kernel.

The GPT experiment is also relatively small, so the results should be treated as an experiment rather than a general comparison of all sparse attention methods.

## 13. Conclusion

This project helped me understand that having fewer attention connections does not automatically make an implementation faster.

In my benchmark, sparse attention used much less GPU memory, especially for longer sequences. However, it was slower because of the Python-level loops in my implementation.

This showed the trade-off between dense and sparse attention.

Dense attention gives every token access to the full context and is highly optimized on GPUs.

Sparse attention can save a lot of memory by reducing the number of connections, but achieving an actual speed improvement requires a more optimized implementation.

The main thing I learned is that the choice of which attention connections to keep is important because it affects both the computational cost and the information available to the model.
