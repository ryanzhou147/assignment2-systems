import torch
from cs336_systems.benchmarking import benchmark_transformer

# (a) Benchmark your attention implementation at different scales. Write a script that will:
# (a) Fix the batch size to 8 and don’t use multihead attention (i.e. remove the head dimension).
# (b) Iterate through the cartesian product of [16, 32, 64, 128] for the head embedding dimension dmodel, and [256, 1024, 4096, 8192, 16384] for the sequence length.
# (c) Create random inputs Q, K, V for the appropriate size.
# (d) Time 100 forward passes through attention using the inputs.
# (e) Measure how much memory is in use before the backward pass starts, and time 100 backward
# passes.
# (f) Make sure to warm up, and to call torch.cuda.synchronize() after each forward/backward
# pass.
# Report the timings (or out-of-memory errors) you get for these configurations. At what size do
# you get out-of-memory errors? Do the accounting for the memory usage of attention in one of the
# smallest configurations you find that runs out of memory (you can use the equations for memory
# usage of Transformers from Assignment 1). How does the memory saved for backward change
# with the sequence length? What would you do to eliminate this memory cost?
# Deliverable: A table with your timings, your working out for the memory usage, and a 1-2
# paragraph response.

import timeit

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    head_dims = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]

    for d_model in head_dims:
        for seq_len in seq_lengths:
            print(f"\nBenchmarking d_model={d_model}, seq_len={seq_len}")
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                from cs336_basics.transformer.multihead_self_attention import MultiHeadSelfAttention
                mha = MultiHeadSelfAttention(d_model=d_model, num_heads=1, device=device).to(device)
                Q = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)

                # Warmup
                for _ in range(10):
                    out = mha(Q)
                    out.mean().backward()
                    mha.zero_grad()
                torch.cuda.synchronize()

                # Time forward passes
                torch.cuda.synchronize()
                start = timeit.default_timer()
                for _ in range(100):
                    out = mha(Q)
                    torch.cuda.synchronize()
                forward_time = timeit.default_timer() - start
                print(f"Forward (100 passes): {forward_time:.4f}s")

                # Measure memory after forward, before backward
                mem_before_backward = torch.cuda.max_memory_allocated()
                print(f"Memory before backward: {mem_before_backward / (1024**2):.2f} MB")

                # Time backward passes
                torch.cuda.synchronize()
                start = timeit.default_timer()
                for _ in range(100):
                    out = mha(Q)
                    torch.cuda.synchronize()
                    out.mean().backward()
                    torch.cuda.synchronize()
                    mha.zero_grad()
                backward_time = timeit.default_timer() - start
                print(f"Forward+Backward (100 passes): {backward_time:.4f}s")

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"OOM for d_model={d_model}, seq_len={seq_len}")
                    torch.cuda.empty_cache()
                else:
                    raise